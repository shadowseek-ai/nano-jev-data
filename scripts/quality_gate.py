"""语料质量门禁：schema / 取值域 / 长度 / 语言纯度 / 去重 / 近似重复 / 标签泄漏 / 标签均衡。

用法：
    py artifacts/llm/quality_gate.py                 # 审计所有种子（legacy + api）
    py artifacts/llm/quality_gate.py --split-report  # 额外给出每 split 的交叉污染

输出：artifacts/llm/quality_report.json（UTF-8），控制台只打印 ASCII 摘要。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from opencc import OpenCC

ROOT = Path(__file__).resolve().parents[2]
SPECS = ROOT / "artifacts" / "llm" / "specs"
SEED_DIRS = (ROOT / "artifacts" / "llm" / "seeds", ROOT / "artifacts" / "llm" / "seeds_api")
DATA = ROOT / "data"
LANGS = ("zh-Hans", "zh-Hant", "en")
CJK = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
T2S, S2T = OpenCC("t2s"), OpenCC("s2t")


def load_specs() -> dict:
    out = {}
    for path in SPECS.glob("*.json"):
        if path.stem.startswith("_"):
            continue
        out[path.stem] = json.loads(path.read_text(encoding="utf-8"))
    return out


def domain_of(question: dict) -> set:
    qtype = question["type"]
    if qtype == "noul":
        return {0, 1}
    criteria = question["criteria"]["en"]
    return set(criteria.keys()) if qtype == "choice" else set(range(len(criteria)))


def shingles(text: str, size: int = 5) -> set[str]:
    norm = re.sub(r"\s+", "", text)
    return {norm[i : i + size] for i in range(max(len(norm) - size + 1, 0))}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def iter_seeds():
    for folder in SEED_DIRS:
        for path in sorted(folder.rglob("*.jsonl")):
            category = path.parent.name
            for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
                if not line.strip():
                    continue
                try:
                    yield category, path, line_no, json.loads(line)
                except json.JSONDecodeError:
                    yield category, path, line_no, None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--near-dup", type=float, default=0.80)
    args = parser.parse_args(argv)

    specs = load_specs()
    problems: list[dict] = []
    seen: dict[tuple[str, str], dict[str, str]] = defaultdict(dict)
    buckets: dict[tuple[str, str], list[tuple[str, set[str]]]] = defaultdict(list)
    label_counts: dict[tuple[str, str, str], Counter] = defaultdict(Counter)
    total = 0
    per_lang = Counter()

    for category, path, line_no, seed in iter_seeds():
        where = f"{path.relative_to(ROOT)}:{line_no}"
        total += 1
        if seed is None:
            problems.append({"where": where, "kind": "bad-json"})
            continue
        task_id, lang, state = seed.get("task"), seed.get("lang"), seed.get("state", "")
        labels = seed.get("labels")
        spec = specs.get(category, {})
        task = next((t for t in spec.get("tasks", []) if t["id"] == task_id), None)
        if task is None:
            problems.append({"where": where, "kind": "unknown-task", "task": task_id})
            continue
        if lang not in LANGS:
            # 8 语言时期的遗留种子（ru/ko/ja/ar/bo）：装配时被 ALLOWED_LANGS 跳过，不算缺陷
            skipped_foreign += 1
            continue
        per_lang[lang] += 1
        if not isinstance(state, str) or not 140 <= len(state) <= 700:
            problems.append({"where": where, "kind": "length", "len": len(state or "")})
        if not isinstance(labels, dict) or set(labels) != set(task["questions"]):
            problems.append({"where": where, "kind": "label-keys"})
            continue
        for name, question in task["questions"].items():
            if labels[name] not in domain_of(question):
                problems.append({"where": where, "kind": "label-domain", "q": name, "value": labels[name]})
            label_counts[(task_id, lang, name)][str(labels[name])] += 1
        if lang == "zh-Hans" and T2S.convert(state) != state:
            problems.append({"where": where, "kind": "hans-not-simplified"})
        if lang == "zh-Hant" and T2S.convert(state) == state:
            # t2s 后完全不变 => 全文没有繁体字，标成 zh-Hant 是错标
            problems.append({"where": where, "kind": "hant-is-simplified"})
        if lang == "en" and CJK.search(state):
            problems.append({"where": where, "kind": "cjk-in-en"})
        for name in task["questions"]:
            if re.search(rf"(?<![A-Za-z]){re.escape(name)}(?![A-Za-z])", state):
                problems.append({"where": where, "kind": "field-name-leak", "q": name})
                break
        digest = hashlib.blake2b(re.sub(r"\s+", "", state).lower().encode(), digest_size=8).hexdigest()
        if digest in seen[(category, lang)]:
            problems.append({"where": where, "kind": "exact-dup", "first": seen[(category, lang)][digest]})
            continue
        seen[(category, lang)][digest] = where
        buckets[(category, lang)].append((where, shingles(state)))

    near_dups = 0
    for key, items in buckets.items():
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                if jaccard(items[i][1], items[j][1]) >= args.near_dup:
                    near_dups += 1
                    if near_dups <= 30:
                        problems.append({"where": items[j][0], "kind": "near-dup", "first": items[i][0]})

    imbalanced = []
    for (task_id, lang, name), counter in label_counts.items():
        total_q = sum(counter.values())
        if total_q < 10:
            continue
        share = {k: v / total_q for k, v in counter.items()}
        smallest = min(share.values())
        if smallest < 0.05:
            imbalanced.append({
                "task": task_id, "lang": lang, "question": name,
                "n": total_q, "min_share": round(smallest, 3),
                "counts": dict(counter),
            })

    summary = {
        "total_records": total,
        "per_lang": dict(per_lang),
        "problems": len(problems),
        "by_kind": dict(Counter(p["kind"] for p in problems)),
        "near_dups": near_dups,
        "imbalanced_questions": imbalanced,
    }
    out = ROOT / "artifacts" / "llm" / "quality_report.json"
    out.write_text(json.dumps({"summary": summary, "problems": problems[:500]}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=True, indent=1))
    return 0 if summary["problems"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
