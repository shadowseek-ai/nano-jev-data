"""对齐翻译的小批次工具：给子代理取 4 条待译记录，再把它写回的译文校验入库。

子代理只负责翻译文本；占位符/数字一致性、繁简、条数对齐都在这里校验，
不合格的条目会被拒绝并保持"未翻译"状态，等待重试。

用法
----
    py artifacts/llm/aligned_translate.py next --n 4            # 产出 _pending.json
    py artifacts/llm/aligned_translate.py merge --file out.json # 校验并写回
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from opencc import OpenCC

ROOT = Path(__file__).resolve().parents[2]
API_DIR = ROOT / "artifacts" / "llm" / "seeds_api"
PENDING = ROOT / "artifacts" / "llm" / "_pending_translate.json"
TARGETS = ("zh-Hant", "en")
SOURCE = "zh-Hans"
T2S = OpenCC("t2s")
CJK = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
PLACEHOLDER = re.compile(r"\{[a-z_]+\}")
NUMBER = re.compile(r"\d+(?:[.,]\d+)*")


def source_records(category: str, task: str) -> list[dict]:
    path = API_DIR / category / f"{task}.{SOURCE}.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def translated_count(category: str, task: str, target: str) -> int:
    path = API_DIR / category / f"{task}.{target}.jsonl"
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def cmd_next(args: argparse.Namespace) -> int:
    groups = []
    for path in sorted(API_DIR.rglob(f"*.{SOURCE}.jsonl")):
        category = path.parent.name
        task = path.name[: -(len(SOURCE) + 7)]  # 去掉 ".zh-Hans.jsonl"
        records = source_records(category, task)
        done_hant = translated_count(category, task, "zh-Hant")
        done_en = translated_count(category, task, "en")
        backlog = min(len(records) - done_hant, len(records) - done_en)
        if backlog > 0:
            groups.append((backlog, category, task, records, done_hant, done_en))
    groups.sort(reverse=True)
    picked = []
    for backlog, category, task, records, done_hant, done_en in groups:
        start = max(done_hant, done_en)
        for offset in range(min(args.n - len(picked), backlog)):
            index = start + offset
            if index >= len(records):
                break
            record = records[index]
            picked.append({
                "category": category,
                "task": task,
                "index": index,
                "state": record["state"],
                "labels": record["labels"],
            })
        if len(picked) >= args.n:
            break
    PENDING.write_text(json.dumps({"items": picked}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({"pending": len(picked), "file": str(PENDING.relative_to(ROOT))}, ensure_ascii=True))
    return 0


def validate(text: str, source: str, target: str) -> tuple[bool, str]:
    value = (text or "").strip()
    if not 100 <= len(value) <= 800:
        return False, "length"
    if set(PLACEHOLDER.findall(value)) != set(PLACEHOLDER.findall(source)):
        return False, "placeholder"
    remaining = list(NUMBER.findall(value))
    for token in NUMBER.findall(source):
        if token in remaining:
            remaining.remove(token)
        else:
            return False, "number"
    if target == "en":
        if CJK.search(value):
            return False, "cjk-in-en"
        if len(value.split()) < 12:
            return False, "too-short-en"
    elif target == "zh-Hant" and T2S.convert(value) == value:
        return False, "not-traditional"
    return True, "ok"


def cmd_merge(args: argparse.Namespace) -> int:
    accepted = 0
    rejected: list[dict] = []
    payload = json.loads(Path(args.file).read_text(encoding="utf-8"))
    for item in payload.get("items", []):
        category, task, index = item.get("category"), item.get("task"), item.get("index")
        records = source_records(category, task)
        if not (isinstance(index, int) and 0 <= index < len(records)):
            rejected.append({"item": item, "reason": "bad-index"})
            continue
        source = records[index]
        for target in TARGETS:
            expected = translated_count(category, task, target)
            if expected != index:
                rejected.append({"task": task, "index": index, "target": target, "reason": f"order-mismatch({expected})"})
                continue
            ok, reason = validate(item.get(target, ""), source["state"], target)
            if not ok:
                rejected.append({"task": task, "index": index, "target": target, "reason": reason})
                continue
            path = API_DIR / category / f"{task}.{target}.jsonl"
            with path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(json.dumps({
                    "task": task,
                    "lang": target,
                    "state": item[target].strip(),
                    "labels": source["labels"],
                }, ensure_ascii=False) + "\n")
            accepted += 1
    print(json.dumps({"accepted": accepted, "rejected": rejected[:10], "rejected_count": len(rejected)},
                     ensure_ascii=True))
    return 0 if not rejected else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    nxt = sub.add_parser("next")
    nxt.add_argument("--n", type=int, default=4)
    nxt.set_defaults(func=cmd_next)
    mrg = sub.add_parser("merge")
    mrg.add_argument("--file", required=True)
    mrg.set_defaults(func=cmd_merge)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
