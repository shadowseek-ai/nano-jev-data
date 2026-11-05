"""把后训练集补齐到“每语言每垂类 >= N 条记录（每条 3 题）”。

口径（2026-11-05）：后训练集 >= 300 题 / 组 = >= 100 条记录 / 组。

流程与预训练一致：简体中文生成 -> 对齐翻译（繁体 OpenCC、英文 LLM）-> 写入
``train/posttrain/00-llm.<category>.<lang>.jsonl``，记录 ``style`` 为 ``llm-post``。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import topup_pretrain as core  # noqa: E402

POST = ROOT / "train" / "posttrain"
LANGS = core.LANGS


def count_post() -> Counter:
    counter: Counter = Counter()
    for path in sorted(POST.glob("*.jsonl")):
        category, lang = path.stem.split(".")[1], path.stem.split(".")[2]
        for line in path.open(encoding="utf-8"):
            if line.strip():
                counter[(category, lang)] += 1
    return counter


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="补齐后训练集到每语言每垂类 >= N 条记录")
    parser.add_argument("--min-records", type=int, default=100)
    parser.add_argument("--categories", nargs="*", default=None)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    specs = core.load_specs()
    counts = count_post()
    gaps: dict[str, dict[str, int]] = {}
    for category in sorted(specs):
        if args.categories and category not in args.categories:
            continue
        row = {lang: max(0, args.min_records - counts[(category, lang)]) for lang in LANGS}
        if any(row.values()):
            gaps[category] = row
    if not gaps:
        core.log("后训练集已达标")
        return 0
    core.log(f"后训练缺口：{json.dumps(gaps, ensure_ascii=False)}")

    existing: dict[tuple[str, str], set[str]] = {}
    indexes: dict[tuple[str, str], int] = {}
    lines: dict[tuple[str, str], list[str]] = {}
    for category in gaps:
        for lang in LANGS:
            path = POST / f"00-llm.{category}.{lang}.jsonl"
            existing[(category, lang)] = set()
            indexes[(category, lang)] = 0
            lines[(category, lang)] = []
            if not path.exists():
                continue
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                lines[(category, lang)].append(line)
                record = json.loads(line)
                existing[(category, lang)].add(core.fingerprint(record["state"]))
                match = re.search(r"-(\d+)$", record["id"])
                if match:
                    indexes[(category, lang)] = max(indexes[(category, lang)], int(match.group(1)))

    for category, row in gaps.items():
        spec = specs[category]
        tasks = spec["tasks"]
        need = max(row.values())
        produced: list[dict] = []
        attempt = 0
        while len(produced) < min(4, need) and attempt < 4:
            task = tasks[(attempt + len(produced)) % len(tasks)]
            prompt = core.build_generation_prompt(category, task, [], attempt)
            attempt += 1
            try:
                text = core.call_api(
                    [
                        {"role": "system", "content": "你是资深业务数据撰写员。只输出 JSON。"},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=1.0,
                    max_tokens=12000,
                )
            except Exception as exc:  # noqa: BLE001
                core.log(f"生成失败 {category}：{type(exc).__name__} {str(exc)[:80]}")
                continue
            for payload in core.parse_json_records(text):
                state = str(payload.get("state") or "")
                labels = payload.get("labels")
                ok, _ = core.validate_state(state, core.SOURCE, task)
                if not ok:
                    continue
                ok, _ = core.validate_labels(labels, task)
                if not ok:
                    continue
                produced.append({"task": task["id"], "state": state.strip(), "labels": labels})
        core.log(f"{category}: 生成 {len(produced)} 条后训练源记录")

        for source in produced:
            digest = core.fingerprint(source["state"])
            for lang in LANGS:
                if digest in existing[(category, lang)]:
                    continue
                if lang == core.SOURCE:
                    text = source["state"]
                elif lang == "zh-Hant":
                    text = core.S2T.convert(source["state"])
                    ok, _ = core.translation_validate(text, source["state"], lang)
                    if not ok:
                        continue
                else:
                    fake = {
                        "source_id": f"{category}-{lang}-{len(produced)}",
                        "state": source["state"],
                        "category": category,
                        "task": source["task"],
                        "labels": source["labels"],
                    }
                    text = core.translate_with_repair(fake, lang, attempts=3)
                    if not text:
                        continue
                task = next(item for item in spec["tasks"] if item["id"] == source["task"])
                variant = hashlib.blake2b(text.encode(), digest_size=2).digest()[0] % 2
                indexes[(category, lang)] += 1
                record = {
                    "id": f"train-llm-{category}-{lang.lower()}-{indexes[(category, lang)]:06d}",
                    "split": "train",
                    "task": source["task"],
                    "category": category,
                    "lang": lang,
                    "style": "llm-post",
                    "state": text,
                    "questions": core.question_schema(spec, task, lang, variant),
                    "labels": source["labels"],
                }
                lines[(category, lang)].append(json.dumps(record, ensure_ascii=False))
                existing[(category, lang)].add(core.fingerprint(text))

    if args.dry_run:
        core.log("dry-run：不写文件")
        return 0
    for (category, lang), rows in lines.items():
        path = POST / f"00-llm.{category}.{lang}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            for line in rows:
                handle.write(line + "\n")
    after = count_post()
    low = {f"{c}/{l}": after[(c, l)] for c in specs for l in LANGS if after[(c, l)] < args.min_records}
    core.log(f"后训练集写入完成，未达标 {len(low)} 组：{low}")
    return 0 if not low else 1


if __name__ == "__main__":
    raise SystemExit(main())
