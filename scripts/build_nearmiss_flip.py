"""把翻盘样本落成近似反例子集（§3.2 修正版）。

每条样本 = 原案情 + 一个数字被改写 + 翻转后的标签：
- `case_id` 加 `-nm` 后缀，另有 `derived_from` 指回原案；
- 三语同步替换（数字与语言无关），要求 `from` 在每种语言的 state 里**恰好出现一次**；
- `labels` 与原 case 不同（已由生成阶段校验），`style = "near-miss"`。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LANGS = ("zh-Hans", "zh-Hant", "en")


def replace_once(text: str, old: str, new: str) -> str | None:
    pattern = re.compile(rf"(?<!\d){re.escape(old)}(?!\d)")
    if len(pattern.findall(text)) != 1:
        return None
    return pattern.sub(new, text, count=1)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", nargs="*", default=["train"])
    parser.add_argument("--per-category", type=int, default=500)
    parser.add_argument("--work", default=str(ROOT / "artifacts" / "llm" / "nearmiss2"))
    args = parser.parse_args(argv)

    cache_path = Path(args.work) / "flips.jsonl"
    if not cache_path.exists():
        print("没有翻盘缓存")
        return 1
    flips: dict[str, dict] = {}
    for line in cache_path.open(encoding="utf-8"):
        if line.strip():
            row = json.loads(line)
            flips.setdefault(row["case_id"], row)
    print(f"翻盘缓存 {len(flips)} 条")

    for split in args.splits:
        folder = ROOT / split
        records: dict[str, dict] = defaultdict(dict)
        for path in sorted(folder.glob("00-llm.*.jsonl")):
            category, lang = path.stem.split(".")[1], path.stem.split(".")[2]
            for line in path.open(encoding="utf-8"):
                if not line.strip():
                    continue
                record = json.loads(line)
                records[record["case_id"]][lang] = record

        per_cat: Counter = Counter()
        buckets: dict[tuple[str, str], list[str]] = defaultdict(list)
        stats: Counter = Counter()
        for case_id, by_lang in records.items():
            flip = flips.get(case_id)
            if flip is None or "zh-Hans" not in by_lang:
                continue
            category = flip["category"]
            if per_cat[category] >= args.per_category:
                continue
            new_states = {}
            for lang in LANGS:
                record = by_lang.get(lang)
                if record is None:
                    break
                replaced = replace_once(record["state"], flip["from"], flip["to"])
                if replaced is None:
                    break
                new_states[lang] = (replaced, record)
            if len(new_states) != len(LANGS):
                stats["lang-mismatch"] += 1
                continue
            labels = {name: int(v) if isinstance(v, int) else v for name, v in flip["labels"].items()}
            per_cat[category] += 1
            for lang, (state, record) in new_states.items():
                derived = dict(record)
                derived["id"] = record["id"].replace(f"{split}-", f"{split}-nearmiss-", 1)
                derived["split"] = record.get("split", split)
                derived["case_id"] = f"{case_id}-nm"
                derived["derived_from"] = case_id
                derived["style"] = "near-miss"
                derived["state"] = state
                derived["labels"] = labels
                buckets[(category, lang)].append(json.dumps(derived, ensure_ascii=False))
        out_dir = folder / "nearmiss"
        if out_dir.exists():
            for old in out_dir.glob("*.jsonl"):
                old.unlink()
        if not buckets:
            print(f"{split}: 无可用近似反例")
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        for (category, lang), rows in buckets.items():
            path = out_dir / f"30-nearmiss.{category}.{lang}.jsonl"
            path.write_text("\n".join(rows) + "\n", encoding="utf-8", newline="\n")
        total = sum(len(rows) for rows in buckets.values())
        print(f"{split}: {total} 条，覆盖 {len(per_cat)} 个垂类，每垂类 {min(per_cat.values())}–{max(per_cat.values())}，跳过 {dict(stats)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
