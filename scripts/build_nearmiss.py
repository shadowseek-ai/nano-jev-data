"""把干扰项落成近似反例子集。

对每条有干扰项的记录，复制原记录并把干扰项以 `near_miss` 为 key 加进 choice 题的
``criteria``；``labels`` 完全不变（正确选项仍是原 key），因此标签不变性成立。

输出：``train/nearmiss/30-nearmiss.<category>.<lang>.jsonl``
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import case_meta  # noqa: E402

LANGS = ("zh-Hans", "zh-Hant", "en")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", nargs="*", default=["train"])
    parser.add_argument("--per-category", type=int, default=500)
    parser.add_argument("--work", default=str(ROOT / "artifacts" / "llm" / "nearmiss"))
    args = parser.parse_args(argv)

    cache_path = Path(args.work) / "distractors.jsonl"
    if not cache_path.exists():
        print("没有干扰项缓存")
        return 1
    by_case: dict[str, dict] = {}
    for line in cache_path.open(encoding="utf-8"):
        if line.strip():
            row = json.loads(line)
            by_case[row["case_id"]] = row
    print(f"干扰项缓存 {len(by_case)} 条")

    for split in args.splits:
        folder = ROOT / split
        records: dict[str, dict] = defaultdict(dict)  # case_id -> {lang: record}
        for path in sorted(folder.glob("00-llm.*.jsonl")):
            category, lang = path.stem.split(".")[1], path.stem.split(".")[2]
            for line in path.open(encoding="utf-8"):
                if not line.strip():
                    continue
                record = json.loads(line)
                records[record["case_id"]][lang] = record
        # 按垂类限额
        per_cat: Counter = Counter()
        out_dir = folder / "nearmiss"
        buckets: dict[tuple[str, str], list[str]] = defaultdict(list)
        for case_id, by_lang in records.items():
            info = by_case.get(case_id)
            if info is None or "zh-Hans" not in by_lang:
                continue
            category = info["category"]
            if per_cat[category] >= args.per_category:
                continue
            key = info["key"]
            ok = True
            for lang in LANGS:
                record = by_lang.get(lang)
                if record is None or key not in record["questions"]:
                    ok = False
                    break
                question = record["questions"][key]
                if question["type"] != "choice" or record["labels"].get(key) != info["gold"]:
                    ok = False
                    break
            if not ok:
                continue
            per_cat[category] += 1
            for lang in LANGS:
                record = dict(by_lang[lang])
                questions = {name: dict(value) for name, value in record["questions"].items()}
                question = questions[key]
                criteria = dict(question["criteria"])
                near_miss = info["texts"][lang].strip()
                if not near_miss or near_miss in criteria.values():
                    raise ValueError(f"{case_id}/{lang}: 干扰项为空或与已有选项重复")
                criteria["near_miss"] = near_miss
                question["criteria"] = criteria
                questions[key] = question
                record["questions"] = questions
                record["style"] = "near-miss"
                record["id"] = record["id"].replace(f"{split}-", f"{split}-nearmiss-", 1)
                state, _ = case_meta.render_state(record["state"], case_id, lang)
                record["state"] = state
                assert record["labels"].get(key) in question["criteria"], "标签必须仍在选项内"
                buckets[(category, lang)].append(json.dumps(record, ensure_ascii=False))
        if not buckets:
            print(f"{split}: 无可用案例")
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        for old in out_dir.glob("30-nearmiss.*.jsonl"):
            old.unlink()
        for (category, lang), rows in buckets.items():
            path = out_dir / f"30-nearmiss.{category}.{lang}.jsonl"
            path.write_text("\n".join(rows) + "\n", encoding="utf-8", newline="\n")
        total = sum(len(rows) for rows in buckets.values())
        print(f"{split}: 写出 {total} 条，覆盖 {len(per_cat)} 个垂类，最小 {min(per_cat.values())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
