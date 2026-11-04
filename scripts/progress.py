"""对齐翻译进度与剩余量估算。

统计口径：
- 源：zh-Hans 每个 (垂类, 任务) 的行数
- 译文：目标语言同组的行数
- 缺口 = max(0, 源 - 译文)，超量 = max(0, 译文 - 源)

用法：
    py artifacts/llm/progress.py --rate-en 1500 --rate-hant 2300
    py artifacts/llm/progress.py --rate-en 1500 --rate-hant 2300 --json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
API = ROOT / "artifacts" / "llm" / "seeds_api"
LANGS = ("zh-Hans", "zh-Hant", "en")
OTHER = ("zh-Hant", "en")


def counts() -> Counter:
    counter: Counter = Counter()
    for lang in LANGS:
        for path in API.rglob(f"*.{lang}.jsonl"):
            task = path.name[: -(len(lang) + 6)]
            counter[(path.parent.name, task, lang)] += sum(
                1 for line in path.open(encoding="utf-8") if line.strip()
            )
    return counter


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rate-en", type=float, default=0.0, help="英文每小时翻译条数")
    parser.add_argument("--rate-hant", type=float, default=0.0, help="繁体每小时翻译条数")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    counter = counts()
    groups = sorted({(key[0], key[1]) for key in counter})
    source_total = sum(counter[(cat, task, "zh-Hans")] for cat, task in groups)
    rates = {"en": args.rate_en, "zh-Hant": args.rate_hant}

    report = {"source_total": source_total, "groups": len(groups), "languages": {}}
    for lang in OTHER:
        done = sum(counter[(cat, task, lang)] for cat, task in groups)
        deficit = sum(max(0, counter[(cat, task, "zh-Hans")] - counter[(cat, task, lang)]) for cat, task in groups)
        surplus = sum(max(0, counter[(cat, task, lang)] - counter[(cat, task, "zh-Hans")]) for cat, task in groups)
        open_groups = sum(1 for cat, task in groups if counter[(cat, task, lang)] < counter[(cat, task, "zh-Hans")])
        rate = rates[lang]
        hours = deficit / rate if rate else None
        report["languages"][lang] = {
            "done": done,
            "source": source_total,
            "coverage": round(done / source_total * 100, 1) if source_total else 0,
            "deficit_rows": deficit,
            "surplus_rows": surplus,
            "open_groups": open_groups,
            "rate_per_hour": rate or None,
            "eta_hours": round(hours, 2) if hours else None,
        }

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=1))
        return 0

    print(f"源语言 zh-Hans 总量: {source_total}  组数: {len(groups)}")
    for lang, info in report["languages"].items():
        eta = f"{info['eta_hours']} 小时" if info["eta_hours"] else "未提供速率"
        print(
            f"{lang:8} 已译 {info['done']:6} ({info['coverage']:5.1f}%)  "
            f"缺口 {info['deficit_rows']:5} 行 / {info['open_groups']} 组  "
            f"超量 {info['surplus_rows']:4}  速率 {info['rate_per_hour'] or '-'} 条/小时  预计 {eta}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
