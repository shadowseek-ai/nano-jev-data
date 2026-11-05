"""核查三语言 × 13 垂类的分项配额（按记录数）。

口径（2026-11-05 需求）：
- 预训练：每垂类每语言 >= 3000 条记录（每条 1 个 state + 3 道题，即 >= 9000 题）
- 测试集：每垂类每语言 >=  500 条记录（>= 1500 题）
- 后训练：每垂类每语言 >=  300 条记录（>=  900 题）
- 开发集：每垂类每语言 >= 100 条记录（上一轮口径，作为参考项）

用法：python scripts/check_quota.py --dir .
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

LANGS = ("zh-Hans", "zh-Hant", "en")
TRAIN_MIN_RECORDS = 3000
TEST_MIN_RECORDS = 500
POST_MIN_RECORDS = 300
DEV_MIN_RECORDS = 100
QUESTIONS_PER_RECORD = 3


def count(folder: Path) -> Counter:
    counter: Counter = Counter()
    if not folder.exists():
        return counter
    for path in folder.glob("*.jsonl"):
        for line in path.open(encoding="utf-8"):
            if line.strip():
                record = json.loads(line)
                counter[(record["category"], record["lang"])] += 1
    return counter


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default=".")
    args = parser.parse_args(argv)
    root = Path(args.dir)
    train = count(root / "train")
    test = count(root / "test")
    post = count(root / "train" / "posttrain")
    dev = count(root / "dev")
    categories = sorted({key[0] for key in train})
    failures = []
    print(f"{'category':<24}{'lang':<9}{'train':>7}{'test':>7}{'post':>7}{'dev':>6}   （记录数，括号内为题数）")
    for category in categories:
        for lang in LANGS:
            tr = train[(category, lang)]
            te = test[(category, lang)]
            po = post[(category, lang)]
            dv = dev[(category, lang)]
            flags = []
            if tr < TRAIN_MIN_RECORDS:
                flags.append(f"train<{TRAIN_MIN_RECORDS}")
            if te < TEST_MIN_RECORDS:
                flags.append(f"test<{TEST_MIN_RECORDS}")
            if po < POST_MIN_RECORDS:
                flags.append(f"post<{POST_MIN_RECORDS}")
            print(
                f"{category:<24}{lang:<9}{tr:>7}{te:>7}{po:>7}{dv:>6}   "
                f"({tr * QUESTIONS_PER_RECORD}/{te * QUESTIONS_PER_RECORD}/{po * QUESTIONS_PER_RECORD})  {' '.join(flags)}"
            )
            if flags:
                failures.append((category, lang, flags))
    print()
    for lang in LANGS:
        total_dev = sum(dev[(category, lang)] for category in categories)
        print(f"dev total {lang}: {total_dev} (requirement: >=100 per language, informational)")
    worst_train = min((train[(c, l)] for c in categories for l in LANGS), default=0)
    worst_test = min((test[(c, l)] for c in categories for l in LANGS), default=0)
    worst_post = min((post[(c, l)] for c in categories for l in LANGS), default=0)
    print(
        f"categories={len(categories)} languages={len(LANGS)} violating={len(failures)} "
        f"min(train/test/post)={worst_train}/{worst_test}/{worst_post}"
    )
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
