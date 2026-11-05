"""最终数据门禁：schema、标签域、语言纯度、重复、跨 split 交叉与 3000 配额。"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import topup_pretrain as core  # noqa: E402

TARGET = 3000
LANGS = ("zh-Hans", "zh-Hant", "en")
CJK = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")


def digest(record: dict) -> str:
    payload = record["state"] + "\x00" + json.dumps(record["questions"], sort_keys=True, ensure_ascii=False)
    return hashlib.blake2b(payload.encode(), digest_size=8).hexdigest()


def main() -> int:
    specs = core.load_specs()
    problems: list[str] = []
    counts: Counter = Counter()
    split_digests: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for relative in ("train", "train/posttrain", "test", "dev"):
        folder = ROOT / relative
        if not folder.exists():
            continue
        for path in sorted(folder.glob("*.jsonl")):
            category, lang = path.stem.split(".")[1], path.stem.split(".")[2]
            if relative == "train":
                counts[(category, lang)] += 0
            seen: set[str] = set()
            for line_no, line in enumerate(path.open(encoding="utf-8"), start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                counts[(category, lang)] += 1 if relative == "train" else 0
                task = next((t for t in specs[category]["tasks"] if t["id"] == record.get("task")), None)
                if task is None:
                    problems.append(f"{relative}/{path.name}:{line_no} unknown task")
                    continue
                if set(record["labels"]) != set(task["questions"]):
                    problems.append(f"{relative}/{path.name}:{line_no} label keys")
                for name, question in task["questions"].items():
                    value = record["labels"].get(name)
                    if question["type"] == "noul":
                        if value not in (0, 1) or isinstance(value, bool):
                            problems.append(f"{relative}/{path.name}:{line_no} {name} noul")
                    elif question["type"] == "choice":
                        if value not in question["criteria"][lang]:
                            problems.append(f"{relative}/{path.name}:{line_no} {name} choice")
                    else:
                        levels = question["criteria"][lang]
                        if not isinstance(value, int) or isinstance(value, bool) or not 0 <= value < len(levels):
                            problems.append(f"{relative}/{path.name}:{line_no} {name} score")
                state = record["state"]
                if lang == "en" and CJK.search(state):
                    problems.append(f"{relative}/{path.name}:{line_no} cjk-in-en")
                if lang == "zh-Hans" and not core.is_simplified(state):
                    problems.append(f"{relative}/{path.name}:{line_no} not-simplified")
                if lang == "zh-Hant" and core.T2S.convert(state) == state:
                    problems.append(f"{relative}/{path.name}:{line_no} not-traditional")
                key = digest(record)
                if key in seen:
                    problems.append(f"{relative}/{path.name}:{line_no} duplicate")
                seen.add(key)
                split_digests[category][relative].add(key)

    for category in sorted(specs):
        for lang in LANGS:
            if counts[(category, lang)] < TARGET:
                problems.append(f"quota {category}/{lang}={counts[(category, lang)]} < {TARGET}")
        train = split_digests[category].get("train", set())
        for other in ("test", "dev", "train/posttrain"):
            overlap = train & split_digests[category].get(other, set())
            if overlap:
                problems.append(f"overlap {category}: train∩{other}={len(overlap)}")

    print(f"{'category':<26}{'zh-Hans':>9}{'zh-Hant':>9}{'en':>9}")
    for category in sorted(specs):
        print(f"{category:<26}{counts[(category, 'zh-Hans')]:>9}{counts[(category, 'zh-Hant')]:>9}{counts[(category, 'en')]:>9}")
    print(f"\nproblems={len(problems)}")
    for item in problems[:40]:
        print("  -", item)
    return 0 if not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())
