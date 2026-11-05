"""审计 topup 种子：标签域、语言、占位符/数字、三语对齐与重复。"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import topup_pretrain as core  # noqa: E402


def main() -> int:
    work = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "artifacts" / "llm" / "topup"
    specs = core.load_specs()
    sources: dict[str, dict] = {}
    translations: dict[tuple[str, str], dict] = {}
    for line in (work / "seeds.jsonl").open(encoding="utf-8"):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row["kind"] == "source":
            sources[row["source_id"]] = row
    trans_path = work / "translations.jsonl"
    if trans_path.exists():
        for line in trans_path.open(encoding="utf-8"):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row["kind"] == "translation":
                translations[(row["source_id"], row["lang"])] = row

    problems: list[str] = []
    counts: Counter = Counter()
    tasks: Counter = Counter()
    labels: dict[tuple[str, str], Counter] = defaultdict(Counter)
    seen: dict[tuple[str, str], set[str]] = defaultdict(set)
    for source_id, row in sources.items():
        category, task_id = row["category"], row["task"]
        task = next(t for t in specs[category]["tasks"] if t["id"] == task_id)
        counts[(category, "zh-Hans")] += 1
        tasks[(category, task_id)] += 1
        ok, reason = core.validate_state(row["state"], "zh-Hans", task)
        if not ok:
            problems.append(f"{source_id} source {reason}")
        ok, reason = core.validate_labels(row["labels"], task)
        if not ok:
            problems.append(f"{source_id} labels {reason}")
        digest = core.fingerprint(row["state"])
        if digest in seen[(category, "zh-Hans")]:
            problems.append(f"{source_id} duplicate source")
        seen[(category, "zh-Hans")].add(digest)
        for name, value in row["labels"].items():
            labels[(category, name)][str(value)] += 1
        for target in ("zh-Hant", "en"):
            translated = translations.get((source_id, target))
            if translated is None:
                problems.append(f"{source_id} missing {target}")
                continue
            counts[(category, target)] += 1
            if translated["labels"] != row["labels"]:
                problems.append(f"{source_id} {target} labels mismatch")
            ok, reason = core.translation_validate(translated["state"], row["state"], target)
            if not ok:
                problems.append(f"{source_id} {target} {reason}")

    print(f"sources={len(sources)} translations={len(translations)}")
    print(f"{'category':<26}{'zh-Hans':>9}{'zh-Hant':>9}{'en':>9}")
    for category in sorted(specs):
        print(f"{category:<26}{counts[(category, 'zh-Hans')]:>9}{counts[(category, 'zh-Hant')]:>9}{counts[(category, 'en')]:>9}")
    print("\nper task:")
    for (category, task_id), value in sorted(tasks.items()):
        print(f"{category:<26}{task_id:<32}{value:>5}")
    print("\nlabel balance:")
    for (category, name), counter in sorted(labels.items()):
        print(f"{category:<26}{name:<24}{dict(counter)}")
    print(f"\nproblems={len(problems)}")
    for item in problems[:40]:
        print("  -", item)
    return 0 if not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())
