"""旧数据改造：渲染占位符 + 补 case_id / question_lang / source_lang。

旧数据是三语言各自独立生成的，彼此没有同案情对应关系，因此：
- ``case_id`` 逐条唯一（``<category>-legacy-<n>``），不参与混合分片派生；
- ``question_lang == lang``，``source_lang == lang``（案情源语言即自身语言）；
- ``state`` 中所有 ``{date}`` / ``{ticket}`` / ``{case_id}`` 等模板占位符渲染成真实值。

单进程串行执行（1 并发），不与生成/翻译抢额度。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import case_meta  # noqa: E402

SPLIT_DIRS = ("train", "train/posttrain", "test", "dev")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default=str(ROOT))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    root = Path(args.dir)

    counters: Counter = Counter()
    stats: Counter = Counter()
    unknown: Counter = Counter()
    for relative in SPLIT_DIRS:
        folder = root / relative
        if not folder.exists():
            continue
        for path in sorted(folder.glob("*.jsonl")):
            lines: list[str] = []
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                record = json.loads(line)
                category = record["category"]
                counters[category] += 1
                case_id = record.get("case_id") or f"{category}-legacy-{counters[category]:06d}"
                if case_meta.residual_placeholders(record["state"]):
                    rendered, lost = case_meta.render_state(record["state"], case_id)
                    unknown.update(lost)
                    record["state"] = rendered
                    stats["rendered"] += 1
                record["case_id"] = case_id
                record.setdefault("question_lang", record["lang"])
                record.setdefault("source_lang", record["lang"])
                if case_meta.residual_placeholders(record["state"]):
                    stats["residual"] += 1
                stats["records"] += 1
                lines.append(json.dumps(record, ensure_ascii=False))
            stats[f"files"] += 1
            if not args.dry_run:
                path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")

    print(f"records={stats['records']} files={stats['files']} rendered={stats['rendered']} residual={stats['residual']}")
    if unknown:
        print("未知占位符：", dict(unknown))
    print("case_id 前缀示例：", [f"{c}-legacy-{n:06d}" for c, n in list(counters.items())[:3]])
    return 0 if stats["residual"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
