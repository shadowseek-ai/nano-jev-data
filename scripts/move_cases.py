"""把指定 case 从 train 整体搬到 test（不是复制），用于补齐 test 缺口且不产生跨 split 泄漏。"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LANGS = ("zh-Hans", "zh-Hant", "en")


def read(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
                    encoding="utf-8", newline="\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", required=True)
    parser.add_argument("--count", type=int, required=True)
    args = parser.parse_args(argv)

    source = ROOT / "train"
    target = ROOT / "test"
    moved = 0
    for lang in LANGS:
        src_path = source / f"00-llm.{args.category}.{lang}.jsonl"
        dst_path = target / f"00-llm.{args.category}.{lang}.jsonl"
        src_rows = read(src_path)
        dst_rows = read(dst_path)
        existing = {row["case_id"] for row in dst_rows}
        max_index = 0
        for row in dst_rows:
            match = re.search(r"-(\d+)$", row["id"])
            if match:
                max_index = max(max_index, int(match.group(1)))
        keep, taken = [], []
        for row in src_rows:
            if len(taken) < args.count and row["case_id"] not in existing:
                taken.append(row)
                continue
            keep.append(row)
        for row in taken:
            max_index += 1
            row = dict(row)
            row["id"] = f"test-llm-{args.category}-{lang.lower()}-{max_index:06d}"
            row["split"] = "test"
            dst_rows.append(row)
        write(src_path, keep)
        write(dst_path, dst_rows)
        moved = max(moved, len(taken))
        print(f"{lang}: 移出 {len(taken)}，train={len(keep)} test={len(dst_rows)}")
    return 0 if moved else 1


if __name__ == "__main__":
    raise SystemExit(main())
