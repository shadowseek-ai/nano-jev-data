"""删除跨 split 重复的 case（整案三语一起删），用于消除 train/test、dev/test 的 case_id 交叉。"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ("train", "train/posttrain", "test", "dev")


def main() -> int:
    ids: dict[str, set[str]] = {}
    for relative in SPLITS:
        folder = ROOT / relative
        found = set()
        for path in folder.glob("00-llm.*.jsonl"):
            for line in path.open(encoding="utf-8"):
                if line.strip():
                    found.add(json.loads(line)["case_id"])
        ids[relative] = found
    leak = (ids["train/posttrain"] & ids["test"]) | (ids["train/posttrain"] & ids["dev"]) | (ids["dev"] & ids["test"])
    print("待删除 case:", sorted(leak))
    # 从 test 与 dev 中删除（训练侧优先保留）
    for relative in ("test", "dev"):
        drop = leak & ids[relative]
        if not drop:
            continue
        for path in (ROOT / relative).glob("00-llm.*.jsonl"):
            kept = []
            removed = 0
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                if json.loads(line)["case_id"] in drop:
                    removed += 1
                    continue
                kept.append(line)
            if removed:
                path.write_text("\n".join(kept) + "\n", encoding="utf-8", newline="\n")
                print(f"  {relative}/{path.name}: -{removed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
