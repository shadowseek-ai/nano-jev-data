"""删除同一文件内重复（同 case_id 出现多次）的记录，只保留第一次出现。"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import hashlib

ROOT = Path(__file__).resolve().parents[1]


def digest(record: dict) -> str:
    payload = record["state"] + "\x00" + json.dumps(record["questions"], sort_keys=True, ensure_ascii=False)
    return hashlib.blake2b(payload.encode(), digest_size=8).hexdigest()


def main() -> int:
    removed = Counter()
    for folder in ("train", "test", "dev"):
        for path in sorted((ROOT / folder).rglob("*.jsonl")):
            lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
            seen: set[str] = set()
            seen_case: set[str] = set()
            kept = []
            for line in lines:
                record = json.loads(line)
                fingerprint = digest(record)
                key = (record["case_id"], record["lang"], record.get("style"), record.get("pair_role"))
                if fingerprint in seen or key in seen_case:
                    removed[f"{path.relative_to(ROOT).as_posix()}"] += 1
                    continue
                seen.add(fingerprint)
                seen_case.add(key)
                kept.append(line)
            if len(kept) != len(lines):
                path.write_text("\n".join(kept) + "\n", encoding="utf-8", newline="\n")
    total = sum(removed.values())
    print("removed", total, dict(list(removed.items())[:6]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
