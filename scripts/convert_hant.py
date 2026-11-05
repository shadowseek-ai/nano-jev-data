"""把 splits 种子的源记录转成繁体（OpenCC，纯本地，不占用翻译 API）。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import topup_pretrain as core  # noqa: E402


def main() -> int:
    work = ROOT / "artifacts" / "llm" / "splits"
    seeds = work / "seeds.jsonl"
    trans = work / "translations.jsonl"
    done = set()
    if trans.exists():
        for line in trans.open(encoding="utf-8"):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("kind") == "translation":
                done.add((row["source_id"], row["lang"]))
    added = 0
    with trans.open("a", encoding="utf-8", newline="\n") as handle:
        for line in seeds.open(encoding="utf-8"):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("kind") != "source":
                continue
            if (row["source_id"], "zh-Hant") in done:
                continue
            text = core.S2T.convert(row["state"])
            handle.write(json.dumps({
                "kind": "translation",
                "row_id": f"{row['source_id']}-zh-Hant",
                "source_id": row["source_id"],
                "category": row["category"],
                "task": row["task"],
                "lang": "zh-Hant",
                "state": text,
                "labels": row["labels"],
            }, ensure_ascii=False) + "\n")
            added += 1
    print(f"zh-Hant +{added}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
