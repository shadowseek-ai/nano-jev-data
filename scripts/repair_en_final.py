"""最后一档补译：逐条 LLM 修复（带缺失 token 回灌），只处理仍缺失的英文。"""

from __future__ import annotations

import argparse
import json
import sys
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import topup_pretrain as core  # noqa: E402

LOCK = threading.Lock()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", default=str(ROOT / "artifacts" / "llm" / "splits"))
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args(argv)
    work = Path(args.work)
    core.LOG_PATH = work / "repair_en_final.log"

    sources = {}
    for line in (work / "seeds.jsonl").open(encoding="utf-8"):
        if line.strip():
            row = json.loads(line)
            if row.get("kind") == "source":
                sources[row["source_id"]] = row
    done = set()
    for line in (work / "translations.jsonl").open(encoding="utf-8"):
        if line.strip():
            row = json.loads(line)
            if row.get("kind") == "translation":
                done.add((row["source_id"], row["lang"]))
    pending = [row for row in sources.values() if (row["source_id"], "en") not in done]
    core.log(f"源 {len(sources)}，仍需英文 {len(pending)}")
    if not pending:
        return 0

    trans_path = work / "translations.jsonl"
    stats: Counter = Counter()

    def run(row: dict) -> None:
        text = core.translate_with_repair(row, "en", attempts=4)
        if not text:
            with LOCK:
                stats["fail"] += 1
            return
        with LOCK:
            with trans_path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(json.dumps({
                    "kind": "translation",
                    "row_id": f"{row['source_id']}-en",
                    "source_id": row["source_id"],
                    "category": row["category"],
                    "task": row["task"],
                    "lang": "en",
                    "state": text.strip(),
                    "labels": row["labels"],
                }, ensure_ascii=False) + "\n")
            stats["ok"] += 1
            if stats["ok"] % 50 == 0:
                core.log(f"修复 +{stats['ok']}")

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(run, pending))
    core.log(f"最终 {dict(stats)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
