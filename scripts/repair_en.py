"""补译缺失的英文译文：Edge 放宽校验 -> LLM 批量兜底。

用于 ``topup_splits.py --translate`` 首轮校验不过的记录：先按“数值很多时允许漏 1 个”
放宽接受，仍不行再用 LLM（每请求 4 条）兜底。只处理缺失样本，不重复计费。
"""

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
import edge_translate  # noqa: E402
import topup_pretrain as core  # noqa: E402

LOCK = threading.Lock()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", default=str(ROOT / "artifacts" / "llm" / "splits"))
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args(argv)
    work = Path(args.work)
    core.LOG_PATH = work / "repair_en.log"

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
    core.log(f"源 {len(sources)}，英文缺失 {len(pending)}")
    if not pending:
        return 0

    trans_path = work / "translations.jsonl"
    stats: Counter = Counter()

    def save(row: dict, text: str) -> None:
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

    def relaxed_ok(row: dict, text: str) -> bool:
        ok, reason = core.translation_validate(text, row["state"], "en")
        if ok:
            return True
        if reason != "number":
            return False
        if set(core.PLACEHOLDER.findall(text)) != set(core.PLACEHOLDER.findall(row["state"])):
            return False
        if core.CJK.search(text):
            return False
        values = core.numeric_values(row["state"], "zh-Hans")
        missing = values - core.numeric_values(text, "en")
        return sum(values.values()) >= 8 and sum(missing.values()) <= 1

    # 1) Edge 放宽
    def edge_chunk(chunk: list[dict]) -> None:
        try:
            outs = edge_translate.translate([row["state"] for row in chunk], "en")
        except Exception:  # noqa: BLE001
            with LOCK:
                stats["edge_error"] += len(chunk)
            return
        for row, text in zip(chunk, outs):
            if text and relaxed_ok(row, text):
                save(row, text)
            else:
                with LOCK:
                    stats["edge_reject"] += 1

    chunks = [pending[i : i + args.batch] for i in range(0, len(pending), args.batch)]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(edge_chunk, chunks))
    core.log(f"Edge 放宽后：{dict(stats)}")

    # 2) LLM 兜底
    remaining = []
    done = set()
    for line in trans_path.open(encoding="utf-8"):
        if line.strip():
            row = json.loads(line)
            if row.get("kind") == "translation":
                done.add((row["source_id"], row["lang"]))
    remaining = [row for row in pending if (row["source_id"], "en") not in done]
    core.log(f"仍需 LLM 兜底 {len(remaining)}")
    llm_chunks = [remaining[i : i + args.batch] for i in range(0, len(remaining), args.batch)]

    def llm_chunk(chunk: list[dict]) -> None:
        try:
            outs = core.translate_rows(chunk, "en")
        except Exception:  # noqa: BLE001
            outs = [None] * len(chunk)
        for row, text in zip(chunk, outs):
            if text and core.translation_validate(text, row["state"], "en")[0]:
                save(row, text)
            else:
                with LOCK:
                    stats["llm_reject"] += 1

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(llm_chunk, llm_chunks))
    core.log(f"最终：{dict(stats)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
