"""重试被跳过的记录：把 cursor 侧文件里 skipped 的下标重新翻译一遍。

翻译器的"逐条跳过"机制会记录无法通过校验的源下标，避免卡死整组；但个别记录
换个批次/温度是能过的。本脚本只针对这些下标重试，成功即追加并移出 skipped。

用法：
    py artifacts/llm/retry_skipped.py --lang en --workers 8 --batch 8
    py artifacts/llm/retry_skipped.py --lang zh-Hant --categories energy_utilities
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import translate_seeds as T  # noqa: E402


def retry_group(category: str, task: str, lang: str, sources: list[dict], batch: int) -> tuple[int, int]:
    side = T.OUT / category / f"{task}.{lang}.cursor.json"
    if not side.exists():
        return 0, 0
    state = json.loads(side.read_text(encoding="utf-8"))
    skipped = list(state.get("skipped") or [])
    if not skipped:
        return 0, 0
    path = T.OUT / category / f"{task}.{lang}.jsonl"
    recovered = 0
    remaining: list[int] = []
    for start in range(0, len(skipped), batch):
        indices = skipped[start : start + batch]
        chunk = [sources[i] for i in indices if 0 <= i < len(sources)]
        if len(chunk) != len(indices):
            remaining.extend([i for i in indices if not (0 <= i < len(sources))])
            continue
        try:
            texts = T.translate_batch(chunk, lang)
        except Exception:  # noqa: BLE001
            remaining.extend(indices)
            continue
        rows = []
        kept: list[int] = []
        for index, item, text in zip(indices, chunk, texts):
            ok, _ = T.validate(text, item["state"], lang)
            if not ok:
                remaining.append(index)
                continue
            rows.append({"task": task, "lang": lang, "state": text.strip(), "labels": item["labels"]})
            kept.append(index)
        if rows:
            with path.open("a", encoding="utf-8", newline="\n") as handle:
                for row in rows:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            recovered += len(rows)
        _ = kept
        time.sleep(0.5)
    state["skipped"] = remaining
    side.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
    return recovered, len(skipped)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--categories", nargs="*", default=None)
    args = parser.parse_args(argv)

    groups = T.load_sources()
    targets = [
        (category, task, records)
        for (category, task), records in sorted(groups.items(), key=lambda kv: -len(kv[1]))
        if not args.categories or category in set(args.categories)
    ]
    print(f"groups={len(targets)} lang={args.lang}")
    total_recovered = 0
    total_skipped = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(retry_group, category, task, args.lang, records, args.batch): (category, task)
            for category, task, records in targets
        }
        for future in as_completed(futures):
            recovered, skipped = future.result()
            total_recovered += recovered
            total_skipped += skipped
            if skipped:
                print(f"  {futures[future]}: recovered {recovered}/{skipped}")
    print(f"done recovered={total_recovered} attempted={total_skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
