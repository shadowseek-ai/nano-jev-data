"""在线翻译：与源语言生成并行，按“2 并发 × 4 条”翻译英文。

- 源记录从 ``<work>/seeds.jsonl`` 增量读取；
- 繁体中文用 OpenCC s2twp 做确定性转换，保证占位符/数字/事实严格对齐；
- 英文默认走 **Edge 翻译 API**（免费、约 0.2 秒/条），只有 Edge 反复失败的记录
  才回落到 LLM 修复，避免不必要的模型费用；
- 译文写入 ``<work>/translations.jsonl``，不干扰生成进程；
- 生成完成标记 ``<work>/generation.done`` 出现且无缺口后退出。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import threading
import time
from collections import Counter
from pathlib import Path

import edge_translate
import topup_pretrain as core

WRITE_LOCK = threading.Lock()
STOP = threading.Event()
STATE_LOCK = threading.Lock()
ENGINE = "edge"


def translate_english(rows: list[dict]) -> list[str | None]:
    """按批翻译成英文：Edge 为主，逐条校验数值与占位符。"""
    if not rows:
        return []
    texts: list[str | None] = []
    if ENGINE == "edge":
        try:
            outputs = edge_translate.translate([row["state"] for row in rows], "en")
        except Exception as exc:  # noqa: BLE001
            core.log(f"Edge 翻译失败：{type(exc).__name__} {str(exc)[:100]}")
            outputs = [""] * len(rows)
    else:
        outputs = core.translate_raw(rows, "en")
    for row, candidate in zip(rows, outputs):
        if not candidate:
            texts.append(None)
            continue
        ok, _ = core.translation_validate(candidate, row["state"], "en")
        texts.append(candidate.strip() if ok else None)
    if ENGINE == "edge" and any(text is None for text in texts):
        # 同一批里个别句子被 Edge 改写坏了：单条重试一次。
        for index, text in enumerate(texts):
            if text is not None:
                continue
            try:
                single = edge_translate.translate([rows[index]["state"]], "en")
            except Exception:  # noqa: BLE001
                single = [""]
            candidate = (single[0] or "").strip()
            if candidate and core.translation_validate(candidate, rows[index]["state"], "en")[0]:
                texts[index] = candidate
    return texts


def load_sources(path: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    if not path.exists():
        return out
    for line in path.open(encoding="utf-8"):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("kind") == "source":
            out[row["source_id"]] = row
    return out


def load_translations(path: Path) -> dict[tuple[str, str], dict]:
    out: dict[tuple[str, str], dict] = {}
    if not path.exists():
        return out
    for line in path.open(encoding="utf-8"):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("kind") == "translation":
            out[(row["source_id"], row["lang"])] = row
    return out


def append_translation(path: Path, row: dict) -> None:
    with WRITE_LOCK:
        with path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", default=str(core.DEFAULT_WORK))
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--max-hours", type=float, default=14.0)
    parser.add_argument("--engine", choices=("edge", "llm"), default="edge", help="英文翻译引擎，默认 Edge（免费）")
    parser.add_argument("--allow-llm-repair", action="store_true", default=True,
                        help="Edge 反复失败时允许用 LLM 兜底（默认开启，只针对失败样本）")
    parser.add_argument("--no-llm-repair", dest="allow_llm_repair", action="store_false")
    parser.add_argument("--hant-llm", action="store_true", help="繁体也用 LLM（默认用 OpenCC）")
    args = parser.parse_args(argv)
    global ENGINE
    ENGINE = args.engine

    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    core.LOG_PATH = work / "translate.log"
    sources_path = work / "seeds.jsonl"
    trans_path = work / "translations.jsonl"
    done_marker = work / "generation.done"

    sources = load_sources(sources_path)
    translations = load_translations(trans_path)
    translated = set(translations)
    inflight: set[str] = set()
    failed: set[tuple[str, str]] = set()
    attempts: Counter = Counter()
    counters = Counter()
    started = time.time()

    def refresh_sources() -> int:
        latest = load_sources(sources_path)
        with STATE_LOCK:
            new = 0
            for source_id, row in latest.items():
                if source_id not in sources:
                    sources[source_id] = row
                    new += 1
        return new

    def convert_hant() -> int:
        added = 0
        with STATE_LOCK:
            pending = [row for sid, row in sources.items()
                       if (sid, "zh-Hant") not in translated and sid not in inflight
                       and (sid, "zh-Hant") not in failed]
        for row in pending:
            text = core.S2T.convert(row["state"])
            ok, _ = core.translation_validate(text, row["state"], "zh-Hant")
            if not ok:
                with STATE_LOCK:
                    failed.add((row["source_id"], "zh-Hant"))
                core.log(f"繁中转换失败，跳过 {row['source_id']}")
                continue
            record = {
                "kind": "translation",
                "row_id": f"{row['source_id']}-zh-Hant",
                "source_id": row["source_id"],
                "category": row["category"],
                "task": row["task"],
                "lang": "zh-Hant",
                "state": text,
                "labels": row["labels"],
            }
            append_translation(trans_path, record)
            with STATE_LOCK:
                translated.add((row["source_id"], "zh-Hant"))
            added += 1
        return added

    def take_batch() -> list[dict]:
        with STATE_LOCK:
            batch = []
            for source_id, row in sources.items():
                if (source_id, "en") in translated or source_id in inflight:
                    continue
                if (source_id, "en") in failed:
                    continue
                batch.append(row)
                inflight.add(source_id)
                if len(batch) >= args.batch:
                    break
            return batch

    def worker() -> None:
        while not STOP.is_set():
            batch = take_batch()
            if not batch:
                if done_marker.exists():
                    with STATE_LOCK:
                        missing = any(
                            (sid, "en") not in translated and sid not in inflight
                            for sid in sources
                        )
                    if not missing:
                        return
                time.sleep(2)
                continue
            try:
                texts = translate_english(batch)
            except Exception as exc:  # noqa: BLE001
                core.log(f"英文翻译异常：{type(exc).__name__} {str(exc)[:100]}")
                texts = [None] * len(batch)
            accepted = 0
            for row, text in zip(batch, texts):
                if text is None:
                    with STATE_LOCK:
                        attempts[(row["source_id"], "en")] += 1
                        exhausted = attempts[(row["source_id"], "en")] >= 3
                    if exhausted and args.allow_llm_repair:
                        # 只有 Edge 反复失败时才动用 LLM，控制费用。
                        text = core.translate_with_repair(row, "en", attempts=2)
                if text is None:
                    with STATE_LOCK:
                        if attempts[(row["source_id"], "en")] >= 6:
                            failed.add((row["source_id"], "en"))
                            core.log(f"英文翻译失败，跳过 {row['source_id']}")
                    continue
                record = {
                    "kind": "translation",
                    "row_id": f"{row['source_id']}-en",
                    "source_id": row["source_id"],
                    "category": row["category"],
                    "task": row["task"],
                    "lang": "en",
                    "state": text,
                    "labels": row["labels"],
                }
                append_translation(trans_path, record)
                with STATE_LOCK:
                    translated.add((row["source_id"], "en"))
                accepted += 1
            counters["en"] += accepted
            with STATE_LOCK:
                for row in batch:
                    inflight.discard(row["source_id"])
            if accepted == 0:
                time.sleep(3)

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(args.workers)]
    for thread in threads:
        thread.start()

    last_log = 0.0
    while time.time() - started < args.max_hours * 3600:
        new = refresh_sources()
        hant = convert_hant()
        with STATE_LOCK:
            total = len(sources)
            done_hant = sum(1 for sid in sources if (sid, "zh-Hant") in translated)
            done_en = sum(1 for sid in sources if (sid, "en") in translated)
            en_missing = sum(1 for sid in sources if (sid, "en") not in translated and (sid, "en") not in failed)
            hant_missing = sum(1 for sid in sources if (sid, "zh-Hant") not in translated and (sid, "zh-Hant") not in failed)
            active = len(inflight)
        if time.time() - last_log > 30:
            last_log = time.time()
            rate = done_en / max(time.time() - started, 1) * 3600
            core.log(
                f"源 {total}（+{new}） 繁中 {done_hant} 英文 {done_en} "
                f"英文缺口 {en_missing} 在途 {active} 速率 {rate:.0f} 条/小时"
            )
        if done_marker.exists() and en_missing == 0 and hant_missing == 0 and active == 0:
            core.log(f"翻译完成：源 {total}，繁中 {done_hant}，英文 {done_en}，失败 繁中/英文 {sum(1 for k in failed if k[1] == 'zh-Hant')}/{sum(1 for k in failed if k[1] == 'en')}")
            STOP.set()
            break
        if hant:
            core.log(f"繁中转换 +{hant}")
        time.sleep(3)
    else:
        core.log("翻译超时退出")
        STOP.set()

    for thread in threads:
        thread.join(timeout=30)
    core.log(f"结束：繁中 {sum(1 for sid in sources if (sid, 'zh-Hant') in translated)}/{len(sources)}，"
             f"英文 {sum(1 for sid in sources if (sid, 'en') in translated)}/{len(sources)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
