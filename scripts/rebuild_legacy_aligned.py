"""把旧数据集改成「案情中心」：以 zh-Hans 为源，重建 zh-Hant / en 分片。

做法
----
1. 旧数据里 zh-Hans 分片作为唯一真源（labels 只来自它，翻译不改判定）；
2. zh-Hant 用 OpenCC ``s2twp`` 确定性转换（零成本、严格对齐）；
3. en 用 Edge 翻译 API（免费），2 并发 × 4 条/请求，全局限流；
4. 三语共用同一个 ``case_id``，``question_lang`` 为各自语言，``source_lang`` 固定 zh-Hans；
5. 题面按 spec 的目标语言渲染，保证「1 个 state + 3 道该语言原文题」。

断点续跑：英文译文缓存在 ``artifacts/llm/legacy_en.jsonl``，可反复运行。
输出先写到 ``<split>_new/``，成功后原子替换原文件。
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
import case_meta  # noqa: E402
import edge_translate  # noqa: E402
import topup_pretrain as core  # noqa: E402

SPLIT_DIRS = ("train", "train/posttrain", "test", "dev")
CACHE = ROOT / "artifacts" / "llm" / "legacy_en.jsonl"
LOCK = threading.Lock()
FAILED: list[str] = []


def load_cache() -> dict[str, str]:
    cache: dict[str, str] = {}
    if CACHE.exists():
        for line in CACHE.open(encoding="utf-8"):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("kind") == "en":
                cache[row["source_id"]] = row["state"]
    return cache


def append_cache(source_id: str, text: str) -> None:
    with LOCK:
        with CACHE.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps({"kind": "en", "source_id": source_id, "state": text}, ensure_ascii=False) + "\n")


def translate_missing(items: list[dict], cache: dict[str, str], batch: int, workers: int) -> Counter:
    """英文翻译：Edge 批量 -> Edge 单条重试 -> LLM 批量兜底（只处理失败样本）。"""
    pending = [row for row in items if row["source_id"] not in cache]
    stats: Counter = Counter()
    if not pending:
        return stats
    print(f"待翻译 {len(pending)} 条（{workers} 并发 × {batch} 条/请求）", flush=True)

    def strict_ok(row: dict, text: str) -> tuple[bool, str]:
        return core.translation_validate(text, row["state"], "en")

    def relaxed_ok(row: dict, text: str) -> bool:
        """数值很多时允许漏 1 个（MT 偶发丢数字），占位符与语言纯度仍然必须严格满足。"""
        ok, reason = strict_ok(row, text)
        if ok:
            return True
        if reason != "number":
            return False
        if set(core.PLACEHOLDER.findall(text)) != set(core.PLACEHOLDER.findall(row["state"])):
            return False
        if core.CJK.search(text):
            return False
        source_values = core.numeric_values(row["state"], "zh-Hans")
        missing = source_values - core.numeric_values(text, "en")
        total = sum(source_values.values())
        return total >= 8 and sum(missing.values()) <= 1

    def accept(row: dict, text: str, relaxed: bool = False) -> bool:
        ok, reason = strict_ok(row, text)
        if not ok and relaxed and relaxed_ok(row, text):
            ok, reason = True, "ok-relaxed"
        if not ok:
            with LOCK:
                stats[f"reject:{reason}"] += 1
            return False
        cache[row["source_id"]] = text.strip()
        append_cache(row["source_id"], text.strip())
        with LOCK:
            stats["ok" if reason == "ok" else "ok_relaxed"] += 1
            if stats["ok"] and stats["ok"] % 400 == 0:
                print(f"  已翻译 {stats['ok']} 条", flush=True)
        return True

    def edge_pass(chunks: list[list[dict]], label: str, relaxed: bool = False) -> list[dict]:
        failed: list[dict] = []

        def run(chunk: list[dict]) -> None:
            try:
                outputs = edge_translate.translate([row["state"] for row in chunk], "en")
            except Exception:  # noqa: BLE001
                with LOCK:
                    failed.extend(chunk)
                return
            for row, text in zip(chunk, outputs):
                if not accept(row, text, relaxed=relaxed):
                    with LOCK:
                        failed.append(row)

        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(run, chunks))
        print(f"  {label}：累计成功 {stats['ok']}，剩余 {len(failed)}", flush=True)
        return failed

    chunks = [pending[i : i + batch] for i in range(0, len(pending), batch)]
    failed = edge_pass(chunks, "Edge 批量（严格+放宽）", relaxed=True)
    if failed:
        llm_chunks = [failed[i : i + batch] for i in range(0, len(failed), batch)]
        remaining: list[dict] = []

        def run_llm(chunk: list[dict]) -> None:
            try:
                outputs = core.translate_rows(chunk, "en")
            except Exception:  # noqa: BLE001
                outputs = [None] * len(chunk)
            for row, text in zip(chunk, outputs):
                if not text or not accept(row, text):
                    with LOCK:
                        remaining.append(row)

        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(run_llm, llm_chunks))
        print(f"  LLM 兜底后仍失败 {len(remaining)} 条", flush=True)
        for row in remaining:
            FAILED.append(row["source_id"])
    return stats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--only", nargs="*", default=None, help="只处理指定 split 目录")
    args = parser.parse_args(argv)

    specs = core.load_specs()
    cache = load_cache()
    print(f"已有英文缓存 {len(cache)} 条", flush=True)

    # 1) 收集 zh-Hans 真源
    jobs: list[tuple[str, str, list[dict]]] = []
    for relative in (args.only or SPLIT_DIRS):
        folder = ROOT / relative
        if not folder.exists():
            continue
        for path in sorted(folder.glob("*.zh-Hans.jsonl")):
            category = path.stem.split(".")[1]
            rows = []
            for index, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                rows.append({
                    "source_id": record.get("case_id") or f"{category}-legacy-{index:06d}",
                    "category": category,
                    "task": record["task"],
                    "state": record["state"],
                    "labels": record["labels"],
                    "index": index,
                })
            jobs.append((relative, category, rows))

    total = sum(len(rows) for _, _, rows in jobs)
    print(f"源记录 {total} 条，将生成 {total} 条 zh-Hant 与 {total} 条 en", flush=True)
    if args.dry_run:
        return 0

    # 2) 翻译英文
    all_rows = [row for _, _, rows in jobs for row in rows]
    stats = translate_missing(all_rows, cache, args.batch, args.workers)
    print(f"翻译统计 {dict(stats)}，缓存 {len(cache)} 条，失败 {len(FAILED)} 条", flush=True)

    # 3) 重建三种语言分片
    written = Counter()
    for relative, category, rows in jobs:
        spec = specs[category]
        tasks = {task["id"]: task for task in spec["tasks"]}
        outputs: dict[str, list[str]] = {"zh-Hans": [], "zh-Hant": [], "en": []}
        index_by_lang: Counter = Counter()
        for row in rows:
            task = tasks.get(row["task"])
            if task is None:
                continue
            # 三语必须齐：英文没有可用译文就整案跳过，保证各语言条数完全一致。
            if row["source_id"] not in cache:
                continue
            for lang in ("zh-Hans", "zh-Hant", "en"):
                if lang == "zh-Hans":
                    text = row["state"]
                elif lang == "zh-Hant":
                    # OpenCC 只换字形、不动数字，因此这里不再做数值比对（会有解析误报）。
                    text = core.S2T.convert(row["state"])
                else:
                    text = cache[row["source_id"]]
                variant = int(row["source_id"].rsplit("-", 1)[-1]) % 2 if row["source_id"].rsplit("-", 1)[-1].isdigit() else 0
                index_by_lang[lang] += 1
                split_name = relative.split("/")[0]
                record_split = "train" if relative in ("train", "train/posttrain") else split_name
                style = "llm-post" if relative == "train/posttrain" else "llm"
                outputs[lang].append(json.dumps({
                    "id": f"{record_split}-llm-{category}-{lang.lower()}-{index_by_lang[lang]:06d}",
                    "split": record_split,
                    "task": row["task"],
                    "category": category,
                    "lang": lang,
                    "style": style,
                    "state": text,
                    "questions": core.question_schema(spec, task, lang, variant),
                    "labels": row["labels"],
                    "case_id": row["source_id"],
                    "question_lang": lang,
                    "source_lang": "zh-Hans",
                }, ensure_ascii=False))
        folder = ROOT / relative
        for lang, lines in outputs.items():
            if not lines:
                continue
            path = folder / f"00-llm.{category}.{lang}.jsonl"
            path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
            written[lang] += len(lines)
    print(f"写入 {dict(written)}", flush=True)
    if FAILED:
        (ROOT / "artifacts" / "llm" / "legacy_en_failed.json").write_text(
            json.dumps(sorted(set(FAILED)), ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
