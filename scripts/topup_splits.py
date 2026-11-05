"""按“每语言每垂类记录数”补齐三个 split：预训练 / 测试 / 后训练。

口径（2026-11-05）：
- 预训练 ``train/``                >= 3000 条/垂类/语言
- 测试   ``test/``                 >=  500 条/垂类/语言
- 后训练 ``train/posttrain/``      >=  300 条/垂类/语言
每条记录 = 1 个 state + 3 道题（noul / choice / score 各一，题目用该语言原文）。

硬性并发：源语言（简体中文）生成 2 并发 × 4 条/请求；对齐翻译共用 2 并发 × 4 条/请求。

中间产物（可断点续跑，不入库）：``artifacts/llm/splits/seeds.jsonl``、
``artifacts/llm/splits/translations.jsonl``。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import queue
import random
import re
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import topup_pretrain as core

ROOT = core.ROOT
DEFAULT_WORK = ROOT / "artifacts" / "llm" / "splits"
SPLIT_TARGET = {"train": 3000, "test": 500, "post": 300, "dev": 100}
SPLIT_DIR = {
    "train": ROOT / "train",
    "test": ROOT / "test",
    "post": ROOT / "train" / "posttrain",
    "dev": ROOT / "dev",
}
SPLIT_RECORD_SPLIT = {"train": "train", "test": "test", "post": "train", "dev": "dev"}
SPLIT_STYLE = {"train": "llm", "test": "llm", "post": "llm-post", "dev": "llm"}
EN_CACHE = ROOT / "artifacts" / "llm" / "legacy_en.jsonl"
LANGS = core.LANGS
SOURCE = core.SOURCE
TARGETS = core.TARGETS
BATCH = 4
WORKERS = 2

WRITE_LOCK = threading.Lock()
STOP = threading.Event()


def split_of(row: dict) -> str:
    return row.get("split") or "train"


def load_store(path: Path) -> tuple[list[dict], dict[str, dict], dict[tuple[str, str], dict], int]:
    sources: list[dict] = []
    by_id: dict[str, dict] = {}
    translations: dict[tuple[str, str], dict] = {}
    seq = 0
    if path.exists():
        for line in path.open(encoding="utf-8"):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            match = re.search(r"(\d+)$", row.get("row_id", ""))
            if match:
                seq = max(seq, int(match.group(1)))
            if row.get("kind") == "source":
                sources.append(row)
                by_id[row["source_id"]] = row
            elif row.get("kind") == "translation":
                translations[(row["source_id"], row["lang"])] = row
    return sources, by_id, translations, seq


class MultiStore:
    def __init__(self, path: Path, translation_path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.sources, self.by_id, self.translations, self.seq = load_store(path)
        if translation_path.exists():
            for line in translation_path.open(encoding="utf-8"):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if row.get("kind") == "translation":
                    self.translations[(row["source_id"], row["lang"])] = row
        self.counts: Counter = Counter()
        for row in self.sources:
            self.counts[(split_of(row), row["category"], row["task"])] += 1

    def next_id(self) -> str:
        with WRITE_LOCK:
            self.seq += 1
            return f"split-{self.seq:06d}"

    def add_source(self, row: dict) -> None:
        with WRITE_LOCK:
            with self.path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            self.sources.append(row)
            self.by_id[row["source_id"]] = row
            self.counts[(row["split"], row["category"], row["task"])] += 1

    def add_translation(self, row: dict, path: Path) -> None:
        with WRITE_LOCK:
            with path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            self.translations[(row["source_id"], row["lang"])] = row


def split_counts() -> Counter:
    counter: Counter = Counter()
    for split, folder in SPLIT_DIR.items():
        if not folder.exists():
            continue
        for path in sorted(folder.glob("*.jsonl")):
            category, lang = path.stem.split(".")[1], path.stem.split(".")[2]
            n = 0
            for line in path.open(encoding="utf-8"):
                if line.strip():
                    n += 1
            counter[(split, category, lang)] = n
    return counter


def build_plan(specs: dict, counts: Counter) -> dict:
    """每个 (split, 垂类) 的缺口。

    以 **zh-Hans 条数**为基线：案情中心改造后 zh-Hant/en 都由 zh-Hans 对齐翻译而来，
    三语条数必然相等，因此缺口 = 目标 − zh-Hans 条数。
    """
    plan: dict[str, dict] = {}
    for split, target in SPLIT_TARGET.items():
        plan[split] = {}
        for category in sorted(specs):
            baseline = counts[(split, category, core.SOURCE)]
            gap = max(0, target - baseline)
            tasks = [task["id"] for task in specs[category]["tasks"]]
            base, extra = divmod(gap, len(tasks))
            plan[split][category] = {
                "target": target,
                "current": {lang: counts[(split, category, lang)] for lang in LANGS},
                "baseline": baseline,
                "add": gap,
                "tasks": {task: base + (1 if index < extra else 0) for index, task in enumerate(tasks)},
            }
    return plan


def generation_job(job: tuple[str, str, str, int, int], store: MultiStore, specs: dict,
                   examples: dict, seen: dict) -> int:
    split, category, task_id, batch_no, target = job
    if STOP.is_set():
        return 0
    if store.counts[(split, category, task_id)] >= target:
        return 0
    spec = specs[category]
    task = next(item for item in spec["tasks"] if item["id"] == task_id)
    prompt = core.build_generation_prompt(category, task, examples.get((category, task_id), []), batch_no)
    try:
        text = core.call_api(
            [
                {"role": "system", "content": "你是资深业务数据撰写员。只输出 JSON，不要解释，不要 Markdown 代码块。"},
                {"role": "user", "content": prompt},
            ],
            temperature=1.0,
            max_tokens=12000,
        )
    except Exception as exc:  # noqa: BLE001
        core.log(f"生成失败 {split}/{category}/{task_id}：{type(exc).__name__} {str(exc)[:100]}")
        return 0
    accepted = 0
    for payload in core.parse_json_records(text):
        if accepted >= BATCH:
            break
        state = payload.get("state")
        labels = payload.get("labels")
        ok, _ = core.validate_state(str(state or ""), SOURCE, task)
        if not ok:
            continue
        ok, _ = core.validate_labels(labels, task)
        if not ok:
            continue
        digest = core.fingerprint(state)
        with WRITE_LOCK:
            if digest in seen[(category, SOURCE)]:
                continue
            seen[(category, SOURCE)].add(digest)
        row = {
            "kind": "source",
            "row_id": store.next_id(),
            "source_id": "",
            "split": split,
            "category": category,
            "task": task_id,
            "lang": SOURCE,
            "state": str(state).strip(),
            "labels": labels,
        }
        row["source_id"] = row["row_id"]
        store.add_source(row)
        accepted += 1
    return accepted


def generate(plan: dict, specs: dict, store: MultiStore, examples: dict, seen: dict) -> dict[str, int]:
    total = sum(entry["add"] for entries in plan.values() for entry in entries.values())
    core.log(f"生成目标 {total} 条源记录（{WORKERS} 并发 × {BATCH} 条/请求）")

    def handler(job):
        accepted = generation_job(job, store, specs, examples, seen)
        if accepted:
            split, category, task_id, _, _ = job
            core.log(f"生成 +{accepted} {split}/{category}/{task_id}")

    for round_no in range(1, 6):
        jobs = []
        for split, categories in plan.items():
            for category, entry in categories.items():
                for task_id, target in entry["tasks"].items():
                    done = store.counts[(split, category, task_id)]
                    need = max(0, target - done)
                    if need <= 0:
                        continue
                    batches = (need * 11 + BATCH * 10 - 1) // (BATCH * 10) + 1
                    for batch_no in range(batches):
                        jobs.append((split, category, task_id, batch_no + round_no * 1000, target))
        if not jobs:
            break
        random.Random(100 + round_no).shuffle(jobs)
        core.log(f"生成第 {round_no} 轮：{len(jobs)} 批")
        core.run_jobs(jobs, handler)
        missing = {
            f"{split}/{category}/{task}": target - store.counts[(split, category, task)]
            for split, categories in plan.items()
            for category, entry in categories.items()
            for task, target in entry["tasks"].items()
            if store.counts[(split, category, task)] < target
        }
        if not missing:
            return {}
        core.log(f"第 {round_no} 轮后缺口 {len(missing)} 项，合计 {sum(missing.values())} 条")
    return missing


def merge(plan: dict, specs: dict, store: MultiStore, dry_run: bool = False) -> dict:
    complete = [row for row in store.sources
                if all((row["source_id"], target) in store.translations for target in TARGETS)]
    grouped: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for source in complete:
        split = split_of(source)
        grouped[(split, source["category"], source["lang"])].append(source)
        for target in TARGETS:
            grouped[(split, source["category"], target)].append(store.translations[(source["source_id"], target)])

    report = {"added": {}, "skipped": {}}
    added_en: list[tuple[str, str]] = []
    for split, categories in plan.items():
        for category in categories:
            spec = specs[category]
            tasks = {task["id"]: task for task in spec["tasks"]}
            folder = SPLIT_DIR[split]
            state: dict[str, dict] = {}
            for lang in LANGS:
                path = folder / f"00-llm.{category}.{lang}.jsonl"
                existing_lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
                existing_states = set()
                max_index = 0
                for line in existing_lines:
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    existing_states.add(core.fingerprint(record["state"]))
                    match = re.search(r"-(\d+)$", record["id"])
                    if match:
                        max_index = max(max_index, int(match.group(1)))
                state[lang] = {"path": path, "lines": existing_lines, "states": existing_states,
                               "index": max_index, "new": []}

            # 同一案情三语必须同时写入，保证各语言条数严格一致。
            skipped = 0
            for row in grouped.get((split, category, "zh-Hans"), []):
                task = tasks.get(row["task"])
                if task is None:
                    skipped += 1
                    continue
                payload: dict[str, dict] = {}
                bad = False
                for lang in LANGS:
                    translated = row if lang == core.SOURCE else store.translations.get((row["source_id"], lang))
                    if translated is None:
                        bad = True
                        break
                    ok, _ = core.validate_state(translated["state"], lang, task)
                    if not ok or core.fingerprint(translated["state"]) in state[lang]["states"]:
                        bad = True
                        break
                    payload[lang] = translated
                if bad:
                    skipped += 1
                    continue
                suffix = re.sub(r"\D", "", row["source_id"][-6:])
                case_id = f"{category}-{int(suffix or 0):06d}"
                for lang in LANGS:
                    translated = payload[lang]
                    variant = hashlib.blake2b(translated["state"].encode(), digest_size=2).digest()[0] % 2
                    entry = state[lang]
                    entry["index"] += 1
                    record_split = SPLIT_RECORD_SPLIT[split]
                    entry["new"].append({
                        "id": f"{record_split}-llm-{category}-{lang.lower()}-{entry['index']:06d}",
                        "split": record_split,
                        "task": row["task"],
                        "category": category,
                        "lang": lang,
                        "style": SPLIT_STYLE[split],
                        "state": translated["state"],
                        "questions": core.question_schema(spec, task, lang, variant),
                        "labels": row["labels"],
                        "case_id": case_id,
                        "question_lang": lang,
                        "source_lang": core.SOURCE,
                    })
                    entry["states"].add(core.fingerprint(translated["state"]))
                added_en.append((case_id, payload["en"]["state"]))

            for lang in LANGS:
                entry = state[lang]
                report["added"][f"{split}/{category}/{lang}"] = len(entry["new"])
                report["skipped"][f"{split}/{category}/{lang}"] = skipped
                if dry_run or not entry["new"]:
                    continue
                folder.mkdir(parents=True, exist_ok=True)
                with entry["path"].open("w", encoding="utf-8", newline="\n") as handle:
                    for line in entry["lines"]:
                        if line.strip():
                            handle.write(line + "\n")
                    for record in entry["new"]:
                        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                core.log(f"合并 {split}/{category}/{lang} +{len(entry['new'])}（跳过 {skipped}）")
    if added_en and not dry_run:
        # 写回英文缓存，使 rebuild_legacy_aligned 的重复运行保持幂等。
        with EN_CACHE.open("a", encoding="utf-8", newline="\n") as cache:
            for case_id, text in added_en:
                cache.write(json.dumps({"kind": "en", "source_id": case_id, "state": text}, ensure_ascii=False) + "\n")
        core.log(f"英文缓存追加 {len(added_en)} 条")
    return report

def translate(store: MultiStore, batch: int = BATCH, workers: int = WORKERS) -> Counter:
    """补译：zh-Hant 用 OpenCC，en 用 Edge（2 并发 × 4 条）；结果写入 translations.jsonl。"""
    import edge_translate

    stats: Counter = Counter()
    pending = [row for row in store.sources
               if any((row["source_id"], lang) not in store.translations for lang in TARGETS)]
    core.log(f"待翻译 {len(pending)} 条源记录（{workers} 并发 × {batch} 条/请求）")
    if not pending:
        return stats

    trans_path = store.path.parent / "translations.jsonl"

    def save(row: dict, lang: str, text: str) -> bool:
        if (row["source_id"], lang) in store.translations:
            return False
        store.add_translation({
            "kind": "translation",
            "row_id": f"{row['source_id']}-{lang}",
            "source_id": row["source_id"],
            "category": row["category"],
            "task": row["task"],
            "lang": lang,
            "state": text,
            "labels": row["labels"],
        }, trans_path)
        return True

    # zh-Hant：确定性转换，一次性完成
    hant = 0
    for row in pending:
        if (row["source_id"], "zh-Hant") in store.translations:
            continue
        text = core.S2T.convert(row["state"])
        if save(row, "zh-Hant", text):
            hant += 1
    stats["zh-Hant"] = hant
    core.log(f"繁中转换 +{hant}")

    # en：Edge 批量翻译
    todo = [row for row in pending if (row["source_id"], "en") not in store.translations]
    chunks = [todo[i : i + batch] for i in range(0, len(todo), batch)]
    lock = threading.Lock()

    def run(chunk: list[dict]) -> None:
        try:
            outputs = edge_translate.translate([row["state"] for row in chunk], "en")
        except Exception as exc:  # noqa: BLE001
            with lock:
                stats["error"] += len(chunk)
            return
        for row, text in zip(chunk, outputs):
            ok, reason = core.translation_validate(text, row["state"], "en")
            if not ok:
                with lock:
                    stats[f"reject:{reason}"] += 1
                continue
            if save(row, "en", text.strip()):
                with lock:
                    stats["en"] += 1
                    if stats["en"] % 400 == 0:
                        core.log(f"英文 +{stats['en']}")

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(run, chunks))
    core.log(f"英文 +{stats['en']}，stats={dict(stats)}")
    return stats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="按记录数补齐 train/test/post 三个 split")
    parser.add_argument("--work", default=str(DEFAULT_WORK))
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--merge", action="store_true")
    parser.add_argument("--translate", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--rounds", type=int, default=1)
    args = parser.parse_args(argv)

    specs = core.load_specs()
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    core.LOG_PATH = work / "splits.log"

    counts = split_counts()
    plan = build_plan(specs, counts)
    (work / "plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    for split, categories in plan.items():
        total = sum(entry["add"] for entry in categories.values())
        core.log(f"{split}: 目标 {SPLIT_TARGET[split]} 条/垂类/语言，需补 {total} 条源记录")
        for category, entry in categories.items():
            if entry["add"]:
                core.log(f"  {category}: {entry['current']} +{entry['add']} -> {entry['tasks']}")
    if args.plan:
        return 0

    store = MultiStore(work / "seeds.jsonl", work / "translations.jsonl")
    core.log(f"已有种子：source={len(store.sources)} translation={len(store.translations)}")

    if args.generate:
        _, _, examples = core.load_existing()
        _counts, seen, _ex = core.load_existing()
        for round_no in range(args.rounds):
            missing = generate(plan, specs, store, examples, seen)
            if not missing:
                break
        if not missing:
            (work / "generation.done").write_text("done\n", encoding="utf-8")
            core.log("生成已达标")
    if args.translate:
        translate(store)
    if args.merge:
        report = merge(plan, specs, store, dry_run=args.dry_run)
        (work / "merge_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        core.log(f"合并完成：共 {sum(report['added'].values())} 条")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
