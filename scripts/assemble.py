"""把 LLM 撰写的种子（state + labels）装配成合法数据集分片。

输出（每个垂类 × 每个语言 × 每个 split 一个独立文件，便于单独扩量）
--------------------------------------------------------------------
    data/train/00-llm.<category>.<lang>.jsonl            # 预训练
    data/train/posttrain/00-llm.<category>.<lang>.jsonl  # 后训练（split 仍是 train）
    data/test/00-llm.<category>.<lang>.jsonl             # 测试集
    data/dev/00-llm.<category>.<lang>.jsonl              # 开发集

设计
----
- 种子只含 `task` / `lang` / `state` / `labels`；问题定义一律从 `src/nano_jev/synth/*.py`
  的 `TaskDef` 取，保证候选集与线上 API 契约一致。
- 每条记录做 schema 校验（`validate_record`）、标签域校验（`label_domain`）。
- 去重口径为内容指纹（state + questions），在同一垂类的 **所有 split 之间共享**，
  因此 train/test/dev/post 之间不会出现同一内容。
- 分桶按内容哈希稳定分配，重复运行结果一致。
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from inject_translations import inject  # noqa: E402
from nano_jev.data import validate_record  # noqa: E402
from nano_jev.quotas import LANGUAGE_PLAN  # noqa: E402
from nano_jev.synth import MODULE_NAMES  # noqa: E402

LANG_KEY = {
    "zh-Hans": "zh", "en": "en", "zh-Hant": "zh-Hant", "ru": "ru",
    "ko": "ko", "ja": "ja", "ar": "ar", "bo": "bo",
}
# 基础版本只保留三种语言；其余语言的种子留在 artifacts/llm/seeds 里，不写入 data/。
ALLOWED_LANGS = ("zh-Hans", "zh-Hant", "en")
DATA_ROOT = ROOT / "data"
SEED_ROOT = ROOT / "artifacts" / "llm" / "seeds"
SEED_API_ROOT = ROOT / "artifacts" / "llm" / "seeds_api"


def module_for(category: str):
    for name in MODULE_NAMES:
        try:
            mod = importlib.import_module(f"nano_jev.synth.{name}")
        except ImportError:
            continue
        if getattr(mod, "CATEGORY", None) == category:
            return mod
    return None


def digest_of(record: dict) -> bytes:
    payload = record["state"] + "\x00" + json.dumps(
        record["questions"], sort_keys=True, ensure_ascii=False
    )
    return hashlib.blake2b(payload.encode(), digest_size=8).digest()


def collect_seeds(category: str) -> list[dict]:
    seeds: list[dict] = []
    for folder in (SEED_ROOT / category, SEED_API_ROOT / category):
        if not folder.exists():
            continue
        for path in sorted(folder.glob("*.jsonl")):
            with path.open(encoding="utf-8") as handle:
                for line_no, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    try:
                        seeds.append(json.loads(line))
                    except json.JSONDecodeError as exc:
                        raise SystemExit(f"{path}:{line_no}: 非法 JSON: {exc}") from exc
    return seeds


SPLIT_WEIGHTS = (("train", 864), ("test", 87), ("post", 43), ("dev", 6))  # 目标比例 20000/2000/1000/100，dev 提到 6‰ 保证 >=100


def split_for(category: str, task_id: str, lang: str, index: int) -> str:
    """按目标比例的稳定哈希分桶。

    目标口径：每垂类每语言 train 5000 / test 500 / post 200 / dev 50，即
    86.9% / 8.7% / 3.5% / 0.9%。用哈希而不是取模游标，保证小语料也能分出
    test/post/dev，且同一内容重复装配结果不变。
    """
    digest = hashlib.blake2b(f"{category}:{lang}:{index}".encode(), digest_size=4).digest()
    bucket = int.from_bytes(digest, "big") % 1000
    cursor = 0
    for name, weight in SPLIT_WEIGHTS:
        cursor += weight
        if bucket < cursor:
            return name
    return "train"


def build_records(category: str, module, seeds: list[dict]) -> tuple[dict[str, list[dict]], dict]:
    tasks = {task.id: task for task in module.TASKS}
    seen: set[bytes] = set()
    buckets: dict[str, list[dict]] = {"train": [], "test": [], "dev": [], "post": []}
    stats: Counter = Counter()
    rejected: list[str] = []
    for index, seed in enumerate(seeds):
        task_id = seed.get("task")
        lang = seed.get("lang")
        task = tasks.get(task_id)
        if task is None:
            rejected.append(f"未知任务 {task_id}")
            continue
        if lang not in LANGUAGE_PLAN:
            rejected.append(f"未知语言 {lang}")
            continue
        if lang not in ALLOWED_LANGS:
            stats["skipped_lang"] += 1
            continue
        labels = seed.get("labels") or {}
        if set(labels) != set(task.questions):
            rejected.append(f"{task_id}/{lang}: labels 键不匹配 {sorted(labels)}")
            continue
        bad = False
        for name, value in labels.items():
            if value not in task.label_domain(name):
                rejected.append(f"{task_id}/{lang}.{name}: 标签 {value!r} 越界")
                bad = True
                break
        if bad:
            continue
        key = LANG_KEY[lang]
        rng = random.Random(f"llm:{category}:{task_id}:{lang}:{index}")
        try:
            questions = {name: question.for_lang(key, rng) for name, question in task.questions.items()}
        except KeyError as exc:
            rejected.append(f"{task_id}/{lang}: 缺 {exc} 译文，跳过")
            continue

        split = split_for(category, task_id, lang, index)
        style = "llm-post" if split == "post" else "llm"
        record_split = "train" if split in ("train", "post") else split
        record = {
            "id": f"{record_split}-llm-{category}-{lang.lower()}-{index:06d}",
            "split": record_split,
            "task": task_id,
            "category": category,
            "lang": lang,
            "style": style,
            "state": seed["state"].strip(),
            "questions": questions,
            "labels": labels,
        }
        try:
            validate_record(record, where=f"{category}:{task_id}/{lang}:{index}")
        except Exception as exc:  # noqa: BLE001
            rejected.append(f"{task_id}/{lang}#{index}: {exc}")
            continue
        digest = digest_of(record)
        if digest in seen:
            stats["duplicate"] += 1
            continue
        seen.add(digest)
        buckets[split].append(record)
        stats[f"lang:{lang}"] += 1
        stats[f"task:{task_id}"] += 1
        stats[f"split:{split}"] += 1
    for reason in rejected[:10]:
        print(f"    ! {reason}")
    stats["rejected"] = len(rejected)
    return buckets, dict(stats)


def output_path(category: str, split: str, lang: str) -> Path:
    """每个垂类、每个语言、每个 split 各一个文件，便于单独扩量。

    - 预训练：``data/train/00-llm.<category>.<lang>.jsonl``
    - 后训练：``data/train/posttrain/00-llm.<category>.<lang>.jsonl``
      （放在 train 子目录下：``data freeze`` 递归扫描时 split 仍取第一段路径 = train，
      而训练侧 ``--data data/train`` 的 ``glob("*.jsonl")`` 不会读到后训练分片。）
    """
    if split == "post":
        return DATA_ROOT / "train" / "posttrain" / f"00-llm.{category}.{lang}.jsonl"
    return DATA_ROOT / split / f"00-llm.{category}.{lang}.jsonl"


def cleanup_legacy(category: str) -> list[Path]:
    """删除该垂类的旧分片（含每语言分片），避免历史文件残留导致 split 计数虚高。"""
    stale = [
        DATA_ROOT / "train" / f"00-llm.{category}.jsonl",
        DATA_ROOT / "train" / f"50-llm-post.{category}.jsonl",
        DATA_ROOT / "test" / f"00-llm.{category}.jsonl",
        DATA_ROOT / "dev" / f"00-llm.{category}.jsonl",
    ]
    removed = []
    for path in stale:
        if path.exists():
            path.unlink()
            removed.append(path)
    for folder in (DATA_ROOT / "train", DATA_ROOT / "test", DATA_ROOT / "dev", DATA_ROOT / "train" / "posttrain"):
        if not folder.exists():
            continue
        for path in folder.glob(f"00-llm.{category}.*.jsonl"):
            path.unlink()
            removed.append(path)
    return removed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="装配 LLM 种子为数据集分片")
    parser.add_argument("--categories", nargs="*", default=None)
    parser.add_argument("--write", action="store_true", help="写入 data/（默认只校验）")
    args = parser.parse_args(argv)

    if args.categories:
        categories = args.categories
    else:
        found: set[str] = set()
        for root in (SEED_ROOT, SEED_API_ROOT):
            if root.exists():
                found.update(p.name for p in root.iterdir() if p.is_dir())
        categories = sorted(found)

    summary = {}
    for category in categories:
        module = module_for(category)
        if module is None:
            print(f"{category}: 模块未找到")
            continue
        inject(category, apply=True)
        seeds = collect_seeds(category)
        buckets, stats = build_records(category, module, seeds)
        counts = {split: len(items) for split, items in buckets.items()}
        print(f"{category:26s} seeds={len(seeds):5d} {counts} rejected={stats['rejected']}")
        summary[category] = {"seeds": len(seeds), **counts, **stats}
        if args.write:
            for path in cleanup_legacy(category):
                print(f"    - 删除旧分片 {path.relative_to(ROOT)}")
            grouped: dict[tuple[str, str], list[dict]] = {}
            for split, records in buckets.items():
                for record in records:
                    grouped.setdefault((split, record["lang"]), []).append(record)
            for (split, lang), records in sorted(grouped.items()):
                path = output_path(category, split, lang)
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("w", encoding="utf-8", newline="\n") as handle:
                    for record in records:
                        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    out_json = ROOT / "artifacts" / "llm" / "assemble_summary.json"
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n汇总写入 {out_json.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
