"""从案情中心单语分片派生跨语言混合分片与语言对抗子集。

输入（同一 split 内三语共用 ``case_id``）::

    train/00-llm.<category>.<lang>.jsonl
    test/00-llm.<category>.<lang>.jsonl

输出::

    train/mix/10-mix.<category>.<stateLang>_<questionLang>.jsonl   # 6 种组合
    test/mix/10-mix.<category>.<stateLang>_<questionLang>.jsonl
    train/adv/20-adv-lang.<category>.<stateLang>_<questionLang>.jsonl  # 语言对抗（可选）

混合样本 = 同案情的 state（某语言） + questions（另一种语言），labels 与 case_id 不变，
因此零 LLM 成本且天然对齐。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import case_meta  # noqa: E402

LANGS = ("zh-Hans", "zh-Hant", "en")
SPLIT_DIRS = {
    "train": "train",
    "post": "train/posttrain",
    "test": "test",
    "dev": "dev",
}
SPLIT_LABEL = {"train": "train", "post": "train", "test": "test", "dev": "dev"}
SENT_SPLIT = {"zh-Hans": re.compile(r"(?<=[。；！？\n])"), "zh-Hant": re.compile(r"(?<=[。；！？\n])"),
              "en": re.compile(r"(?<=[.;!?\n])\s*")}
NUM = re.compile(r"\d")


def load_split(folder: Path) -> dict[str, dict[str, dict]]:
    """返回 {category: {case_id: {lang: record}}}。"""
    data: dict[str, dict[str, dict]] = defaultdict(lambda: defaultdict(dict))
    for path in sorted(folder.glob("00-llm.*.jsonl")):
        parts = path.stem.split(".")
        category, lang = parts[1], parts[2]
        for line in path.open(encoding="utf-8"):
            if not line.strip():
                continue
            record = json.loads(line)
            case_id = record.get("case_id")
            if case_id:
                data[category][case_id][lang] = record
    return data


def combos(record_by_lang: dict[str, dict]) -> list[tuple[str, str]]:
    present = [lang for lang in LANGS if lang in record_by_lang]
    return [(state_lang, question_lang) for state_lang in present for question_lang in present
            if state_lang != question_lang]


def mixed_record(record_by_lang: dict[str, dict], state_lang: str, question_lang: str, split: str) -> dict:
    state_record = record_by_lang[state_lang]
    question_record = record_by_lang[question_lang]
    return {
        "id": f"{split}-mix-{state_record['category']}-{state_record['case_id'].split('-')[-1]}-{state_lang.lower()}-{question_lang.lower()}",
        "split": split,
        "task": state_record["task"],
        "category": state_record["category"],
        "lang": state_lang,
        "question_lang": question_lang,
        "source_lang": state_record.get("source_lang", "zh-Hans"),
        "case_id": state_record["case_id"],
        "style": "mix",
        "state": state_record["state"],
        "questions": question_record["questions"],
        "labels": state_record["labels"],
    }


def adversarial_record(record_by_lang: dict[str, dict], state_lang: str, question_lang: str, split: str) -> dict | None:
    """语言对抗：把 state 里“带数字的关键句”替换成另一语言的同句，制造跨语言线索。

    句数对不齐时（中英标点切分差异很大）退化为**追加**一句跨语言证据句，
    保证每个组合都能凑够样本量。
    """
    if state_lang == question_lang:
        return None
    base = record_by_lang[state_lang]
    other = record_by_lang[question_lang]
    base_parts = [s for s in SENT_SPLIT[state_lang].split(base["state"]) if s.strip()]
    other_parts = [s for s in SENT_SPLIT[question_lang].split(other["state"]) if s.strip()]
    state: str | None = None
    if len(base_parts) == len(other_parts):
        for index in range(len(base_parts) - 1, -1, -1):
            if NUM.search(base_parts[index]) and len(other_parts[index]) > 8:
                merged = base_parts[:index] + [other_parts[index]] + base_parts[index + 1:]
                state = "".join(merged)
                break
    if state is None:
        evidence = next((s.strip() for s in other_parts if NUM.search(s) and len(s.strip()) > 12), None)
        if evidence is None:
            return None
        state = f"{base['state'].rstrip()} {evidence}"
    if case_meta.residual_placeholders(state):
        return None
    return {
        "id": f"{split}-advlang-{base['category']}-{base['case_id'].split('-')[-1]}-{state_lang.lower()}-{question_lang.lower()}",
        "split": split,
        "task": base["task"],
        "category": base["category"],
        "lang": state_lang,
        "question_lang": question_lang,
        "source_lang": base.get("source_lang", "zh-Hans"),
        "case_id": base["case_id"],
        "style": "adv-lang",
        "state": state,
        "questions": base["questions"],
        "labels": base["labels"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", nargs="*", default=["train", "test"])
    parser.add_argument("--adversarial", action="store_true", help="同时生成语言对抗子集")
    parser.add_argument("--per-combo", type=int, default=0, help="每个组合最多保留多少条（0=不限）")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    for split in args.splits:
        if split not in SPLIT_DIRS:
            print(f"跳过未知 split: {split}")
            continue
        source_dir = ROOT / SPLIT_DIRS[split]
        label = SPLIT_LABEL[split]
        if not source_dir.exists():
            continue
        data = load_split(source_dir)
        mix_dir = source_dir / "mix"
        adv_dir = source_dir / "adv"
        mix_buffers: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
        adv_buffers: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
        aligned_cases = 0
        for category, cases in data.items():
            for case_id, by_lang in cases.items():
                if len(by_lang) < 2:
                    continue
                aligned_cases += 1
                for state_lang, question_lang in combos(by_lang):
                    mix_buffers[(category, state_lang, question_lang)].append(
                        mixed_record(by_lang, state_lang, question_lang, label))
                    if args.adversarial:
                        adv = adversarial_record(by_lang, state_lang, question_lang, label)
                        if adv:
                            adv_buffers[(category, state_lang, question_lang)].append(adv)
        print(f"{split}: 对齐案情 {aligned_cases}，混合组合文件 {len(mix_buffers)}，对抗文件 {len(adv_buffers)}")
        if args.dry_run:
            continue
        for (category, state_lang, question_lang), rows in sorted(mix_buffers.items()):
            mix_dir.mkdir(parents=True, exist_ok=True)
            path = mix_dir / f"10-mix.{category}.{state_lang}_{question_lang}.jsonl"
            if args.per_combo:
                rows = rows[: args.per_combo]
            with path.open("w", encoding="utf-8", newline="\n") as handle:
                for row in rows:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        if args.adversarial:
            for (category, state_lang, question_lang), rows in sorted(adv_buffers.items()):
                adv_dir.mkdir(parents=True, exist_ok=True)
                path = adv_dir / f"20-adv-lang.{category}.{state_lang}_{question_lang}.jsonl"
                with path.open("w", encoding="utf-8", newline="\n") as handle:
                    for row in rows:
                        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
