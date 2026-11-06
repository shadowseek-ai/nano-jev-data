"""类型 A：干扰项对抗（`10-advdist.*`）。

做法（全确定性，不调用 LLM）
---------------------------
对每条记录取 choice 题，在 state 里注入 1–3 句「提到**错误选项**关键词」的旁注：
案情事实不变、`labels` 逐字段不变，只让文中出现另一个候选类别的字眼。

- 至少注入「与正确选项用词最相似」的那个错误选项（近义混淆），提高干扰强度；
- 注入位置按序号轮转到 句首 / 句中 / 句末，各约 1/3；
- `case_id` = `<原 case_id>-dist<N>`，并加 `derived_from` / `adv_type="distractor"`。
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import case_meta  # noqa: E402

LANGS = ("zh-Hans", "zh-Hant", "en")
SPLIT_DIRS = {"train": "train", "test": "test", "dev": "dev"}

TEMPLATE = {
    "zh-Hans": {
        "lead": "补充一句：有人怀疑问题可能出在「{opt}」上。",
        "trail": "顺带一提，另一方曾提出「{opt}」这种可能。",
        "sep": "",
    },
    "zh-Hant": {
        "lead": "補充一句：有人懷疑問題可能出在「{opt}」上。",
        "trail": "順帶一提，另一方曾提出「{opt}」這種可能。",
        "sep": "",
    },
    "en": {
        "lead": "A side note: someone suspected the issue might be \"{opt}\".",
        "trail": "Incidentally, the other side raised the possibility of \"{opt}\".",
        "sep": " ",
    },
}


def choice_question(record: dict) -> tuple[str, dict] | None:
    for name, question in record["questions"].items():
        if question["type"] == "choice":
            return name, question
    return None


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[。；！？!?;.\n])", text)
    return [part for part in parts if part.strip()]


def similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b).ratio()


def inject(state: str, lang: str, options: list[str], position: int) -> str | None:
    if not options:
        return None
    pieces = split_sentences(state)
    if not pieces:
        return None
    template = TEMPLATE[lang]
    lead = template["lead"].format(opt=options[0])
    extra = template["trail"].format(opt=options[1]) if len(options) > 1 else ""
    sep = template["sep"]

    def glue(left: str, right: str) -> str:
        if not sep:
            return left + right
        return left.rstrip() + sep + right.lstrip()

    if position == 0:
        text = glue(lead, state)
        return glue(text, extra) if extra else text
    if position == 1:
        middle = max(1, len(pieces) // 2)
        pieces.insert(middle, lead + sep)
        if extra:
            pieces.append(extra)
        return "".join(pieces)
    text = glue(state, lead)
    return glue(text, extra) if extra else text


def load_file(path: Path) -> dict[str, dict]:
    table: dict[str, dict] = {}
    if not path.exists():
        return table
    for line in path.open(encoding="utf-8"):
        if line.strip():
            record = json.loads(line)
            table[record["case_id"]] = record
    return table


def build_language(folder: Path, category: str) -> dict[str, dict[str, dict]]:
    return {lang: load_file(folder / f"00-llm.{category}.{lang}.jsonl") for lang in LANGS}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", nargs="*", default=["train", "test"])
    parser.add_argument("--per-category", type=int, default=400)
    parser.add_argument("--test-per-category", type=int, default=100)
    parser.add_argument("--near-share", type=float, default=0.40,
                        help="近义混淆（干扰项与正确选项用词高度相似）占比，需 ≥0.30")
    args = parser.parse_args(argv)

    for split in args.splits:
        folder = ROOT / SPLIT_DIRS[split]
        cap = args.per_category if split == "train" else args.test_per_category
        buckets: dict[tuple[str, str], list[str]] = defaultdict(list)
        counts: Counter = Counter()
        near_count = Counter()
        for path in sorted(folder.glob("00-llm.*.zh-Hans.jsonl")):
            category = path.stem.split(".")[1]
            table = build_language(folder, category)

            # 第一遍：算每条记录「最相似的错误选项」相似度，用于挑近义混淆样本
            candidates: list[tuple[float, str]] = []
            for case_id, record in table["zh-Hans"].items():
                built = choice_question(record)
                if built is None:
                    continue
                name, question = built
                gold = record["labels"].get(name)
                if gold not in question["criteria"]:
                    continue
                wrong = [v for k, v in question["criteria"].items() if k != gold]
                if len(wrong) < 2:
                    continue
                best = max(similarity(question["criteria"][gold], text) for text in wrong)
                candidates.append((best, case_id))
            if not candidates:
                continue
            candidates.sort(key=lambda item: (-item[0], item[1]))
            near_quota = int(round(cap * args.near_share))
            near_set = {case_id for _, case_id in candidates[:near_quota] if _ >= 0.0}
            chosen: list[str] = [case_id for _, case_id in candidates[:near_quota]]
            for _, case_id in candidates[near_quota:]:
                if len(chosen) >= cap:
                    break
                chosen.append(case_id)
            chosen = chosen[:cap]

            # 第二遍：按位置/句数规则注入，并落盘
            for index, case_id in enumerate(chosen):
                record = table["zh-Hans"][case_id]
                name, _ = choice_question(record)
                volume = 1 + index % 3          # 1–3 句
                position = index % 3            # 句首 / 句中 / 句末
                states: dict[str, str] = {}
                similarity_used = None
                ok = True
                for lang in LANGS:
                    lang_record = table[lang].get(case_id)
                    if lang_record is None:
                        ok = False
                        break
                    lang_question = lang_record["questions"].get(name)
                    if lang_question is None or lang_question["type"] != "choice":
                        ok = False
                        break
                    lang_gold = lang_record["labels"].get(name)
                    if lang_gold not in lang_question["criteria"]:
                        ok = False
                        break
                    gold_text = lang_question["criteria"][lang_gold]
                    options = [(k, v) for k, v in lang_question["criteria"].items() if k != lang_gold]
                    if len(options) < 2:
                        ok = False
                        break
                    ordered = sorted(options, key=lambda item: similarity(gold_text, item[1]), reverse=True)
                    if similarity_used is None:
                        similarity_used = round(similarity(gold_text, ordered[0][1]), 3)
                    picks = [item[1] for item in ordered[: min(volume, len(ordered))]]
                    new_state = inject(lang_record["state"], lang, picks, position)
                    if not new_state or new_state == lang_record["state"]:
                        ok = False
                        break
                    if case_meta.residual_placeholders(new_state):
                        ok = False
                        break
                    states[lang] = new_state
                if not ok:
                    continue
                counts[category] += 1
                near_count[category] += int(case_id in near_set)
                for lang in LANGS:
                    derived = dict(table[lang][case_id])
                    derived["id"] = derived["id"].replace(f"{split}-", f"{split}-advdist-", 1)
                    derived["case_id"] = f"{case_id}-dist{counts[category]:04d}"
                    derived["derived_from"] = case_id
                    derived["style"] = "adv-distractor"
                    derived["adv_type"] = "distractor"
                    derived["adv_position"] = ("start", "middle", "end")[position]
                    derived["adv_injected"] = volume
                    derived["adv_similarity"] = similarity_used
                    derived["adv_near_synonym"] = bool(case_id in near_set)
                    derived["state"] = states[lang]
                    buckets[(category, lang)].append(json.dumps(derived, ensure_ascii=False))

        out_dir = folder / "adv"
        out_dir.mkdir(parents=True, exist_ok=True)
        for old in out_dir.glob("10-advdist.*.jsonl"):
            old.unlink()
        for (category, lang), rows in buckets.items():
            path = out_dir / f"10-advdist.{category}.{lang}.jsonl"
            path.write_text("\n".join(rows) + "\n", encoding="utf-8", newline="\n")
        total = sum(len(rows) for rows in buckets.values())
        per_cat = Counter()
        for (category, _), rows in buckets.items():
            per_cat[category] += len(rows)
        share = sum(near_count.values()) / max(1, sum(counts.values()))
        print(f"{split}: {total} 条，覆盖 {len(per_cat)} 垂类，"
              f"每垂类 {min(per_cat.values())}–{max(per_cat.values())}，近义混淆 {share:.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
