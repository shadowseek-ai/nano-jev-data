"""语言对抗子集（§3.1）：同案同标签，把「关键判定线索」换成另一种语言的原文。

与 `mix/` 的区别
----------------
`mix/` 只是把 state 与 questions 换语言组合，state 文本本身没变；
本脚本**改写 state 文本**：从同案的另一语言版本里找出「与目标句数字签名一致」的句子，
把目标句替换成那一句，于是判定线索只出现在另一种语言里。

输出：
``train/adv/20-adv-lang.<category>.<stateLang>_<questionLang>.jsonl``
``test/adv/...``、``dev/adv/...``

每条记录 `case_id` 加 `-adv` 后缀、`style = "lang-adversarial"`，与 `mix` 既不同键也不同文。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import case_meta  # noqa: E402
import build_mixed as bm  # noqa: E402

SPLIT_DIRS = {"train": "train", "test": "test", "dev": "dev"}
NUM = re.compile(r"\d+(?:\.\d+)?")
DECISION = re.compile(r"(判定|结论|因此|所以|故|threshold|conclusion|decision|verdict|判定为|档|等级|评级)", re.I)


def sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[。；！？!?;\n])", text)
    return [part for part in parts if part.strip()]


def signature(text: str) -> tuple[str, ...]:
    return tuple(sorted(NUM.findall(text)))


def _digest(record: dict) -> str:
    payload = record["state"] + "\x00" + json.dumps(record["questions"], sort_keys=True, ensure_ascii=False)
    return hashlib.blake2b(payload.encode(), digest_size=8).hexdigest()


def pick_target(parts: list[str]) -> int | None:
    best, score = None, -1
    for index, sentence in enumerate(parts):
        if len(sentence.strip()) < 10 or not NUM.search(sentence):
            continue
        numbers = len(signature(sentence))
        weight = 4 * numbers
        if DECISION.search(sentence):
            weight += 6
        if 20 <= len(sentence) <= 220:
            weight += 1
        if index == 0:
            weight -= 3
        if weight > score:
            best, score = index, weight
    return best if score > 0 else None


def rewrite(base: str, donor: str) -> str | None:
    base_parts = sentences(base)
    donor_parts = sentences(donor)
    if len(base_parts) < 2 or not donor_parts:
        return None
    target = pick_target(base_parts)
    if target is None:
        return None
    want = signature(base_parts[target])
    matches = [part for part in donor_parts if signature(part) == want]
    if len(matches) != 1:
        # 退化：用数字集合包含关系找唯一候选
        supersets = [part for part in donor_parts if want and set(want) <= set(signature(part))
                     and len(signature(part)) <= len(want) + 1]
        if len(supersets) != 1:
            return None
        matches = supersets
    replacement = matches[0].strip()
    if not replacement or replacement == base_parts[target].strip():
        return None
    new_parts = list(base_parts)
    new_parts[target] = replacement + (" " if not replacement.endswith(("。", "；", "！", "？", "\n")) else "")
    text = "".join(new_parts)
    return text if text != base else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", nargs="*", default=["train", "test", "dev"])
    parser.add_argument("--per-combo", type=int, default=400)
    args = parser.parse_args(argv)

    for split in args.splits:
        folder = ROOT / SPLIT_DIRS[split]
        data = bm.load_split(folder)
        cap = args.per_combo if split == "train" else max(60, args.per_combo // 4)
        buckets: dict[tuple[str, str, str], list[str]] = defaultdict(list)
        counts: Counter = Counter()
        for category, cases in data.items():
            for case_id, by_lang in cases.items():
                present = [lang for lang in ("zh-Hans", "zh-Hant", "en") if lang in by_lang]
                if len(present) < 2:
                    continue
                for state_lang in present:
                    for question_lang in present:
                        if state_lang == question_lang:
                            continue
                        if counts[(category, state_lang, question_lang)] >= cap:
                            continue
                        others = [lang for lang in present if lang != state_lang]
                        donor_lang = others[0] if len(others) == 1 else (
                            others[1] if question_lang == others[0] else others[0])
                        new_state = rewrite(by_lang[state_lang]["state"], by_lang[donor_lang]["state"])
                        if not new_state:
                            continue
                        if case_meta.residual_placeholders(new_state):
                            continue
                        record = {
                            "id": f"{split}-advlang-{category}-{case_id.split('-')[-1]}-{state_lang.lower()}-{question_lang.lower()}",
                            "split": split,
                            "task": by_lang[state_lang]["task"],
                            "category": category,
                            "lang": state_lang,
                            "question_lang": question_lang,
                            "source_lang": by_lang[state_lang].get("source_lang", "zh-Hans"),
                            "case_id": f"{case_id}-adv",
                            "derived_from": case_id,
                            "style": "lang-adversarial",
                            "adv_type": "lang",
                            "state": new_state,
                            "questions": by_lang[question_lang]["questions"],
                            "labels": by_lang[state_lang]["labels"],
                        }
                        buckets[(category, state_lang, question_lang)].append(json.dumps(record, ensure_ascii=False))
                        counts[(category, state_lang, question_lang)] += 1
        out_dir = folder / "adv"
        if not buckets:
            print(f"{split}: 无可用对抗样本")
            continue
        # 去重：与 mix / 单语内容重复、或 adv 内部重复的直接丢弃（审计 §5 要求内容指纹重复 0）
        blocked = set()
        for pattern in ("train/mix/*.jsonl", "test/mix/*.jsonl", "dev/mix/*.jsonl",
                        "train/00-llm.*.jsonl", "test/00-llm.*.jsonl", "dev/00-llm.*.jsonl",
                        "train/posttrain/*.jsonl"):
            for src in ROOT.glob(pattern):
                for line in src.open(encoding="utf-8"):
                    if line.strip():
                        blocked.add(_digest(json.loads(line)))
        seen = set()
        dropped = 0
        for key, rows in list(buckets.items()):
            kept = []
            for raw in rows:
                fingerprint = _digest(json.loads(raw))
                if fingerprint in blocked or fingerprint in seen:
                    dropped += 1
                    continue
                seen.add(fingerprint)
                kept.append(raw)
            buckets[key] = kept
        out_dir.mkdir(parents=True, exist_ok=True)
        for old in out_dir.glob("*.jsonl"):
            old.unlink()
        for (category, state_lang, question_lang), rows in buckets.items():
            if not rows:
                continue
            path = out_dir / f"20-adv-lang.{category}.{state_lang}_{question_lang}.jsonl"
            path.write_text("\n".join(rows) + "\n", encoding="utf-8", newline="\n")
        total = sum(len(rows) for rows in buckets.values())
        print(f"{split}: {total} 条（去重丢弃 {dropped}），组合 {len(buckets)} 个，"
              f"每组合 {min(len(r) for r in buckets.values())}–{max(len(r) for r in buckets.values())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
