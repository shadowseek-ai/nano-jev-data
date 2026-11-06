"""派生/对抗分片门禁：对齐 `dataset-adversarial-requirements.md §6/§8` 与审计文档 §5。

检查项
------
1. `adv` 与 `mix` 的 (task, case_id, lang, question_lang) 交集为 0；
2. 类型 B（`lang`）规模 2–4 万；类型 A（`distractor`）≥15,600；
3. 类型 A：`labels` 与原 case 逐字段相同，`adv_type="distractor"`，位置三档均分；
4. 类型 C：`pair_id`/`pair_role` 齐全，C1 标签不变、C2/C3 标签翻转，组数 ≥4,680；
5. `nearmiss` 与原案标签相同数为 0，`derived_from` 可追溯；
6. 内容指纹全库唯一；
7. train / test / dev / posttrain 的 `case_id` 交叉为 0；
8. `dev/mix` ≥3,900；
9. 占位符残留 0；结构与标签域合法。
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import case_meta  # noqa: E402

REQUIRED = ("id", "split", "task", "category", "lang", "question_lang",
            "source_lang", "case_id", "style", "state", "questions", "labels")


def digest(record: dict) -> str:
    payload = record["state"] + "\x00" + json.dumps(record["questions"], sort_keys=True, ensure_ascii=False)
    return hashlib.blake2b(payload.encode(), digest_size=8).hexdigest()


def load(pattern: str) -> list[tuple[str, dict]]:
    rows = []
    for path in sorted(ROOT.glob(pattern)):
        for line in path.open(encoding="utf-8"):
            if line.strip():
                rows.append((path.relative_to(ROOT).as_posix(), json.loads(line)))
    return rows


def main() -> int:
    problems: list[str] = []

    mix = load("train/mix/*.jsonl") + load("test/mix/*.jsonl") + load("dev/mix/*.jsonl")
    adv = load("train/adv/*.jsonl") + load("test/adv/*.jsonl") + load("dev/adv/*.jsonl")
    near = load("train/nearmiss/*.jsonl")
    single = (load("train/00-llm.*.jsonl") + load("test/00-llm.*.jsonl") + load("dev/00-llm.*.jsonl")
              + load("train/posttrain/*.jsonl"))

    by_type: Counter = Counter()
    for _, record in adv:
        by_type[record.get("adv_type", "missing")] += 1
    if by_type.get("missing"):
        problems.append(f"adv 缺少 adv_type {by_type['missing']} 条")

    mix_keys = {(r["task"], r["case_id"], r["lang"], r["question_lang"]) for _, r in mix}
    adv_keys = {(r["task"], r["case_id"], r["lang"], r["question_lang"]) for _, r in adv}
    cross = mix_keys & adv_keys
    if cross:
        problems.append(f"adv∩mix 键交集 {len(cross)}")

    lang_total = sum(1 for p, r in adv if r.get("adv_type") == "lang" and p.startswith("train/"))
    if not 20000 <= lang_total <= 40000:
        problems.append(f"train 类型 B 规模 {lang_total} 不在 2–4 万")
    dist_total = sum(1 for p, r in adv if r.get("adv_type") == "distractor" and p.startswith("train/"))
    if dist_total < 15600:
        problems.append(f"train 类型 A 规模 {dist_total} < 15,600")

    originals = {}
    for _, record in single:
        originals[(record["case_id"], record["lang"])] = record

    # 类型 A：标签不变 + 位置三档均分 + 近义占比
    dist = [(p, r) for p, r in adv if r.get("adv_type") == "distractor"]
    label_drift = 0
    positions = Counter()
    near_share = 0
    for _, record in dist:
        base = originals.get((record.get("derived_from"), record["lang"]))
        if base is not None and base["labels"] != record["labels"]:
            label_drift += 1
        positions[record.get("adv_position")] += 1
        near_share += int(bool(record.get("adv_near_synonym")))
    if label_drift:
        problems.append(f"类型 A 标签漂移 {label_drift} 条")
    if dist:
        total = len(dist)
        for name in ("start", "middle", "end"):
            share = positions[name] / total
            if not 0.25 <= share <= 0.42:
                problems.append(f"类型 A 位置 {name} 占比 {share:.1%} 偏离 1/3")
        if near_share / total < 0.30:
            problems.append(f"类型 A 近义混淆占比 {near_share / total:.1%} < 30%")

    # 类型 C：成对结构（按 (pair_id, lang) 计组，即每垂类每语言每组一条）
    pol = [(p, r) for p, r in adv if r.get("adv_type") == "polarity"]
    pairs = {}
    for _, record in pol:
        if "pair_id" not in record or "pair_role" not in record:
            problems.append("类型 C 缺少 pair_id/pair_role")
            break
        pairs.setdefault((record["pair_id"], record["lang"]), {})[record["pair_role"]] = record
    complete = 0
    bad_pairs = 0
    for (pair_id, lang), roles in pairs.items():
        if set(roles) != {"C1", "C2", "C3"}:
            bad_pairs += 1
            continue
        base_id = pair_id.rsplit("-pol", 1)[0]
        original = originals.get((base_id, lang))
        if original is None:
            bad_pairs += 1
            continue
        complete += 1
        c1, c2, c3 = roles["C1"], roles["C2"], roles["C3"]
        if c1["labels"] != original["labels"]:
            problems.append(f"类型 C C1 标签应不变: {pair_id}/{lang}")
        if c2["labels"] == original["labels"]:
            problems.append(f"类型 C C2 标签应翻转: {pair_id}/{lang}")
        if c3["labels"] == original["labels"]:
            problems.append(f"类型 C C3 标签应翻转: {pair_id}/{lang}")
        if c2["labels"] != c3["labels"]:
            problems.append(f"类型 C C2/C3 标签应一致: {pair_id}/{lang}")
    if complete < 4680:
        problems.append(f"类型 C 完整组数 {complete} < 4,680")
    if bad_pairs:
        problems.append(f"类型 C 结构不完整 {bad_pairs} 组")

    # nearmiss：标签必须翻转
    same = 0
    untraceable = 0
    for _, record in near:
        base = originals.get((record.get("derived_from"), record["lang"]))
        if base is None:
            untraceable += 1
            continue
        if base["labels"] == record["labels"]:
            same += 1
    if same:
        problems.append(f"nearmiss 标签未翻转 {same} 条")
    if untraceable:
        problems.append(f"nearmiss 无法追溯 {untraceable} 条")

    # 全库内容指纹唯一
    fingerprints = Counter()
    for _, record in mix + adv + near + single:
        fingerprints[digest(record)] += 1
    duplicated = sum(v - 1 for v in fingerprints.values() if v > 1)
    if duplicated:
        problems.append(f"内容指纹重复 {duplicated} 条")

    case_ids = {}
    for split in ("train", "train/posttrain", "test", "dev"):
        case_ids[split] = {record["case_id"] for _, record in load(f"{split}/00-llm.*.jsonl")}
    leaks = (case_ids["train"] & case_ids["test"]) | (case_ids["train"] & case_ids["dev"]) | (
        case_ids["test"] & case_ids["dev"]) | (case_ids["train/posttrain"] & case_ids["test"]) | (
        case_ids["train/posttrain"] & case_ids["dev"]) | (case_ids["train/posttrain"] & case_ids["train"])
    if leaks:
        problems.append(f"case_id 交叉 {len(leaks)}: {sorted(leaks)[:5]}")

    dev_mix = load("dev/mix/*.jsonl")
    if len(dev_mix) < 3900:
        problems.append(f"dev/mix 只有 {len(dev_mix)} 条")

    residual = sum(1 for _, record in mix + adv + near + single
                   if case_meta.residual_placeholders(record["state"]))
    if residual:
        problems.append(f"占位符残留 {residual} 条")

    bad = 0
    for _, record in mix + adv + near + single:
        if any(key not in record for key in REQUIRED):
            bad += 1
            continue
        labels = record["labels"]
        for name, question in record["questions"].items():
            kind = question.get("type")
            if kind == "choice":
                if name not in labels or labels[name] not in question["criteria"]:
                    bad += 1
                    break
            elif kind == "score":
                if name not in labels or not isinstance(labels[name], int) or not 0 <= labels[name] < len(question["criteria"]):
                    bad += 1
                    break
            elif kind == "noul":
                if name not in labels or labels[name] not in (0, 1):
                    bad += 1
                    break
            else:
                bad += 1
                break
    if bad:
        problems.append(f"结构/标签域不合规 {bad} 条")

    print(f"mix={len(mix)} adv={len(adv)} {dict(by_type)} nearmiss={len(near)} "
          f"single={len(single)} dev/mix={len(dev_mix)} 类型C完整组={complete}")
    if problems:
        print("problems:")
        for item in problems:
            print("  -", item)
        return 1
    print("problems=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
