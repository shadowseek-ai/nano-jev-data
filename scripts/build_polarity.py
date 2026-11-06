"""类型 C：极性/否定对照（`12-advpol.*`）。

对每个已校验的翻盘（`flips.jsonl`：把某个数字改掉 → 跨过阈值 → 标签翻转）产出一组配对样本：

| pair_role | 事实 | 结论句 | 标签 |
|---|---|---|---|
| `baseline` | 原样 | 原样 | 原标签 |
| `C1` | 不变 | 追加一句**否定**结论 | 原标签（事实为准）|
| `C2` | 改数字跨阈值 | 原样（结论句仍指向旧判定）| 翻转后标签 |
| `C3` | 改数字跨阈值 | 追加一句与事实一致的结论 | 翻转后标签 |

`pair_id` 把四条串起来，`adv_type="polarity"`，`style="adv-polarity"`。
`case_id` = `<原 case_id>-pol`（C1/C2/C3）与 `<原 case_id>-polbase`（baseline）。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import case_meta  # noqa: E402

LANGS = ("zh-Hans", "zh-Hant", "en")
SPLIT_DIRS = {"train": "train", "test": "test"}

NEGATIVE = {
    "zh-Hans": "补充结论：本单不构成「{text}」。",
    "zh-Hant": "補充結論：本單不構成「{text}」。",
    "en": "Supplementary conclusion: this case does not amount to \"{text}\".",
}
AFFIRM = {
    "zh-Hans": "补充结论：本单构成「{text}」。",
    "zh-Hant": "補充結論：本單構成「{text}」。",
    "en": "Supplementary conclusion: this case amounts to \"{text}\".",
}
RESTATE = {
    "zh-Hans": "结论仍维持：本单构成「{text}」。",
    "zh-Hant": "結論仍維持：本單構成「{text}」。",
    "en": "The conclusion still stands: this case amounts to \"{text}\".",
}
SEP = {"zh-Hans": "", "zh-Hant": "", "en": " "}


def replace_once(text: str, old: str, new: str) -> str | None:
    pattern = re.compile(rf"(?<!\d){re.escape(old)}(?!\d)")
    if len(pattern.findall(text)) != 1:
        return None
    return pattern.sub(new, text, count=1)


def choice_key(record: dict) -> str | None:
    for name, question in record["questions"].items():
        if question["type"] == "choice":
            return name
    return None


def append(text: str, addition: str, lang: str) -> str:
    return text.rstrip() + SEP[lang] + addition


def load_file(path: Path) -> dict[str, dict]:
    table: dict[str, dict] = {}
    if path.exists():
        for line in path.open(encoding="utf-8"):
            if line.strip():
                record = json.loads(line)
                table[record["case_id"]] = record
    return table


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", nargs="*", default=["train", "test"])
    parser.add_argument("--per-category", type=int, default=120)
    parser.add_argument("--work", default=str(ROOT / "artifacts" / "llm" / "nearmiss2"))
    args = parser.parse_args(argv)

    flips: dict[str, dict] = {}
    for line in (Path(args.work) / "flips.jsonl").open(encoding="utf-8"):
        if line.strip():
            row = json.loads(line)
            flips.setdefault(row["case_id"], row)
    print(f"翻盘缓存 {len(flips)} 条")

    for split in args.splits:
        folder = ROOT / SPLIT_DIRS[split]
        buckets: dict[tuple[str, str], list[str]] = defaultdict(list)
        counts: Counter = Counter()
        roles = Counter()
        for path in sorted(folder.glob("00-llm.*.zh-Hans.jsonl")):
            category = path.stem.split(".")[1]
            table = {lang: load_file(folder / f"00-llm.{category}.{lang}.jsonl") for lang in LANGS}
            for case_id, record in table["zh-Hans"].items():
                if counts[category] >= args.per_category:
                    break
                flip = flips.get(case_id)
                if flip is None:
                    continue
                key = choice_key(record)
                if key is None:
                    continue
                rows = {role: {} for role in ("baseline", "C1", "C2", "C3")}
                ok = True
                for lang in LANGS:
                    base = table[lang].get(case_id)
                    if base is None or key not in base["questions"]:
                        ok = False
                        break
                    question = base["questions"][key]
                    orig_label = base["labels"].get(key)
                    new_label = flip["labels"].get(key)
                    if orig_label not in question["criteria"] or new_label not in question["criteria"]:
                        ok = False
                        break
                    orig_text = question["criteria"][orig_label]
                    new_text = question["criteria"][new_label]
                    changed = replace_once(base["state"], flip["from"], flip["to"])
                    if changed is None:
                        ok = False
                        break
                    rows["baseline"][lang] = (base["state"], base["labels"])
                    rows["C1"][lang] = (append(base["state"], NEGATIVE[lang].format(text=orig_text), lang),
                                        base["labels"])
                    # C2：事实跨阈值，但结论句仍维持旧判定（显式复述，避免与 nearmiss 文本重合）
                    rows["C2"][lang] = (append(changed, RESTATE[lang].format(text=orig_text), lang),
                                        flip["labels"])
                    rows["C3"][lang] = (append(changed, AFFIRM[lang].format(text=new_text), lang),
                                        flip["labels"])
                if not ok:
                    continue
                if case_meta.residual_placeholders("".join(v[0] for role in rows.values() for v in role.values())):
                    continue
                counts[category] += 1
                pair_id = f"{case_id}-pol"
                for role, payload in rows.items():
                    roles[role] += 1
                    if role == "baseline":
                        continue  # baseline 与原单语记录完全一致，落盘会与单语分片重复
                    for lang in LANGS:
                        derived = dict(table[lang][case_id])
                        derived["id"] = derived["id"].replace(f"{split}-", f"{split}-advpol-", 1)
                        derived["case_id"] = f"{case_id}-pol"
                        derived["derived_from"] = case_id
                        derived["style"] = "adv-polarity"
                        derived["adv_type"] = "polarity"
                        derived["pair_id"] = pair_id
                        derived["pair_role"] = role
                        derived["state"] = payload[lang][0]
                        derived["labels"] = payload[lang][1]
                        buckets[(category, lang)].append(json.dumps(derived, ensure_ascii=False))
        out_dir = folder / "adv"
        out_dir.mkdir(parents=True, exist_ok=True)
        for old in out_dir.glob("12-advpol.*.jsonl"):
            old.unlink()
        for (category, lang), rows in buckets.items():
            path = out_dir / f"12-advpol.{category}.{lang}.jsonl"
            path.write_text("\n".join(rows) + "\n", encoding="utf-8", newline="\n")
        total = sum(len(v) for v in buckets.values())
        if not counts:
            print(f"{split}: 无可用翻盘样本，跳过")
            continue
        print(f"{split}: {total} 条，{sum(counts.values())} 组（C1/C2/C3 各一组），"
              f"覆盖 {len(counts)} 垂类，每垂类 {min(counts.values())}–{max(counts.values())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
