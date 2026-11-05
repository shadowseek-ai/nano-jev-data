"""清理跨 split 重复与简体纯度问题。

1. 以 ``train`` 的 state 指纹为基准，把 test/dev/post 中重复的**整个案情**（三语一起）删掉，
   保持三语条数一致；
2. 把 zh-Hans 里混入的繁体字用 OpenCC ``t2s`` 规整（同步更新 zh-Hant 为 ``s2twp`` 结果）。
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import topup_pretrain as core  # noqa: E402

SPLITS = ("train", "test", "dev", "train/posttrain")


def load(folder: Path) -> dict[str, dict[str, dict]]:
    data: dict[str, dict[str, dict]] = {}
    for path in sorted(folder.glob("00-llm.*.jsonl")):
        parts = path.stem.split(".")
        category, lang = parts[1], parts[2]
        for line in path.open(encoding="utf-8"):
            if not line.strip():
                continue
            record = json.loads(line)
            data.setdefault(category, {}).setdefault(record.get("case_id") or record["id"], {})[lang] = record
    return data


def write(folder: Path, category: str, grouped: dict[str, dict[str, dict]]) -> None:
    """grouped: {lang: {case_id: record}}"""
    for lang, cases in grouped.items():
        path = folder / f"00-llm.{category}.{lang}.jsonl"
        lines = [json.dumps(record, ensure_ascii=False) for record in cases.values()]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def main() -> int:
    base: dict[str, dict[str, dict]] = {}
    for relative in SPLITS:
        folder = ROOT / relative
        if not folder.exists():
            continue
        base[relative] = load(folder)

    train_prints: dict[str, set[str]] = {}
    for category, cases in base["train"].items():
        prints = set()
        for by_lang in cases.values():
            for record in by_lang.values():
                prints.add(core.fingerprint(record["state"]))
        train_prints[category] = prints

    removed: Counter = Counter()
    for relative in SPLITS:
        folder = ROOT / relative
        data = base[relative]
        for category, cases in data.items():
            keep: dict[str, dict[str, dict]] = {}
            for case_id, by_lang in cases.items():
                dotted = any(core.fingerprint(record["state"]) in train_prints.get(category, set())
                             for record in by_lang.values())
                # train 只与其它 split 去重；同 split 内重复一并清理
                if relative != "train" and dotted:
                    removed[f"{relative}/{category}"] += 1
                    continue
                have = {lang: by_lang[lang] for lang in by_lang}
                if len({core.fingerprint(r["state"]) for r in have.values()}) < len(have):
                    removed[f"{relative}/{category}/lang-dup"] += 1
                keep[case_id] = have
            data[category] = keep

    # 同 split 内去重（以 zh-Hans 指纹为准）
    for relative in SPLITS:
        for category, cases in base[relative].items():
            seen: set[str] = set()
            for case_id in list(cases):
                records = cases[case_id]
                marker = records.get("zh-Hans") or next(iter(records.values()))
                digest = core.fingerprint(marker["state"])
                if digest in seen:
                    del cases[case_id]
                    removed[f"{relative}/{category}/dup"] += 1
                    continue
                seen.add(digest)

    # 简体规整
    fixed = 0
    for relative in SPLITS:
        for category, cases in base[relative].items():
            for case_id, records in cases.items():
                hans = records.get("zh-Hans")
                if hans is None:
                    continue
                original = hans["state"]
                if core.is_simplified(original):
                    continue
                converted = core.T2S.convert(original)
                if converted == original:
                    continue
                hans["state"] = converted
                if "zh-Hant" in records:
                    records["zh-Hant"]["state"] = core.S2T.convert(converted)
                fixed += 1

    for relative in SPLITS:
        folder = ROOT / relative
        for category, cases in base[relative].items():
            grouped: dict[str, dict[str, dict]] = {}
            for case_id, records in cases.items():
                for lang, record in records.items():
                    grouped.setdefault(lang, {})[case_id] = record
            write(folder, category, grouped)

    print("removed", dict(removed))
    print("simplified fixed", fixed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
