"""打印 topup 进度与 ETA（只读，不修改数据）。"""

from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "artifacts" / "llm" / "topup"
TARGET = 3000
LANGS = ("zh-Hans", "zh-Hant", "en")


def main() -> int:
    specs = {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in (ROOT / "specs").glob("*.json") if not p.stem.startswith("_")}
    counts: Counter = Counter()
    for path in sorted((ROOT / "train").glob("*.jsonl")):
        category, lang = path.stem.split(".")[1], path.stem.split(".")[2]
        for line in path.open(encoding="utf-8"):
            if line.strip():
                counts[(category, lang)] += 1

    sources: dict[str, dict] = {}
    seeds = WORK / "seeds.jsonl"
    if seeds.exists():
        for line in seeds.open(encoding="utf-8"):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("kind") == "source":
                sources[row["source_id"]] = row
    translated = set()
    trans = WORK / "translations.jsonl"
    if trans.exists():
        for line in trans.open(encoding="utf-8"):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("kind") == "translation":
                translated.add((row["source_id"], row["lang"]))

    need = sum(max(0, TARGET - min(counts[(c, l)] for l in LANGS)) for c in specs)
    have = len(sources)
    en_done = sum(1 for sid in sources if (sid, "en") in translated)
    zh_done = sum(1 for sid in sources if (sid, "zh-Hant") in translated)
    print(f"需要新增源记录 {need}，已生成 {have}（{have / max(need, 1):.1%}）")
    print(f"译文：zh-Hant {zh_done}/{have}  en {en_done}/{have}")
    print(f"{'category':<24}{'zh-Hans':>9}{'zh-Hant':>9}{'en':>9}{'缺口':>7}")
    for category in sorted(specs):
        row = [counts[(category, lang)] for lang in LANGS]
        print(f"{category:<24}{row[0]:>9}{row[1]:>9}{row[2]:>9}{max(0, TARGET - min(row)):>7}")
    gen_log = WORK / "topup.log"
    if gen_log.exists():
        lines = [line.strip() for line in gen_log.read_text(encoding="utf-8", errors="replace").splitlines() if "生成 +" in line]
        if len(lines) >= 2:
            def stamp(line: str) -> float:
                return time.mktime(time.strptime(line[:19], "%Y-%m-%d %H:%M:%S"))

            window = lines[-200:]
            elapsed = max(stamp(window[-1]) - stamp(window[0]), 1)
            produced = sum(int(line.split("生成 +")[1].split()[0]) for line in window)
            rate = produced / elapsed * 3600
            remaining = max(0, need - have)
            print(f"最近生成速率 {rate:.0f} 条/小时，预计还需 {remaining / max(rate, 1):.1f} 小时")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
