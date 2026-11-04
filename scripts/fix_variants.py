"""修正种子的繁简标注：zh-Hans 一律转简体、zh-Hant 一律转繁体，并报告 en 里的中文字符。

背景：自己手写的 batch 里出现过「lang=zh-Hans 但正文是繁体」的错标（子代理审计发现）。
标签不受影响（只改字形），因此可以机械修正。
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from opencc import OpenCC

ROOT = Path(__file__).resolve().parents[2]
SEEDS = ROOT / "artifacts" / "llm" / "seeds"
CJK = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")


def main() -> int:
    t2s, s2t = OpenCC("t2s"), OpenCC("s2t")
    totals = {"files": 0, "records": 0, "fixed_hans": 0, "fixed_hant": 0, "cjk_in_en": 0}
    for path in sorted(SEEDS.rglob("*.jsonl")):
        raw = path.read_text(encoding="utf-8")
        lines = [line for line in raw.splitlines() if line.strip()]
        out: list[str] = []
        fixed_hans = fixed_hant = cjk_en = 0
        for line in lines:
            record = json.loads(line)
            lang = record.get("lang")
            state = record.get("state", "")
            if lang == "zh-Hans":
                converted = t2s.convert(state)
                if converted != state:
                    record["state"] = converted
                    fixed_hans += 1
            elif lang == "zh-Hant":
                converted = s2t.convert(state)
                if converted != state:
                    record["state"] = converted
                    fixed_hant += 1
            elif lang == "en" and CJK.search(state):
                cjk_en += 1
                print(f"  ! {path.name}: en 记录含中文 {CJK.findall(state)[:5]}")
            out.append(json.dumps(record, ensure_ascii=False))
        if fixed_hans or fixed_hant:
            path.write_text("\n".join(out) + "\n", encoding="utf-8", newline="\n")
        totals["files"] += 1
        totals["records"] += len(lines)
        totals["fixed_hans"] += fixed_hans
        totals["fixed_hant"] += fixed_hant
        totals["cjk_in_en"] += cjk_en
        if fixed_hans or fixed_hant:
            print(
                f"{path.relative_to(ROOT)}  行数={len(lines)}  "
                f"简体修正={fixed_hans} 繁体修正={fixed_hant}"
            )
    print(totals)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
