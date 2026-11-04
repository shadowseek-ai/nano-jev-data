"""去重审计：对 seeds/ 与 seeds_api/ 的语料做精确 + 近似重复检测。

近似重复用 MinHash 分桶（每条约 200 个 5-gram），只对同桶候选做精确 Jaccard，
因此 10 万级记录仍可在几分钟内跑完，而不是 O(n²)。

用法：
    py artifacts/llm/dedupe_audit.py --report     # 只报告
    py artifacts/llm/dedupe_audit.py --apply      # 删除重复（保留首次出现）

报告：artifacts/llm/dedupe_report.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SEED_DIRS = (ROOT / "artifacts" / "llm" / "seeds", ROOT / "artifacts" / "llm" / "seeds_api")
REPORT = ROOT / "artifacts" / "llm" / "dedupe_report.json"
HASHES = 8
BANDS = 4
SHINGLE = 5


def normalize(text: str) -> str:
    return re.sub(r"\s+", "", text).lower()


def shingles(norm: str) -> set[str]:
    return {norm[i : i + SHINGLE] for i in range(max(len(norm) - SHINGLE + 1, 0))}


def signature(grams: set[str]) -> tuple[int, ...]:
    out = []
    for seed in range(HASHES):
        best = None
        for gram in grams:
            value = int.from_bytes(hashlib.blake2b(f"{seed}:{gram}".encode(), digest_size=4).digest(), "big")
            if best is None or value < best:
                best = value
        out.append(best or 0)
    return tuple(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--threshold", type=float, default=0.80)
    args = parser.parse_args(argv)

    files = sorted(p for folder in SEED_DIRS if folder.exists() for p in folder.rglob("*.jsonl"))
    rows_by_file: dict[Path, list[dict]] = {}
    drop: dict[Path, set[int]] = defaultdict(set)

    exact: dict[tuple[str, str], dict[str, str]] = defaultdict(dict)
    total = exact_dups = 0
    for path in files:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        rows_by_file[path] = rows
        for index, row in enumerate(rows):
            total += 1
            key = (row.get("category") or path.parent.name, row.get("lang", ""))
            digest = hashlib.blake2b(normalize(row.get("state", "")).encode(), digest_size=8).hexdigest()
            where = f"{path.relative_to(ROOT)}:{index}"
            if digest in exact[key]:
                exact_dups += 1
                drop[path].add(index)
            else:
                exact[key][digest] = where

    # MinHash 分桶（只对未判重的记录）
    entries: list[tuple[Path, int, str, str, set[str]]] = []
    for path, rows in rows_by_file.items():
        for index, row in enumerate(rows):
            if index in drop[path]:
                continue
            norm = normalize(row.get("state", ""))
            entries.append((path, index, row.get("category") or path.parent.name, row.get("lang", ""), shingles(norm)))

    buckets: dict[tuple[str, str, int, int], list[int]] = defaultdict(list)
    for position, (_, _, category, lang, grams) in enumerate(entries):
        sig = signature(grams)
        for band in range(BANDS):
            bucket = sig[band * (HASHES // BANDS)]
            buckets[(category, lang, band, bucket)].append(position)

    near: list[dict] = []
    compared: set[tuple[int, int]] = set()
    for items in buckets.values():
        if len(items) < 2:
            continue
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                a, b = items[i], items[j]
                pair = (a, b) if a < b else (b, a)
                if pair in compared:
                    continue
                compared.add(pair)
                ga, gb = entries[a][4], entries[b][4]
                if not ga or not gb:
                    continue
                jac = len(ga & gb) / len(ga | gb)
                if jac >= args.threshold and entries[b][1] not in drop[entries[b][0]]:
                    near.append({
                        "category": entries[b][2],
                        "lang": entries[b][3],
                        "keep": f"{entries[a][0].relative_to(ROOT)}:{entries[a][1]}",
                        "drop": f"{entries[b][0].relative_to(ROOT)}:{entries[b][1]}",
                        "jaccard": round(jac, 3),
                    })
                    drop[entries[b][0]].add(entries[b][1])

    if args.apply:
        for path, rows in rows_by_file.items():
            removed = drop.get(path, set())
            if not removed:
                continue
            kept = [json.dumps(row, ensure_ascii=False) for index, row in enumerate(rows) if index not in removed]
            path.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8", newline="\n")

    report = {
        "total_records": total,
        "exact_duplicates": exact_dups,
        "near_duplicates": len(near),
        "removed": sum(len(v) for v in drop.values()) if args.apply else 0,
        "applied": args.apply,
        "compared_pairs": len(compared),
        "samples": near[:30],
    }
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "samples"}, ensure_ascii=True, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
