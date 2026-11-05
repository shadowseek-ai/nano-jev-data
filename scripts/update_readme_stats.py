"""按 MANIFEST.json 与数据文件重写 README 顶部的统计表。

只改数字与版本行，不改结构，避免手工改漏。
"""

from __future__ import annotations

import json
import re
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
LANGS = ("zh-Hans", "zh-Hant", "en")
SPLITS = (("train", "预训练"), ("post", "后训练"), ("test", "测试"), ("dev", "开发"))
CATEGORY_LABELS = {
    "finance_insurance": "金融与保险",
    "retail_ecommerce": "零售与电商",
    "supply_chain_logistics": "供应链与物流",
    "life_sciences_lab": "生命科学（实验自动化）",
    "healthcare_logistics": "医疗后勤（探索性）",
    "manufacturing_quality": "制造业与工业质检",
    "energy_utilities": "能源与公用事业",
    "telecom_network_ops": "电信与网络运维",
    "public_sector_services": "公共部门与政务服务",
    "hr_recruiting": "人力资源与招聘",
    "agent_tool_governance": "数据治理与 Agent 运维",
    "content_safety_ops": "内容安全与合规",
    "embodied_game_ops": "游戏与具身智能",
}


def lang_split_questions() -> dict[str, dict[str, int]]:
    """每种语言 × 每个 split 的题目数。"""
    table = {lang: {split: 0 for split, _ in SPLITS} for lang in LANGS}
    for split, _ in SPLITS:
        folder = ROOT / ("train/posttrain" if split == "post" else split)
        if not folder.exists():
            continue
        for path in sorted(folder.glob("*.jsonl")):
            for line in path.open(encoding="utf-8"):
                if not line.strip():
                    continue
                record = json.loads(line)
                table[record.get("lang")][split] += len(record.get("questions", {}))
    return table


def lang_stats_block() -> str:
    table = lang_split_questions()
    totals = {split: sum(table[lang][split] for lang in LANGS) for split, _ in SPLITS}
    lines = ["| 语言 | " + " | ".join(label for _, label in SPLITS) + " | 合计 |",
             "|---|" + "---:|" * (len(SPLITS) + 1)]
    for lang in LANGS:
        total = sum(table[lang][split] for split, _ in SPLITS)
        lines.append(f"| `{lang}` | " + " | ".join(f"{table[lang][split]:,}" for split, _ in SPLITS) + f" | {total:,} |")
    lines.append("| **合计** | " + " | ".join(f"**{totals[split]:,}**" for split, _ in SPLITS) +
                 f" | **{sum(totals.values()):,}** |")
    return "\n".join(lines)


def combo_stats_block() -> str:
    """state 语言 × 问题语言 的组合条数（含混合分片与对抗子集）。"""
    langs = LANGS
    combine = {lang: {q: 0 for q in langs} for lang in langs}
    adv = {lang: {q: 0 for q in langs} for lang in langs}
    for split in ("train", "test", "dev"):
        folder = ROOT / split
        if not folder.exists():
            continue
        for path in folder.glob("00-llm.*.jsonl"):
            rel = path.relative_to(ROOT).as_posix()
            if rel.startswith(("train/mix", "train/adv", "test/mix", "test/adv")):
                continue
            for line in path.open(encoding="utf-8"):
                if not line.strip():
                    continue
                record = json.loads(line)
                combine[record.get("lang")][record.get("question_lang", record.get("lang"))] += 1
        for path in list((ROOT / split / "mix").glob("*.jsonl")) if (ROOT / split / "mix").exists() else []:
            for line in path.open(encoding="utf-8"):
                if line.strip():
                    record = json.loads(line)
                    combine[record["lang"]][record["question_lang"]] += 1
        for path in list((ROOT / split / "adv").glob("*.jsonl")) if (ROOT / split / "adv").exists() else []:
            for line in path.open(encoding="utf-8"):
                if line.strip():
                    record = json.loads(line)
                    adv[record["lang"]][record["question_lang"]] += 1
    header = "| state 语言 \\ 问题语言 | " + " | ".join(f"`{q}`" for q in langs) + " |"
    lines = [header, "|---|" + "---:|" * len(langs)]
    for state_lang in langs:
        cells = []
        for q in langs:
            cells.append(f"{combine[state_lang][q]:,} / {adv[state_lang][q]:,}")
        lines.append(f"| `{state_lang}` | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> int:
    manifest = json.loads((ROOT / "MANIFEST.json").read_text(encoding="utf-8"))
    counts: dict[str, Counter] = {}
    lang_records: Counter = Counter()
    for relative in ("train", "train/posttrain", "test", "dev"):
        folder = ROOT / relative
        if not folder.exists():
            continue
        counter: Counter = Counter()
        for path in sorted(folder.glob("*.jsonl")):
            for line in path.open(encoding="utf-8"):
                if not line.strip():
                    continue
                record = json.loads(line)
                counter["records"] += 1
                counter["questions"] += len(record.get("questions", {}))
                lang_records[record.get("lang")] += 1
        counts[relative] = counter

    files = [p for d in ("train", "test", "dev") for p in (ROOT / d).rglob("*.jsonl")]
    total_bytes = sum(path.stat().st_size for path in files)
    total_records = sum(c["records"] for c in counts.values())
    total_questions = sum(c["questions"] for c in counts.values())
    frozen = datetime.fromisoformat(manifest["frozen_at"]).astimezone(timezone(timedelta(hours=8)))
    updated = frozen.strftime("%Y-%m-%d %H:%M (+08:00)")
    size_mb = total_bytes / 1024 / 1024

    text = README.read_text(encoding="utf-8")
    text = re.sub(r"\| 版本 \| `[^`]+` \|", f"| 版本 | `{manifest['dataset_version']}` |", text)
    text = re.sub(r"\| 更新时间 \| [^|]+ \|", f"| 更新时间 | {updated} |", text)
    text = re.sub(
        r"\| 规模 \| \*\*[^*]+\*\*[^|]*\|",
        f"| 规模 | **{total_records:,} 条记录 / {total_questions:,} 题**（三语合计），13 个垂类，{len(files)} 个文件，约 {size_mb:.0f} MB |",
        text,
    )
    rows = {
        "pre": f"| 预训练 | `train/` | {counts['train']['records']:,} | {counts['train']['questions']:,} |",
        "post": f"| 后训练 | `train/posttrain/` | {counts['train/posttrain']['records']:,} | {counts['train/posttrain']['questions']:,} |",
        "test": f"| 测试 | `test/` | {counts['test']['records']:,} | {counts['test']['questions']:,} |",
        "dev": f"| 开发 | `dev/` | {counts['dev']['records']:,} | {counts['dev']['questions']:,} |",
    }
    text = re.sub(r"\| 预训练 \| `train/` \|[^\n]*", rows["pre"], text)
    text = re.sub(r"\| 后训练 \| `train/posttrain/` \|[^\n]*", rows["post"], text)
    text = re.sub(r"\| 测试 \| `test/` \|[^\n]*", rows["test"], text)
    text = re.sub(r"\| 开发 \| `dev/` \|[^\n]*", rows["dev"], text)
    text = re.sub(
        r"语言分布（记录）：[^\n]*",
        "语言分布（记录）：" + "、".join(f"`{lang}` {lang_records[lang]:,}" for lang in LANGS) + "。",
        text,
    )
    categories = sorted(CATEGORY_LABELS)
    category_line = "、".join(f"{CATEGORY_LABELS[name]}（`{name}`）" for name in categories)
    text = re.sub(r"13 个垂类：[^\n]*(?:\n[^\n]*)*?\n\n", f"13 个垂类：{category_line}。\n\n", text, count=1)
    block = lang_stats_block()
    text = re.sub(
        r"(<!-- lang-stats:start -->)(.*?)(<!-- lang-stats:end -->)",
        lambda m: f"{m.group(1)}\n{block}\n{m.group(3)}",
        text,
        flags=re.S,
    )
    combo_block = combo_stats_block()
    text = re.sub(
        r"(<!-- combo-stats:start -->)(.*?)(<!-- combo-stats:end -->)",
        lambda m: f"{m.group(1)}\n{combo_block}\n{m.group(3)}",
        text,
        flags=re.S,
    )
    README.write_text(text, encoding="utf-8", newline="\n")
    print(f"README 更新：{manifest['dataset_version']} records={total_records} questions={total_questions} files={len(files)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
