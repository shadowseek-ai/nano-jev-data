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
SPLIT_DIRS = {"train": "train", "post": "train/posttrain", "test": "test", "dev": "dev"}
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


def _combo_counts() -> tuple[dict, dict, dict]:
    """返回 (混合, 语言对抗, 其他对抗) 三类计数。

    混合与语言对抗只有**跨语言**组合（对角线恒为 0）；干扰项 / 极性 / 近似反例
    是**按语言**的子集，不参与 state×question 矩阵。
    """
    langs = LANGS
    mix = {lang: {q: 0 for q in langs} for lang in langs}
    lang_adv = {lang: {q: 0 for q in langs} for lang in langs}
    per_lang: dict[str, dict[str, int]] = {}
    for split in ("train", "post", "test", "dev"):
        folder = ROOT / SPLIT_DIRS[split]
        if not folder.exists():
            continue
        for path in sorted((folder / "mix").glob("*.jsonl")) if (folder / "mix").exists() else []:
            for line in path.open(encoding="utf-8"):
                if line.strip():
                    record = json.loads(line)
                    mix[record["lang"]][record["question_lang"]] += 1
        for path in sorted((folder / "adv").glob("*.jsonl")) if (folder / "adv").exists() else []:
            for line in path.open(encoding="utf-8"):
                if not line.strip():
                    continue
                record = json.loads(line)
                if record.get("adv_type") == "lang":
                    lang_adv[record["lang"]][record["question_lang"]] += 1
                else:
                    bucket = per_lang.setdefault(record.get("adv_type") or "unknown", {})
                    bucket[record["lang"]] = bucket.get(record["lang"], 0) + 1
    for path in sorted((ROOT / "train" / "nearmiss").glob("*.jsonl")):
        for line in path.open(encoding="utf-8"):
            if line.strip():
                record = json.loads(line)
                bucket = per_lang.setdefault("near-miss", {})
                bucket[record["lang"]] = bucket.get(record["lang"], 0) + 1
    return mix, lang_adv, per_lang


def _matrix(counter: dict, label: str) -> str:
    header = f"| {label} \\\\ 问题语言 | " + " | ".join(f"`{q}`" for q in LANGS) + " |"
    lines = [header, "|---|" + "---:|" * len(LANGS)]
    for state_lang in LANGS:
        cells = []
        for q in LANGS:
            value = counter[state_lang][q]
            cells.append("—" if state_lang == q or value == 0 else f"{value:,}")
        lines.append(f"| `{state_lang}` | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def combo_stats_block() -> str:
    mix, _, _ = _combo_counts()
    return _matrix(mix, "state 语言")


def lang_adv_block() -> str:
    _, lang_adv, _ = _combo_counts()
    return _matrix(lang_adv, "state 语言")


def adv_type_block() -> str:
    _, _, per_lang = _combo_counts()
    labels = {"distractor": "A 干扰项", "polarity": "C 极性对照", "near-miss": "近似反例"}
    lines = ["| 类型 | " + " | ".join(f"`{lang}`" for lang in LANGS) + " | 合计 |",
             "|---|" + "---:|" * (len(LANGS) + 1)]
    for adv_type in ("distractor", "polarity", "near-miss"):
        counts = per_lang.get(adv_type, {})
        if not counts:
            continue
        total = sum(counts.values())
        lines.append(f"| {labels[adv_type]} | " +
                     " | ".join(f"{counts.get(lang, 0):,}" for lang in LANGS) + f" | {total:,} |")
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
    blocks = {
        "combo-stats": combo_stats_block(),
        "combo-lang-stats": lang_adv_block(),
        "adv-type-stats": adv_type_block(),
    }
    for marker, block in blocks.items():
        text = re.sub(
            rf"(<!-- {marker}:start -->)(.*?)(<!-- {marker}:end -->)",
            lambda m, b=block: f"{m.group(1)}\n{b}\n{m.group(3)}",
            text,
            flags=re.S,
        )
    README.write_text(text, encoding="utf-8", newline="\n")
    print(f"README 更新：{manifest['dataset_version']} records={total_records} questions={total_questions} files={len(files)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
