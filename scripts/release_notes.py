"""从 `MANIFEST.json` 生成 GitHub Release 说明，避免手写数字与实际数据漂移。

用法：
    python scripts/release_notes.py                 # 打印到 stdout
    python scripts/release_notes.py --out NOTES.md  # 写入文件
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LANGS = ("zh-Hans", "zh-Hant", "en")
SPLIT_LABEL = {"train": "预训练", "test": "测试", "dev": "开发"}
SPLIT_DIR = {"train": "train/", "test": "test/", "dev": "dev/"}
ADV_LABEL = {"distractor": "A 干扰项", "lang": "B 语言对抗", "polarity": "C 极性对照", "near-miss": "近似反例"}


def count_records(folder: Path) -> int:
    total = 0
    for path in sorted(folder.glob("*.jsonl")):
        total += sum(1 for line in path.open(encoding="utf-8") if line.strip())
    return total


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    manifest = json.loads((ROOT / "MANIFEST.json").read_text(encoding="utf-8"))
    version = manifest["dataset_version"]
    splits = manifest["splits"]

    single: dict[str, int] = {
        "train": count_records(ROOT / "train"),
        "post": count_records(ROOT / "train" / "posttrain"),
        "test": count_records(ROOT / "test"),
        "dev": count_records(ROOT / "dev"),
    }
    questions: Counter = Counter()
    for relative, key in (("train", "train"), ("train/posttrain", "post"), ("test", "test"), ("dev", "dev")):
        folder = ROOT / relative
        for path in sorted(folder.glob("*.jsonl")):
            for line in path.open(encoding="utf-8"):
                if line.strip():
                    questions[key] += len(json.loads(line).get("questions", {}))

    derived: dict[str, int] = defaultdict(int)
    for folder in ("train/mix", "train/posttrain/mix", "test/mix", "dev/mix"):
        path = ROOT / folder
        if path.exists():
            derived["mix"] += count_records(path)

    adv = manifest.get("adversarial", {})
    adv_total = Counter()
    for split, types in adv.items():
        for adv_type, counts in types.items():
            langs = {k: v for k, v in counts.items() if not k.startswith("category:")}
            adv_total[adv_type] += sum(langs.values())

    lines = [
        f"# nano-jev-data {version}",
        "",
        "三语（简体中文 / 繁体中文 / 英文）案情中心数据集 + 对抗子集，"
        "13 个垂类，每垂类 3 个任务（`noul` / `choice` / `score`）。",
        "",
        "## 单语分片（三语合计）",
        "",
        "| split | 目录 | 记录 | 题目 |",
        "|---|---|---:|---:|",
        f"| 预训练 | `train/` | {single['train']:,} | {questions['train']:,} |",
        f"| 后训练 | `train/posttrain/` | {single['post']:,} | {questions['post']:,} |",
        f"| 测试 | `test/` | {single['test']:,} | {questions['test']:,} |",
        f"| 开发 | `dev/` | {single['dev']:,} | {questions['dev']:,} |",
        "",
        "## 派生与对抗子集",
        "",
        "| 目录 / 文件 | 类型 | `adv_type` | `labels` | 条数 |",
        "|---|---|---|---|---:|",
        f"| `train/mix/`、`train/posttrain/mix/`、`test/mix/`、`dev/mix/` | 跨语言混合 | — | 不变 | {derived['mix']:,} |",
        f"| `train/adv/10-advdist.*`、`test/adv/10-advdist.*` | A 干扰项 | `distractor` | 不变 | {adv_total['distractor']:,} |",
        f"| `train/adv/20-adv-lang.*`、`test/adv/`、`dev/adv/` | B 语言对抗 | `lang` | 不变 | {adv_total['lang']:,} |",
        f"| `train/adv/12-advpol.*`、`test/adv/12-advpol.*` | C 极性对照 | `polarity` | C1 不变 / C2・C3 翻转 | {adv_total['polarity']:,} |",
        f"| `train/nearmiss/` | 近似反例 | — | 翻转 | {adv_total.get('near-miss', 0):,} |",
        "",
        "## 验收门槛（CI 每次推送自动校验）",
        "",
        "| 项 | 门槛 |",
        "|---|---|",
        "| `adv` 与 `mix` 键交集 | 0 |",
        "| 内容指纹重复（全库） | 0 |",
        "| train / test / dev / posttrain `case_id` 交叉 | 0 |",
        "| 类型 A 规模 | ≥ 15,600 |",
        "| 类型 B 规模 | 20,000–40,000 |",
        "| 类型 C 完整组数 | ≥ 4,680 |",
        "| `dev/mix` | ≥ 3,900 |",
        "| 未替换占位符 | 0 |",
        "| 跨语言标签不变性 | 0 处不一致 |",
        "",
        "## 字段",
        "",
        "`case_id`（跨语言一致）、`question_lang`、`source_lang`、`labels`；"
        "对抗样本另有 `adv_type`、`derived_from`、`pair_id`、`pair_role`、"
        "`adv_position`、`adv_injected`、`adv_similarity`、`adv_near_synonym`。",
        "",
        "## 配额（每垂类每语言记录数）",
        "",
        "预训练 ≥ 3,000、测试 ≥ 500、后训练 ≥ 300、开发 ≥ 100。",
        "",
        "许可证：MIT。数据全部由 LLM 生成，不含真实个人信息。",
        "",
    ]
    text = "\n".join(lines)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8", newline="\n")
        print(f"写入 {args.out}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
