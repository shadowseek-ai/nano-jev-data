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

    files = list(ROOT.glob("train/**/*.jsonl")) + list(ROOT.glob("test/*.jsonl")) + list(ROOT.glob("dev/*.jsonl"))
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
        f"| 规模 | **{total_records:,} 条记录 / {total_questions:,} 题**，13 个垂类，{len(files)} 个文件，约 {size_mb:.0f} MB |",
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
        "语言分布（记录）：" + "、".join(f"`{lang}` {lang_records[lang]:,}" for lang in ("zh-Hans", "zh-Hant", "en")) + "。",
        text,
    )
    README.write_text(text, encoding="utf-8", newline="\n")
    print(f"README 更新：{manifest['dataset_version']} records={total_records} questions={total_questions} files={len(files)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
