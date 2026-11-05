"""把交付数据里残留的模板占位符渲染成真实值（按 case_id 确定性映射，三语一致）。"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import case_meta  # noqa: E402

SPLITS = ("train", "test", "dev")


def main() -> int:
    stats: Counter = Counter()
    for split in SPLITS:
        for path in sorted((ROOT / split).glob("00-llm.*.jsonl")) + sorted((ROOT / split / "posttrain").glob("00-llm.*.jsonl")):
            lines = []
            changed = 0
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                record = json.loads(line)
                tokens = case_meta.residual_placeholders(record["state"])
                if tokens:
                    record["state"], _ = case_meta.render_state(
                        record["state"], record.get("case_id") or record["id"], record.get("lang", "zh-Hans"))
                    changed += 1
                lines.append(json.dumps(record, ensure_ascii=False))
            if changed:
                path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
                stats[split] += changed
    print("rendered", dict(stats), "total", sum(stats.values()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
