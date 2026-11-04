"""装配后校验：逐文件行数/唯一率/语言、每垂类各 split 的交叉重叠、schema 合法性。

适配当前布局：
    data/train/00-llm.<category>.<lang>.jsonl            预训练
    data/train/posttrain/00-llm.<category>.<lang>.jsonl  后训练
    data/test|dev/00-llm.<category>.<lang>.jsonl         测试 / 开发
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from nano_jev.data import validate_record  # noqa: E402

ALLOWED_LANGS = ("zh-Hans", "zh-Hant", "en")
DATA = ROOT / "data"
GROUP_OF = {"train": "train", "test": "test", "dev": "dev"}


def digest_of(record: dict) -> bytes:
    payload = record["state"] + "\x00" + json.dumps(
        record["questions"], sort_keys=True, ensure_ascii=False
    )
    return hashlib.blake2b(payload.encode(), digest_size=8).digest()


def group_of(relative: str) -> str:
    parts = relative.split("/")
    if parts[0] == "train" and len(parts) > 2 and parts[1] == "posttrain":
        return "post"
    return GROUP_OF.get(parts[0], parts[0])


def main() -> int:
    files_report: dict[str, dict] = {}
    digests: dict[str, dict[str, set[bytes]]] = defaultdict(lambda: defaultdict(set))
    problems: list[str] = []
    for path in sorted(DATA.rglob("*.jsonl")):
        relative = path.relative_to(DATA).as_posix()
        group = group_of(relative)
        rows = 0
        seen: set[bytes] = set()
        langs: Counter = Counter()
        category = None
        with path.open(encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                rows += 1
                category = record.get("category", category)
                langs[record.get("lang")] += 1
                try:
                    validate_record(record, where=f"{relative}:{line_no}")
                except Exception as exc:  # noqa: BLE001
                    problems.append(f"{relative}:{line_no} {exc}")
                digest = digest_of(record)
                if digest in seen:
                    problems.append(f"{relative}: 重复内容（行 {line_no}）")
                seen.add(digest)
        if category is None:
            problems.append(f"{relative}: 空文件")
            continue
        digests[category][group].update(seen)
        files_report[relative] = {
            "group": group,
            "rows": rows,
            "unique": len(seen),
            "langs": dict(sorted(langs.items())),
        }
        print(f"{relative:64s} rows={rows:5d} unique={len(seen):5d} langs={','.join(sorted(langs))}")

    print("\n=== 每垂类（去重后）===")
    for category in sorted(digests):
        blocks = digests[category]
        train, post, test, dev = (blocks.get(k, set()) for k in ("train", "post", "test", "dev"))
        overlaps = {
            "train∩test": len(train & test),
            "train∩dev": len(train & dev),
            "train∩post": len(train & post),
            "post∩test": len(post & test),
            "post∩dev": len(post & dev),
        }
        total_overlap = sum(overlaps.values())
        print(
            f"{category:26s} train={len(train):5d} post={len(post):4d} test={len(test):4d} "
            f"dev={len(dev):3d} 交叉={total_overlap}"
        )
        if total_overlap:
            problems.append(f"{category}: split 交叉 {overlaps}")

    for relative, stats in files_report.items():
        if stats["group"] in ("train", "post") and len(stats["langs"]) != len(ALLOWED_LANGS):
            problems.append(f"{relative}: 语言数 {len(stats['langs'])} != 3")

    print(f"\n问题数：{len(problems)}")
    for item in problems[:20]:
        print("  -", item)
    out = ROOT / "artifacts" / "llm" / "verify_report.json"
    out.write_text(
        json.dumps({"files": files_report, "problems": problems}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return 0 if not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())
