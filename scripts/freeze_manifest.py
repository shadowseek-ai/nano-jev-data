"""重新生成 MANIFEST.json。

只扫描数据目录 ``train/``、``test/``、``dev/``，不会把 ``artifacts/`` 下的种子文件
误当成数据分片。字段与 nano-jev 的 data freeze 保持一致，``nano-jev data verify``
可以直接校验。
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter, defaultdict
from datetime import date, datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from topup_pretrain import load_specs  # noqa: E402

TARGET_TRAIN_RECORDS = 3000
NOTE = (
    "三语对齐数据集：简体中文为源语言，每垂类每语言 train>=3000 条 / test>=500 题 / "
    "post>=200 题，已去重；MIT 许可"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def split_of(relative: str) -> str:
    parts = relative.split("/")
    return parts[0]


def next_version(previous: str | None) -> str:
    today = date.today().isoformat().replace("-", ".")
    if not previous:
        return f"{today}.1"
    parts = previous.rsplit(".", 1)
    if len(parts) == 2 and parts[0] == today and parts[1].isdigit():
        return f"{today}.{int(parts[1]) + 1}"
    return f"{today}.1"


def main() -> int:
    dry_run = "--dry-run" in sys.argv[1:]
    manifest_path = ROOT / "MANIFEST.json"
    previous = None
    if manifest_path.exists():
        try:
            previous = json.loads(manifest_path.read_text(encoding="utf-8")).get("dataset_version")
        except json.JSONDecodeError:
            previous = None

    files = []
    for path in sorted(ROOT.glob("train/**/*.jsonl")) + sorted(ROOT.glob("test/*.jsonl")) + sorted(ROOT.glob("dev/*.jsonl")):
        if not path.is_file():
            continue
        relative = path.relative_to(ROOT).as_posix()
        split = split_of(relative)
        records = 0
        questions = 0
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                records += 1
                questions += len(record.get("questions", {}))
        files.append({
            "path": relative,
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
            "records": records,
            "questions": questions,
            "split": split,
            "origin": "custom",
        })

    splits: dict[str, dict[str, int]] = {}
    per_category: dict[str, dict[str, dict[str, int]]] = defaultdict(
        lambda: defaultdict(lambda: {"records": 0, "questions": 0})
    )
    for entry in files:
        split = entry["split"]
        category = Path(entry["path"]).name.split(".")[1]
        splits.setdefault(split, {"records": 0, "questions": 0})
        splits[split]["records"] += entry["records"]
        splits[split]["questions"] += entry["questions"]
        per_category[category][split]["records"] += entry["records"]
        per_category[category][split]["questions"] += entry["questions"]

    manifest = {
        "schema_version": 1,
        "dataset_version": next_version(previous),
        "generator": "nano-jev-synth/1",
        "seed": None,
        "frozen_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "note": NOTE,
        "license": "MIT",
        "root": "data",
        "splits": splits,
        "files": {entry["path"]: {key: value for key, value in entry.items() if key != "path"} for entry in files},
        "per_category": {category: dict(stats) for category, stats in sorted(per_category.items())},
        "quota": {
            "train_records_per_category_language": TARGET_TRAIN_RECORDS,
            "train_questions_per_category": TARGET_TRAIN_RECORDS * 3,
            "test_questions_per_category": 500,
            "post_questions_per_category": 200,
        },
    }
    if not dry_run:
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")

    # 打印核验摘要
    specs = load_specs()
    counts: Counter = Counter()
    for path in sorted((ROOT / "train").glob("*.jsonl")):
        category, lang = path.stem.split(".")[1], path.stem.split(".")[2]
        for line in path.open(encoding="utf-8"):
            if line.strip():
                counts[(category, lang)] += 1
    bad = []
    for category in sorted(specs):
        for lang in ("zh-Hans", "zh-Hant", "en"):
            if counts[(category, lang)] < TARGET_TRAIN_RECORDS:
                bad.append(f"{category}/{lang}={counts[(category, lang)]}")
    print(f"version={manifest['dataset_version']} files={len(files)} records={sum(e['records'] for e in files)}")
    print(f"train={splits.get('train', {}).get('records')} test={splits.get('test', {}).get('records')} dev={splits.get('dev', {}).get('records')}")
    print(f"quota violations={len(bad)}")
    for item in bad:
        print("  -", item)
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())
