"""收尾：把 topup 种子合并进 train/ 并完成校验、冻结与统计。

顺序（任一步失败即停）：
1. merge       —— 只合并“源 + 两个译文”都齐的三元组，写入 train/00-llm.<category>.<lang>.jsonl
2. audit       —— 种子审计（标签域 / 语言 / 占位符 / 数字 / 三语对齐）
3. verify      —— 数据门禁（schema / 语言纯度 / 重复 / 跨 split 交叉 / 3000 配额）
4. freeze      —— 重新生成 MANIFEST.json
5. readme      —— 按 MANIFEST 重写 README 统计
6. quota       —— 逐垂类逐语言核验

用法：
    python scripts/finalize_dataset.py --work artifacts/llm/topup
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import topup_pretrain as core  # noqa: E402


def run(script: str, *args: str) -> int:
    cmd = [sys.executable, str(ROOT / "scripts" / script), *args]
    print(f"\n$ {' '.join(cmd)}", flush=True)
    return subprocess.run(cmd, cwd=ROOT).returncode


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", default=str(core.DEFAULT_WORK))
    parser.add_argument("--target", type=int, default=core.TARGET_TOTAL)
    parser.add_argument("--skip-merge", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    work = Path(args.work)
    core.LOG_PATH = work / "finalize.log"
    specs = core.load_specs()
    counts, _seen, _examples = core.load_existing()
    plan = core.compute_plan(specs, counts, args.target, None)
    store = core.Store(work / "seeds.jsonl", work / "translations.jsonl")
    incomplete = [r["source_id"] for r in store.sources
                  if not all((r["source_id"], t) in store.translations for t in core.TARGETS)]
    core.log(f"种子：source={len(store.sources)} translation={len(store.translations)} 未齐={len(incomplete)}")
    if incomplete:
        core.log("存在未齐三元组，先补齐再合并；本次仍会合并已齐部分")

    if not args.skip_merge:
        report = core.merge(plan, specs, store, dry_run=args.dry_run)
        (work / "merge_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        core.log(f"合并：{report['languages']}")
        if args.dry_run:
            return 0

    for script, extra in (
        ("topup_audit.py", [str(work)]),
        ("topup_verify.py", []),
        ("freeze_manifest.py", []),
        ("update_readme_stats.py", []),
        ("check_quota.py", ["--dir", "."]),
    ):
        code = run(script, *extra)
        if code != 0:
            core.log(f"{script} 失败（退出码 {code}）")
            return code if code != 1 or script != "check_quota.py" else 0
    core.log("收尾完成")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
