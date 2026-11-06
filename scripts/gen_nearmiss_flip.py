"""近似反例（§3.2 修正版）：改一个数字让判定翻转，标签随之改变。

流程
----
1. 对每条 zh-Hans 记录，让 LLM 找出一个「改了就翻盘」的数字，并给出新的三题标签；
2. 校验：该数字在原 state 中**恰好出现一次**、新值原本不存在、新旧标签不同、
   档位/选项只相邻一档、新 state 仍满足语言与占位符约束；
3. 三语同步替换（数字与语言无关），`case_id` 加 `-nm` 后缀并保留 `derived_from`。

中间产物：``artifacts/llm/nearmiss2/flips.jsonl``（可断点续跑）。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import topup_pretrain as core  # noqa: E402

LANGS = ("zh-Hans", "zh-Hant", "en")
LOCK = threading.Lock()
NUM = re.compile(r"\d+(?:\.\d+)?")

SYSTEM = "你是数据标注专家，负责把决策题改造成「差一点就翻盘」的近似反例。只输出 JSON。"

PROMPT = """下面是 {count} 条决策案情及其标签。请为每条找出**一个关键数字**，把它改掉就能让判定翻盘，
并给出翻盘后的三题标签。

要求：
1. `from` 必须是案情原文中出现**且只出现一次**的数字（例如 9、12、0.85、85）；
2. `to` 是修改后的数字，必须与原值不同，且原案情里没有出现过这个数；
3. 修改必须正好跨过案情里写明的阈值/档位边界；
4. `labels` 是翻盘后的标签：noul 取 0/1，choice 取下面选项里的 key，score 取档位下标（从 0 开始）；
5. `evidence` 是案情里包含该数字的那一小段原文（照抄，≥6 字）。

{cases}

只输出 JSON 数组，长度 {count}，每项形如：
{{"index": 1, "from": "9", "to": "11", "labels": {{"noul产生的问题key": 0}}, "evidence": "..."}}"""

CASE = """--- 第 {index} 条 ---
案情：{state}
标签：{labels}
题目：{questions}"""


def parse_array(text: str) -> list[dict]:
    text = text.strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.M).strip()
    start, end = text.find("["), text.rfind("]")
    if start == -1 or end == -1:
        raise ValueError("no array")
    return json.loads(text[start : end + 1])


def question_summary(record: dict) -> str:
    parts = []
    for name, question in record["questions"].items():
        if question["type"] == "choice":
            options = "、".join(f"{k}={v}" for k, v in question["criteria"].items())
            parts.append(f"{name}(choice: {options})")
        elif question["type"] == "score":
            parts.append(f"{name}(score: {'、'.join(f'{i}={level}' for i, level in enumerate(question['criteria']))})")
        else:
            parts.append(f"{name}(noul: 0/1)")
    return "；".join(parts)


def validate(parsed: dict, item: dict) -> str | None:
    old, new = str(parsed.get("from", "")).strip(), str(parsed.get("to", "")).strip()
    if not NUM.fullmatch(old) or not NUM.fullmatch(new) or old == new:
        return "bad-number"
    if len(NUM.findall(item["state"])) < 2:
        return "too-few-numbers"
    if len(re.findall(rf"(?<!\d){re.escape(old)}(?!\d)", item["state"])) != 1:
        return "from-not-unique"
    if re.search(rf"(?<!\d){re.escape(new)}(?!\d)", item["state"]):
        return "to-already-present"
    labels = parsed.get("labels")
    if not isinstance(labels, dict) or set(labels) != set(item["orig_labels"]):
        return "bad-labels"
    if labels == item["orig_labels"]:
        return "labels-unchanged"
    changed = 0
    for name, question in item["questions"].items():
        value = labels[name]
        original = item["orig_labels"][name]
        if question["type"] == "noul":
            if value not in (0, 1):
                return f"bad-{name}"
            changed += int(value != original)
        elif question["type"] == "choice":
            keys = list(question["criteria"])
            if value not in keys:
                return f"bad-{name}"
            if value != original and abs(keys.index(value) - keys.index(original)) != 1:
                return "choice-not-adjacent"
            changed += int(value != original)
        else:
            levels = question["criteria"]
            if not isinstance(value, int) or not 0 <= value < len(levels):
                return f"bad-{name}"
            if value != original and abs(value - original) != 1:
                return "score-not-adjacent"
            changed += int(value != original)
    if changed == 0:
        return "nothing-changed"
    evidence = parsed.get("evidence")
    if not isinstance(evidence, str) or len(evidence.strip()) < 6 or evidence.strip() not in item["state"]:
        return "bad-evidence"
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", nargs="*", default=["train"])
    parser.add_argument("--categories", nargs="*", default=None)
    parser.add_argument("--per-category", type=int, default=500)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--work", default=str(ROOT / "artifacts" / "llm" / "nearmiss2"))
    args = parser.parse_args(argv)

    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    core.LOG_PATH = work / "nearmiss2.log"
    cache_path = work / "flips.jsonl"

    done: set[str] = set()
    if cache_path.exists():
        for line in cache_path.open(encoding="utf-8"):
            if line.strip():
                done.add(json.loads(line)["case_id"])

    targets: list[dict] = []
    for split in args.splits:
        for path in sorted((ROOT / split).glob("00-llm.*.zh-Hans.jsonl")):
            category = path.stem.split(".")[1]
            if args.categories and category not in args.categories:
                continue
            count = sum(1 for cid in done if cid.startswith(f"{category}-"))
            if args.per_category and count >= args.per_category:
                continue
            for line in path.open(encoding="utf-8"):
                if not line.strip():
                    continue
                record = json.loads(line)
                if record["case_id"] in done:
                    continue
                targets.append({
                    "split": split,
                    "category": category,
                    "case_id": record["case_id"],
                    "state": record["state"],
                    "labels": record["labels"],
                    "orig_labels": record["labels"],
                    "questions": record["questions"],
                    "summary": question_summary(record),
                })
                if args.per_category and count + sum(1 for t in targets if t["category"] == category) >= args.per_category:
                    break
    core.log(f"待生成翻盘样本 {len(targets)} 条（已有 {len(done)}）")
    if not targets:
        return 0

    stats: Counter = Counter()
    chunks = [targets[i : i + args.batch] for i in range(0, len(targets), args.batch)]

    def run(chunk: list[dict]) -> None:
        cases = "\n\n".join(
            CASE.format(index=index + 1, state=item["state"][:1400],
                        labels=json.dumps(item["labels"], ensure_ascii=False),
                        questions=item["summary"])
            for index, item in enumerate(chunk)
        )
        prompt = PROMPT.format(count=len(chunk), cases=cases)
        try:
            raw = core.call_api(
                [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
                temperature=0.3, max_tokens=4000,
            )
            results = parse_array(raw)
        except Exception:  # noqa: BLE001
            with LOCK:
                stats["request-failed"] += len(chunk)
            return
        by_index = {entry["index"]: entry for entry in results
                    if isinstance(entry, dict) and isinstance(entry.get("index"), int)}
        rows = []
        for index, item in enumerate(chunk, start=1):
            parsed = by_index.get(index)
            reason = "missing" if parsed is None else validate(parsed, item)
            if reason:
                with LOCK:
                    stats[reason] += 1
                continue
            rows.append({
                "case_id": item["case_id"],
                "category": item["category"],
                "split": item["split"],
                "from": str(parsed["from"]).strip(),
                "to": str(parsed["to"]).strip(),
                "labels": parsed["labels"],
                "evidence": parsed["evidence"].strip(),
            })
        if rows:
            with LOCK:
                with cache_path.open("a", encoding="utf-8", newline="\n") as handle:
                    for row in rows:
                        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                stats["ok"] += len(rows)
                if stats["ok"] % 200 < len(rows):
                    core.log(f"  已生成 {stats['ok']}")

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(run, chunks))
    core.log(f"统计 {dict(stats)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
