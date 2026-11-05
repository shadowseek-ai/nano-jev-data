"""近似反例（§3.2）：给 choice 题加一个「近义但错误」的干扰项。

做法
----
每批 4 条记录，让 LLM 为每条产出三语干扰项文本 + 一个 ``contradicts`` 片段；
``contradicts`` 必须是 state 里的**原文子串**，用来证明该干扰项与案情冲突。
校验通过才落盘，因此标签不变、干扰项可追溯到具体证据。

中间产物：``artifacts/llm/nearmiss/distractors.jsonl``（可断点续跑）。
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

SYSTEM = (
    "你是数据标注专家，擅长为决策类选择题构造「近义但错误」的干扰项。"
    "只输出 JSON，不要解释。"
)

PROMPT = """下面有 {count} 条案情，每条都要构造 1 个近似反例干扰项。

构造要求：
1. 干扰项 = 「正确选项的意思」+「一个与案情冲突的限定语/参数」，读起来像正常选项；
   例如正确选项是「附条件批准」，干扰项可以是「附条件批准，材料可放款后补」；
2. **绝对不能**照抄或同义改写任何一个已有选项——必须有超出已有选项的额外信息；
3. 案情里有明确冲突的证据，把那段**原文照抄**到 contradicts 字段（≥4 个字，必须是案情原文子串）；
4. 干扰项不超过 24 字，不要自称“错误”；
5. 给出 zh-Hans / zh-Hant / en 三种语言的等价表述。

{cases}

只输出 JSON 数组，长度 {count}，每项形如：
{{"index": 1, "zh-Hans": "...", "zh-Hant": "...", "en": "...", "contradicts": "..."}}"""

CASE = """--- 第 {index} 条 ---
案情：{state}
正确选项 `{gold}` = 「{gold_text}」
其他选项：{options}"""


def build_choice(record: dict) -> tuple[str, dict] | None:
    for key, question in record["questions"].items():
        if question["type"] == "choice":
            return key, question
    return None


def parse_array(text: str) -> list[dict]:
    text = text.strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.M).strip()
    start, end = text.find("["), text.rfind("]")
    if start == -1 or end == -1:
        raise ValueError("no array")
    return json.loads(text[start : end + 1])


def validate(parsed: dict, item: dict) -> str | None:
    limits = {"zh-Hans": 40, "zh-Hant": 40, "en": 90}
    for lang in LANGS:
        text = parsed.get(lang)
        if not isinstance(text, str) or not 2 <= len(text.strip()) <= limits[lang]:
            return f"bad-{lang}"
        if text.strip() in item["option_texts"]:
            return "duplicate-option"
    contradicts = parsed.get("contradicts")
    if not isinstance(contradicts, str) or len(contradicts.strip()) < 4:
        return "no-evidence"
    if contradicts.strip() not in item["state"]:
        return "evidence-not-in-state"
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", nargs="*", default=["train"])
    parser.add_argument("--categories", nargs="*", default=None)
    parser.add_argument("--per-category", type=int, default=0)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--work", default=str(ROOT / "artifacts" / "llm" / "nearmiss"))
    args = parser.parse_args(argv)

    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    core.LOG_PATH = work / "nearmiss.log"
    cache_path = work / "distractors.jsonl"

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
            count = 0
            for line in path.open(encoding="utf-8"):
                if not line.strip():
                    continue
                record = json.loads(line)
                if args.per_category and count >= args.per_category:
                    break
                if record.get("case_id") in done:
                    continue
                built = build_choice(record)
                if built is None:
                    continue
                key, question = built
                gold = record["labels"].get(key)
                if gold not in question["criteria"]:
                    continue
                targets.append({
                    "split": split,
                    "category": category,
                    "case_id": record["case_id"],
                    "state": record["state"],
                    "key": key,
                    "gold": gold,
                    "gold_text": question["criteria"][gold],
                    "option_texts": list(question["criteria"].values()),
                    "options": {k: v for k, v in question["criteria"].items() if k != gold},
                })
                count += 1
    core.log(f"待生成干扰项 {len(targets)} 条（已有 {len(done)}）")
    if not targets:
        return 0

    stats: Counter = Counter()
    chunks = [targets[i : i + args.batch] for i in range(0, len(targets), args.batch)]

    def run(chunk: list[dict]) -> None:
        cases = "\n\n".join(
            CASE.format(index=index + 1, state=item["state"][:1200], gold=item["gold"],
                        gold_text=item["gold_text"],
                        options=json.dumps(item["options"], ensure_ascii=False))
            for index, item in enumerate(chunk)
        )
        prompt = PROMPT.format(count=len(chunk), cases=cases)
        try:
            raw = core.call_api(
                [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
                temperature=0.4, max_tokens=4000,
            )
            results = parse_array(raw)
        except Exception:  # noqa: BLE001
            with LOCK:
                stats["request-failed"] += len(chunk)
            return
        by_index = {}
        for entry in results:
            if isinstance(entry, dict) and isinstance(entry.get("index"), int):
                by_index[entry["index"]] = entry
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
                "key": item["key"],
                "gold": item["gold"],
                "texts": {lang: parsed[lang].strip() for lang in LANGS},
                "contradicts": parsed["contradicts"].strip(),
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
