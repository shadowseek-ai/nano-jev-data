"""把单语言（默认 zh-Hans）的语料翻译成其它语言，保证跨语言内容严格对齐。

为什么翻译而不是各自生成
------------------------
分别生成时，同一 (task, lang) 下的场景、数字、结论互不相关，三语言"同义样本"
其实并不同义；训练时等于把三份不同分布混在一起。翻译能保证：同一条记录在三种
语言里描述的是同一个业务事实，标签天然一致。

用法
----
    $env:LLM_API_KEY="..."
    py artifacts/llm/translate_seeds.py --targets zh-Hant,en --workers 4 --batch 16

产物：artifacts/llm/seeds_api/<category>/<task>.<lang>.jsonl
进度：按已有译文行数续跑（与源文件顺序对齐）
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from opencc import OpenCC

ROOT = Path(__file__).resolve().parents[2]
SEED_DIRS = (ROOT / "artifacts" / "llm" / "seeds", ROOT / "artifacts" / "llm" / "seeds_api")
OUT = ROOT / "artifacts" / "llm" / "seeds_api"
LOG = ROOT / "artifacts" / "llm" / "translate.log"

MODEL = os.environ.get("LLM_MODEL", "")  # 必须由环境变量提供，不写死模型名
API_BASE = os.environ.get("LLM_API_BASE", "").rstrip("/")  # 必须由环境变量提供，不写死网关
API_KEYS = [k.strip() for k in os.environ.get("LLM_API_KEYS", os.environ.get("LLM_API_KEY", "")).split(",") if k.strip()]
MAX_TOKENS = int(os.environ.get("LLM_MAX_TOKENS", "128000"))
SOURCE_LANG = os.environ.get("LLM_SOURCE_LANG", "zh-Hans")

T2S, S2T = OpenCC("t2s"), OpenCC("s2t")
CJK = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
PLACEHOLDER = re.compile(r"\{[a-z_]+\}")
NUMBER = re.compile(r"\d+(?:[.,]\d+)*")
LANG_NAME = {"zh-Hans": "简体中文", "zh-Hant": "繁体中文（台湾用语）", "en": "英文"}

_LOCK = threading.Lock()
_STOP = threading.Event()
_STATS = Counter()
_CLIENTS: dict[str, object] = {}
_CURSOR = 0


def log(message: str) -> None:
    line = f"{time.strftime('%H:%M:%S')} {message}"
    with LOG.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
    print(message.encode("ascii", "replace").decode("ascii"), flush=True)


def client():
    global _CURSOR
    from openai import OpenAI

    with _LOCK:
        if not API_BASE or not MODEL:
            raise RuntimeError("LLM_API_BASE / LLM_MODEL 未设置：必须由环境变量提供")
        if not API_KEYS:
            raise RuntimeError("LLM_API_KEY 未设置")
        key = API_KEYS[_CURSOR % len(API_KEYS)]
        _CURSOR += 1
    if key not in _CLIENTS:
        _CLIENTS[key] = OpenAI(api_key=key, base_url=API_BASE, max_retries=2)
    return _CLIENTS[key]


def fingerprint(text: str) -> str:
    return hashlib.blake2b(re.sub(r"\s+", "", text).lower().encode(), digest_size=8).hexdigest()


def load_sources() -> dict[tuple[str, str], list[dict]]:
    """收集源语言记录（seeds 与 seeds_api 合并，按 task 分组）。"""
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    seen: dict[tuple[str, str], set[str]] = defaultdict(set)
    for folder in SEED_DIRS:
        for path in sorted(folder.rglob("*.jsonl")):
            category = path.parent.name
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                record = json.loads(line)
                if record.get("lang") != SOURCE_LANG:
                    continue
                task = record.get("task")
                digest = fingerprint(record.get("state", ""))
                if digest in seen[(category, task)]:
                    continue
                seen[(category, task)].add(digest)
                grouped[(category, task)].append(record)
    return grouped


def validate(translated: str, source: str, lang: str) -> tuple[bool, str]:
    text = translated.strip()
    if not 40 <= len(text) <= 6000:
        return False, "length"
    if set(PLACEHOLDER.findall(text)) != set(PLACEHOLDER.findall(source)):
        return False, "placeholder"
    remaining = list(NUMBER.findall(text))
    for token in NUMBER.findall(source):
        if token in remaining:
            remaining.remove(token)
            continue
        units = re.search(re.escape(token) + r"\s*(万|亿)", source)
        alt = []
        if units:
            base = int(float(token.replace(",", "")))
            scaled = base * (10000 if units.group(1) == "万" else 100000000)
            alt = [str(scaled), f"{scaled:,}"]
        hit = next((a for a in alt if a in remaining), None)
        if hit is not None:
            remaining.remove(hit)
            continue
        return False, "number"
    if lang == "en":
        if CJK.search(text):
            return False, "cjk-in-en"
        if len(text.split()) < 12:
            return False, "too-short-en"
    elif lang == "zh-Hant":
        if T2S.convert(text) == text:
            return False, "not-traditional"
    elif lang == "zh-Hans":
        if T2S.convert(text) != text:
            return False, "not-simplified"
    return True, "ok"


def translate_batch(items: list[dict], target: str) -> list[str]:
    payload = [{"i": index, "text": item["state"]} for index, item in enumerate(items)]
    required = " ".join(sorted({*PLACEHOLDER.findall(" ".join(i["state"] for i in items)), *NUMBER.findall(" ".join(i["state"] for i in items))}))
    prompt = f"""把下面 {len(items)} 条业务语料逐条翻译成{LANG_NAME[target]}。

要求：
1. 逐条一一对应，条数不变，顺序不变。
2. 保留所有占位符原样（如 {{ticket}}、{{date}}）与所有数字、单位、编号，不得改写或省略。
3. 只翻译语言，不改变事实、结论与措辞强度；不要添加解释。
4. 下文列出的数字与占位符必须在译文中原样、按原数量出现（可多不可少）：
   {required}
5. 输出一个 JSON 对象：{{"records": [{{"i": 0, "text": "<译文>"}}, ...]}}

输入：
{json.dumps(payload, ensure_ascii=False)}"""
    response = client().chat.completions.create(
        model=MODEL,
        messages=[{"role": "system", "content": "你是专业本地化译者，只输出 JSON。"},
                  {"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=MAX_TOKENS,
        response_format={"type": "json_object"},
        timeout=int(os.environ.get("LLM_TIMEOUT", "300")),
    )
    content = response.choices[0].message.content or ""
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        return ["" for _ in items]
    rows = data.get("records") or data.get("data") or []
    out = ["" for _ in items]
    for row in rows:
        if isinstance(row, dict) and isinstance(row.get("i"), int) and 0 <= row["i"] < len(items):
            out[row["i"]] = str(row.get("text", ""))
    return out


def run_group(category: str, task: str, sources: list[dict], targets: list[str], batch: int) -> int:
    """翻译一组。游标与跳过记录持久化到 sidecar，保证断点续跑与覆盖可恢复。"""
    written = 0
    for target in targets:
        path = OUT / category / f"{task}.{target}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        side = OUT / category / f"{task}.{target}.cursor.json"
        try:
            state = json.loads(side.read_text(encoding="utf-8")) if side.exists() else {"next": 0, "skipped": []}
        except json.JSONDecodeError:
            state = {"next": 0, "skipped": []}
        cursor = int(state.get("next", 0))
        if target == SOURCE_LANG:
            with _LOCK:
                with path.open("a", encoding="utf-8", newline="\n") as handle:
                    for item in sources[cursor:]:
                        handle.write(json.dumps({"task": task, "lang": target, "state": item["state"].strip(),
                                                 "labels": item["labels"]}, ensure_ascii=False) + "\n")
            state["next"] = len(sources)
            side.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
            written += len(sources) - cursor
            continue
        failures = 0
        size = batch
        while cursor < len(sources) and not _STOP.is_set():
            chunk = sources[cursor : cursor + size]
            try:
                texts = translate_batch(chunk, target)
            except Exception as exc:  # noqa: BLE001
                name = type(exc).__name__
                if "Authentication" in name or "401" in str(exc) or "-10007" in str(exc):
                    _STOP.set()
                    log(f"鉴权/额度失败：{name}")
                    return written
                with _LOCK:
                    _STATS[f"error:{name}"] += 1
                failures += 1
                if failures >= 8:
                    with _LOCK:
                        _STATS["abandoned_group"] += 1
                        log(f"暂停该组（请求连续失败）{category}/{task}/{target} cursor={cursor}")
                    return written
                time.sleep(3)
                continue
            rows = []
            for item, text in zip(chunk, texts):
                ok, reason = validate(text, item["state"], target)
                if not ok:
                    with _LOCK:
                        _STATS[f"reject:{reason}"] += 1
                    break
                rows.append({"task": task, "lang": target, "state": text.strip(), "labels": item["labels"]})
            if len(rows) == len(chunk):
                with _LOCK:
                    with path.open("a", encoding="utf-8", newline="\n") as handle:
                        for row in rows:
                            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                written += len(rows)
                cursor += len(chunk)
                state["next"] = cursor
                side.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
                failures = 0
                size = batch
                continue
            failures += 1
            if size > 1:
                size = max(1, size // 2)
                continue
            state.setdefault("skipped", []).append(cursor)
            cursor += 1
            state["next"] = cursor
            side.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
            failures = 0
            if len(state["skipped"]) >= 40:
                with _LOCK:
                    _STATS["abandoned_group"] += 1
                    log(f"暂停该组（跳过过多）{category}/{task}/{target} skipped={len(state['skipped'])}")
                return written
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", default="zh-Hans,zh-Hant,en")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--limit", type=int, default=0, help="只翻译前 N 组，用于试跑")
    parser.add_argument("--categories", nargs="*", default=None, help="只处理指定垂类")
    args = parser.parse_args(argv)

    targets = [t.strip() for t in args.targets.split(",") if t.strip()]
    sources = load_sources()
    groups = sorted(sources.items(), key=lambda kv: -len(kv[1]))
    if args.limit:
        groups = groups[: args.limit]
    log(f"groups={len(groups)} sources={sum(len(v) for _, v in groups)} targets={targets} workers={args.workers}")

    total = 0
    started = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(run_group, category, task, records, targets, args.batch): (category, task)
            for (category, task), records in groups
        }
        for future in as_completed(futures):
            if _STOP.is_set():
                break
            try:
                total += future.result()
            except Exception as exc:  # noqa: BLE001
                with _LOCK:
                    _STATS[f"fatal:{type(exc).__name__}"] += 1
            elapsed = max(time.time() - started, 1)
            log(f"已译 {total} 条，{total / elapsed * 3600:.0f} 条/小时，stats={dict(_STATS)}")
    log(f"结束：写入 {total} 条，stats={dict(_STATS)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
