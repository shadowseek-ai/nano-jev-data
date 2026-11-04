"""调用 LLM API 批量生成语料种子（只产出 state + labels，问题定义仍取自 synth 模块）。

用法
----
    $env:LLM_API_KEY = "..."          # 只放环境变量，不落盘
    py artifacts/llm/api_gen.py --plan            # 只打印计划
    py artifacts/llm/api_gen.py --probe 1         # 单批试跑，看通过率
    py artifacts/llm/api_gen.py --workers 12      # 正式跑

产物：artifacts/llm/seeds_api/<category>/<task>.<lang>.jsonl
进度：artifacts/llm/api_state.json（可断点续跑）
日志：artifacts/llm/api_gen.log（UTF-8）

配额口径（每垂类每语言）：train 5000 / test 500 / post 200 / dev 50，按 3 个任务均分。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import threading
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SPECS = ROOT / "artifacts" / "llm" / "specs"
OUT = ROOT / "artifacts" / "llm" / "seeds_api"
LEGACY_SEEDS = ROOT / "artifacts" / "llm" / "seeds"
STATE_FILE = ROOT / "artifacts" / "llm" / "api_state.json"
LOG_FILE = ROOT / "artifacts" / "llm" / "api_gen.log"

API_BASE = os.environ.get("LLM_API_BASE", "").rstrip("/")  # 必须由环境变量提供，不写死网关
API_KEYS = [k.strip() for k in os.environ.get("LLM_API_KEYS", os.environ.get("LLM_API_KEY", "")).split(",") if k.strip()]
API_KEY = API_KEYS[0] if API_KEYS else ""
_KEY_CURSOR = 0
_CLIENTS: dict[str, object] = {}
MODEL = os.environ.get("LLM_MODEL", "")  # 必须由环境变量提供，不写死模型名
MAX_TOKENS = int(os.environ.get("LLM_MAX_TOKENS", "128000"))

LANGS = ("zh-Hans", "zh-Hant", "en")
GEN_LANGS = tuple(x.strip() for x in os.environ.get("LLM_GEN_LANGS", "zh-Hans").split(",") if x.strip())  # 只生成源语言，其余靠翻译对齐
TARGET_PER_LANG = {"train": 20000, "test": 2000, "post": 1000, "dev": 100}  # 每语言总量（跨全部垂类）
TASKS_PER_CATEGORY = 3
OVERSUPPLY = 1.15
PER_TASK_OVERRIDE = int(os.environ.get("LLM_PER_TASK", "0"))  # 超发 15%，抵消去重与拒收损耗，保证去重后仍 >= 目标
BATCH = 32

_LOCK = threading.Lock()
_SEEN: dict[tuple[str, str], set[str]] = defaultdict(set)
_WRITTEN: dict[tuple[str, str, str], int] = defaultdict(int)
_STATS = Counter()
STOP_EVENT = threading.Event()  # 额度/鉴权失败时立即止损


def log(message: str) -> None:
    line = f"{time.strftime('%H:%M:%S')} {message}"
    with LOG_FILE.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
    print(message.encode("ascii", "replace").decode("ascii"), flush=True)


def normalize(state: str) -> str:
    return re.sub(r"\s+", "", state).lower()


def fingerprint(state: str, questions_hint: str = "") -> str:
    return hashlib.blake2b((normalize(state) + questions_hint).encode(), digest_size=8).hexdigest()


def load_spec(category: str) -> dict:
    return json.loads((SPECS / f"{category}.json").read_text(encoding="utf-8"))


def categories() -> list[str]:
    return sorted(path.stem for path in SPECS.glob("*.json") if not path.stem.startswith("_"))


def domain_of(question: dict) -> set:
    qtype = question["type"]
    if qtype == "noul":
        return {0, 1}
    criteria = question["criteria"]["en"]
    if qtype == "choice":
        return set(criteria.keys())
    return set(range(len(criteria)))


def question_block(question: dict, lang: str, name: str) -> str:
    qtype = question["type"]
    instructions = question["instructions"].get(lang) or question["instructions"]["en"]
    criteria = question["criteria"].get(lang) or question["criteria"]["en"]
    if qtype == "noul":
        options = f"0 = {criteria.get('false')} / 1 = {criteria.get('true')}"
    elif qtype == "choice":
        options = " | ".join(f"{key} = {label}" for key, label in criteria.items())
    else:
        options = " | ".join(f"{index} = {label}" for index, label in enumerate(criteria))
    return f"- {name}（{qtype}）问题：{instructions[0]}\n  取值：{options}"


SYSTEM = (
    "你是业务数据撰写员。你的输出会被用作决策模型的训练语料：每条记录是一段真实业务输入，"
    "外加三个可由输入唯一确定的标签。只输出 JSONL，不要解释、不要 Markdown 代码块。"
)

STYLES = [
    "工单正文", "内部邮件", "即时通讯群聊记录", "系统日志片段", "值班电话记录转述",
    "审批备注", "表格粘贴片段", "JSON 导出片段", "交接班便签", "客户投诉原文",
    "监控告警窗口", "会议纪要摘录", "短信通报", "复盘报告摘要",
]


def build_prompt(spec: dict, task: dict, lang: str, count: int, theme: dict) -> list[dict]:
    task_id = task["id"]
    lines = [question_block(task["questions"][name], lang, name) for name in task["questions"]]
    labels_shape = ", ".join(f'"{name}": <取值>' for name in task["questions"])
    lang_rule = {
        "zh-Hans": "简体中文",
        "zh-Hant": "繁体中文（台湾用语，不得出现简体字）",
        "en": "英文（不得出现任何中文字符）",
    }[lang]
    user = f"""业务场景：{spec['category']} / {task_id}（{task['title']}）

三个待判定问题：
{chr(10).join(lines)}

请写 {count} 条该场景下的真实业务输入，语言：{lang_rule}。

只输出一个 JSON 对象，格式：
{{"records": [{{"state": "<输入原文>", "labels": {{{labels_shape}}}}}, ...]}}
records 必须有 {count} 个元素。

硬性要求：
1. 长度不设限：真实用户输入长短不一，短的可以只有一两句话，长的可以是完整工单描述；
   但必须在 40-4000 字符之间，且不得为了凑长度堆砌空话；
2. 必须包含 1-3 句干扰信息（例如顺带提到别的工单、别的设备、无关的时间或人物），
   干扰项只增加噪声，不得改变标签判定所依据的关键信号；必须是"人写的业务输入"，形态在本批内尽量分散，例如：{", ".join(theme["styles"])}。
2. labels 的每个值必须能由 state 里的显式事实唯一推出，不允许猜测。若某个分档是加权累计分，请在 state 里写出分项与累计分。
3. 与标签相关的数字（金额、天数、比例、温度、分数、人数、用量等）必须写死成具体值，不要用占位符。
4. 非关键实体用占位符：{{case_id}}、{{date}}、{{time}}、{{ticket}}、{{customer}}、{{order_no}}、{{site}}、{{device}}、{{batch}}。
5. 不要出现任何英文字段名（如 {", ".join(task["questions"])} 之类的键名）、不要写"答案是"。
6. 不要出现真实人名、真实公司名、真实机构名；患者/客户用 {theme['persona']} 这类泛称。
7. {count} 条之间场景、数字、句式都要不同，不要套模板。
8. 场景主题参考：{theme['topic']}；涉事方参考：{theme['persona']}。"""
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def call_api(messages: list[dict], max_tokens: int | None = None) -> str:
    """调用 OpenAI 兼容接口，强制 JSON object 输出。"""
    response = _client().chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=1.0,
        max_tokens=max_tokens or MAX_TOKENS,
        response_format={"type": "json_object"},
        timeout=300,
    )
    return response.choices[0].message.content or ""


_CLIENT = None
_CLIENT_LOCK = threading.Lock()


def _client():
    global _CLIENT
    with _CLIENT_LOCK:
        if _CLIENT is None:
            from openai import OpenAI

            _CLIENT = OpenAI(api_key=API_KEY, base_url=API_BASE, max_retries=2)
        return _CLIENT


def parse_records(text: str) -> list[dict]:
    """优先按 {"records": [...]} 解析，兼容 JSONL 与裸数组。"""
    text = re.sub(r"^```(?:json|jsonl)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    payloads: list = []
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, dict):
        for key in ("records", "data", "items", "results"):
            if isinstance(parsed.get(key), list):
                payloads = parsed[key]
                break
        else:
            payloads = [parsed]
    elif isinstance(parsed, list):
        payloads = parsed
    else:
        for line in text.splitlines():
            line = line.strip().rstrip(",")
            if line.startswith("{") and line.endswith("}"):
                try:
                    payloads.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return [
        item
        for item in payloads
        if isinstance(item, dict) and "state" in item and "labels" in item
    ]


def validate(payload: dict, task: dict, lang: str) -> tuple[bool, str]:
    state = payload.get("state")
    labels = payload.get("labels")
    if not isinstance(state, str) or not isinstance(labels, dict):
        return False, "shape"
    state = state.strip()
    if not 40 <= len(state) <= 4000:
        return False, "length"
    if set(labels) != set(task["questions"]):
        return False, "keys"
    for name, question in task["questions"].items():
        value = labels[name]
        if question["type"] == "noul":
            if value not in (0, 1) or isinstance(value, bool):
                return False, "noul"
        elif question["type"] == "choice":
            if value not in domain_of(question):
                return False, "choice"
        else:
            levels = question["criteria"]["en"]
            if not isinstance(value, int) or isinstance(value, bool) or not 0 <= value < len(levels):
                return False, "score"
    if lang == "en" and re.search(r"[\u3400-\u4dbf\u4e00-\u9fff]", state):
        return False, "cjk-in-en"
    return True, "ok"


def themes(task_id: str, category: str, size: int) -> list[dict]:
    topics = [
        "月末结算高峰、系统并发升高", "节前订单激增、人力紧张", "新规上线首周、口径变更",
        "设备老化与备件短缺同时发生", "跨区域协同、时区与语言不一致", "审计抽查期间的历史遗留问题",
        "供应商临时涨价、合同条款争议", "旺季前的容量压测与扩容评审", "客户投诉升级到监管渠道",
        "系统迁移窗口期、回滚预案待确认", "第三方数据延迟导致的连锁异常", "极端天气下的应急响应",
    ]
    personas = ["某连锁门店运营主管", "某区域服务中心值班员", "某制造企业设备工程师", "某平台客服专员",
                "某医院行政值班人员", "某实验室技术员", "某物流调度员", "某电网调度员",
                "某银行风控专员", "某招聘 HRBP"]
    styles = random.sample(STYLES, k=6)
    out = []
    for index in range(size):
        out.append({
            "topic": topics[(hash(task_id) + index) % len(topics)],
            "persona": personas[(hash(category) + index) % len(personas)],
            "styles": random.sample(STYLES, k=6),
        })
    return out


def output_path(category: str, task_id: str, lang: str) -> Path:
    path = OUT / category / f"{task_id}.{lang}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def load_existing() -> None:
    for folder in (LEGACY_SEEDS, OUT):
        for path in sorted(folder.rglob("*.jsonl")):
            category = path.parent.name if path.parent != OUT else path.parent.name
            try:
                for line in path.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    lang = record.get("lang")
                    state = record.get("state", "")
                    if lang and state:
                        with _LOCK:
                            _SEEN[(category, lang)].add(fingerprint(state))
            except (json.JSONDecodeError, OSError):
                continue


def load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    return {}


def save_state(state: dict) -> None:
    with _LOCK:
        STATE_FILE.write_text(json.dumps(state, ensure_ascii=False, indent=1), encoding="utf-8")


def run_batch(category: str, spec: dict, task: dict, lang: str, count: int, attempt_seed: int) -> tuple[int, int]:
    """生成一批；返回 (写入条数, attempt 数)。"""
    accepted = 0
    tries = 0
    theme_list = themes(task["id"], category, 3)
    while accepted < count and tries < 4:
        tries += 1
        if STOP_EVENT.is_set():
            return accepted, tries
        want = count - accepted
        messages = build_prompt(spec, task, lang, want, theme_list[tries % len(theme_list)])
        try:
            text = call_api(messages)
        except Exception as exc:  # noqa: BLE001 - 区分鉴权/额度与可重试错误
            name = type(exc).__name__
            message = str(exc)
            if "Authentication" in name or "PermissionDenied" in name or "401" in message or "-10007" in message:
                if not STOP_EVENT.is_set():
                    STOP_EVENT.set()
                    with _LOCK:
                        _STATS["fatal-auth"] += 1
                    log(f"鉴权/额度失败，停止生成：{name} {message[:120]}")
                return accepted, tries
            with _LOCK:
                _STATS[f"error:{name}"] += 1
            time.sleep(2 * tries)
            continue
        records = parse_records(text)
        with _LOCK:
            _STATS["responses"] += 1
            _STATS["parsed"] += len(records)
        path = output_path(category, task["id"], lang)
        rows = []
        for payload in records:
            ok, reason = validate(payload, task, lang)
            if not ok:
                with _LOCK:
                    _STATS[f"reject:{reason}"] += 1
                continue
            state = payload["state"].strip()
            key = fingerprint(state)
            with _LOCK:
                if key in _SEEN[(category, lang)]:
                    _STATS["dup"] += 1
                    continue
                _SEEN[(category, lang)].add(key)
            rows.append({"task": task["id"], "lang": lang, "state": state, "labels": payload["labels"]})
        if rows:
            with _LOCK:
                with path.open("a", encoding="utf-8", newline="\n") as handle:
                    for row in rows:
                        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                _WRITTEN[(category, task["id"], lang)] += len(rows)
            accepted += len(rows)
    return accepted, tries


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="并发调用 LLM 生成语料种子")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--batch", type=int, default=BATCH)
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--probe", type=int, default=0, help="只跑 N 个批次")
    parser.add_argument("--categories", nargs="*", default=None)
    args = parser.parse_args(argv)

    if not API_BASE or not MODEL:
        print("LLM_API_BASE / LLM_MODEL 未设置：两者都必须由环境变量提供，脚本内不写死网关与模型名")
        return 2
    if not API_KEY:
        print("LLM_API_KEY 未设置")
        return 2

    cats = args.categories or categories()
    state = load_state()
    load_existing()

    plan: list[tuple[str, dict, dict, str, int]] = []
    for category in cats:
        spec = load_spec(category)
        if PER_TASK_OVERRIDE:
            per_task = PER_TASK_OVERRIDE
        else:
            per_task = int(((sum(TARGET_PER_LANG.values()) // max(len(cats), 1)) // TASKS_PER_CATEGORY) * OVERSUPPLY)
        for task in spec["tasks"]:
            for lang in GEN_LANGS:
                key = f"{category}|{task['id']}|{lang}"
                done = int(state.get(key, 0)) + _WRITTEN[(category, task["id"], lang)]
                need = per_task - done
                if need <= 0:
                    continue
                batches = (need + args.batch - 1) // args.batch
                for _ in range(batches):
                    plan.append((category, spec, task, lang, args.batch))
    random.shuffle(plan)
    log(f"plan: {len(plan)} 批 × {args.batch} 条 (workers={args.workers})")
    if args.plan:
        return 0
    if args.probe:
        plan = plan[: args.probe]

    started = time.time()
    completed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(run_batch, category, spec, task, lang, count, index): (category, task["id"], lang)
            for index, (category, spec, task, lang, count) in enumerate(plan)
        }
        for future in as_completed(futures):
            category, task_id, lang = futures[future]
            try:
                accepted, tries = future.result()
            except Exception as exc:  # noqa: BLE001
                with _LOCK:
                    _STATS[f"fatal:{type(exc).__name__}"] += 1
                accepted, tries = 0, 0
            key = f"{category}|{task_id}|{lang}"
            with _LOCK:
                state[key] = int(state.get(key, 0)) + accepted
            completed += 1
            if completed % 20 == 0:
                total = sum(_WRITTEN.values())
                elapsed = time.time() - started
                rate = total / max(elapsed, 1) * 3600
                log(f"进度 {completed}/{len(plan)} 批，写入 {total} 条，{rate:.0f} 条/小时，stats={dict(_STATS)}")
                save_state(state)
    save_state(state)
    log(f"结束：写入 {sum(_WRITTEN.values())} 条，stats={dict(_STATS)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
