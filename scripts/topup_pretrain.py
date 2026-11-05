"""把预训练集补齐到“每语言每垂类 >= 3000 条”。

流水线
------
1. 以简体中文（zh-Hans）为源语言，按缺口生成 4 条/请求；
2. 把每条源记录对齐翻译成繁体中文（zh-Hant）与英文（en），同样 4 条/请求；
3. 合并进 train/00-llm.<category>.<lang>.jsonl，保持三语标签一致；
4. 输出缺口、生成、翻译、合并报告。

并发口径（硬性）
----------------
源语言生成：2 并发 × 4 条/请求。
对齐翻译：2 并发 × 4 条/请求，两个目标语言共用同一组并发。

凭据只从环境变量读取：
    LLM_API_BASE / LLM_API_KEY / LLM_MODEL
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import queue
import random
import re
import sys
import threading
import time
from collections import Counter, defaultdict
from pathlib import Path

from openai import OpenAI
from opencc import OpenCC

ROOT = Path(__file__).resolve().parents[1]
SPECS = ROOT / "specs"
TRAIN = ROOT / "train"
SPLIT_DIRS = ("train", "train/posttrain", "test", "dev")
DEFAULT_WORK = ROOT / "artifacts" / "llm" / "topup"
LANGS = ("zh-Hans", "zh-Hant", "en")
SOURCE = "zh-Hans"
TARGETS = ("zh-Hant", "en")
TARGET_TOTAL = 3000
BATCH = 4
WORKERS = 2
CJK = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
PLACEHOLDER = re.compile(r"\{[a-z_]+\}")
ANY_BRACE = re.compile(r"\{[^{}]*\}")
NUMBER = re.compile(r"\d+(?:[.,]\d+)*")
T2S, S2T = OpenCC("t2s"), OpenCC("s2twp")
# OpenCC 会把“覆/著/於/藉”等简体也用字一并转掉，避免误判为繁体。
SIMPLIFIED_OK = set("覆著於藉瞭麽")


def is_simplified(text: str) -> bool:
    if T2S.convert(text) == text:
        return True
    changed = {char for char in text if T2S.convert(char) != char}
    return changed <= SIMPLIFIED_OK
LOG_LOCK = threading.Lock()
WRITE_LOCK = threading.Lock()
STOP = threading.Event()
CLIENT: OpenAI | None = None
LOG_PATH: Path | None = None

STYLES = [
    "工单正文", "内部邮件", "即时通讯群聊记录", "系统日志片段", "值班电话记录转述",
    "审批备注", "表格粘贴片段", "JSON 导出片段", "交接班便签", "客户投诉原文",
    "监控告警窗口", "会议纪要摘录", "短信通报", "复盘报告摘要", "供应商来函",
]

CONTEXT = {
    "agent_tool_governance": {
        "topics": ["智能体调用生产配置", "工具权限与审批链", "任务路由与编排", "跨系统写操作"],
        "personas": ["平台值班工程师", "某银行风控专员", "某物流调度员", "某医院信息科值班员"],
    },
    "content_safety_ops": {
        "topics": ["UGC 内容审核", "创作者申诉复核", "评论举报处置", "账号风险分级"],
        "personas": ["内容审核员", "社区运营专员", "创作者支持专员", "风控策略运营"],
    },
    "embodied_game_ops": {
        "topics": ["NPC 行为决策", "机器人任务调度", "游戏经济异常", "对局事件升级"],
        "personas": ["游戏运营值班员", "NPC 行为策划", "机器人调度员", "客服专员"],
    },
    "energy_utilities": {
        "topics": ["电网告警处置", "计量异常核查", "负荷调度与限电", "新能源并网"],
        "personas": ["电网调度员", "计量运维工程师", "供电所值班长", "新能源场站值班员"],
    },
    "finance_insurance": {
        "topics": ["理赔分诊", "交易反欺诈", "信贷审批", "保险核保"],
        "personas": ["理赔审核员", "银行风控专员", "信贷审批经理", "反欺诈值班员"],
    },
    "healthcare_logistics": {
        "topics": ["床位分配", "医疗设备调度", "耗材短缺应对", "院内转运"],
        "personas": ["医院行政值班员", "设备科工程师", "护理部调度员", "供应链专员"],
    },
    "hr_recruiting": {
        "topics": ["简历筛选", "面试排期", "离职风险预警", "编制与招聘计划"],
        "personas": ["招聘 HRBP", "用人经理", "招聘专员", "员工关系专员"],
    },
    "life_sciences_lab": {
        "topics": ["检测质控", "实验室安全", "试剂分装", "样本流转"],
        "personas": ["实验室技术员", "质控专员", "生物安全员", "样本管理员"],
    },
    "manufacturing_quality": {
        "topics": ["缺陷分诊", "设备维修派工", "工单路由", "来料检验"],
        "personas": ["质量工程师", "设备工程师", "生产班长", "工艺工程师"],
    },
    "public_sector_services": {
        "topics": ["政务事项分办", "惠民政策资格", "投诉工单路由", "窗口服务"],
        "personas": ["政务热线坐席", "街道办工作人员", "审批窗口人员", "督办专员"],
    },
    "retail_ecommerce": {
        "topics": ["订单异常处理", "商品内容审核", "客服工单路由", "售后纠纷"],
        "personas": ["电商客服专员", "订单运营", "内容审核员", "售后主管"],
    },
    "supply_chain_logistics": {
        "topics": ["运输风险", "入库上架路由", "供应商审计", "仓配异常"],
        "personas": ["物流调度员", "仓储主管", "供应商质量工程师", "关务专员"],
    },
    "telecom_network_ops": {
        "topics": ["告警关联", "网络工单派发", "容量规划", "基站退服"],
        "personas": ["网络运维值班员", "无线优化工程师", "传输工程师", "客服支撑专员"],
    },
}


def log(message: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}"
    with LOG_LOCK:
        if LOG_PATH is not None:
            with LOG_PATH.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
        print(message.encode("ascii", "replace").decode("ascii"), flush=True)


def fingerprint(text: str) -> str:
    return hashlib.blake2b(re.sub(r"\s+", "", text).lower().encode(), digest_size=8).hexdigest()


def load_specs() -> dict[str, dict]:
    out = {}
    for path in sorted(SPECS.glob("*.json")):
        if path.stem.startswith("_"):
            continue
        out[path.stem] = json.loads(path.read_text(encoding="utf-8"))
    return out


def question_schema(spec: dict, task: dict, lang: str, variant: int = 0) -> dict:
    questions = {}
    for name, question in task["questions"].items():
        variants = question["instructions"][lang]
        questions[name] = {
            "type": question["type"],
            "instructions": variants[variant % len(variants)],
            "criteria": question["criteria"][lang],
        }
    return questions


def domain_of(question: dict) -> set:
    qtype = question["type"]
    if qtype == "noul":
        return {0, 1}
    criteria = question["criteria"]
    return set(criteria) if qtype == "choice" else set(range(len(criteria)))


def count_split_records() -> Counter:
    counter: Counter = Counter()
    for relative in SPLIT_DIRS:
        folder = ROOT / relative
        if not folder.exists():
            continue
        for path in sorted(folder.glob("*.jsonl")):
            for line in path.open(encoding="utf-8"):
                if not line.strip():
                    continue
                record = json.loads(line)
                counter[(record["category"], record["lang"])] += 1
    return counter


def load_existing() -> tuple[Counter, dict[tuple[str, str], set[str]], dict[tuple[str, str], list[dict]]]:
    """返回 train 计数、各 (category, lang) 的 state 指纹、以及任务样例。"""
    counts: Counter = Counter()
    seen: dict[tuple[str, str], set[str]] = defaultdict(set)
    examples: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for relative in SPLIT_DIRS:
        folder = ROOT / relative
        if not folder.exists():
            continue
        for path in sorted(folder.glob("*.jsonl")):
            for line in path.open(encoding="utf-8"):
                if not line.strip():
                    continue
                record = json.loads(line)
                category, lang = record["category"], record["lang"]
                if relative == "train" and path.parent == TRAIN:
                    counts[(category, lang)] += 1
                seen[(category, lang)].add(fingerprint(record["state"]))
                if lang == SOURCE and len(examples[(category, record["task"])]) < 80:
                    examples[(category, record["task"])].append(
                        {"state": record["state"], "labels": record["labels"]}
                    )
    return counts, seen, examples


def compute_plan(specs: dict[str, dict], counts: Counter, target: int, only: list[str] | None) -> dict:
    plan = {"target": target, "categories": {}}
    for category in sorted(specs):
        if only and category not in only:
            continue
        current = {lang: counts[(category, lang)] for lang in LANGS}
        add = max(0, target - min(current.values()))
        tasks = [task["id"] for task in specs[category]["tasks"]]
        base, extra = divmod(add, len(tasks))
        allocations = {task: base + (1 if index < extra else 0) for index, task in enumerate(tasks)}
        plan["categories"][category] = {"current": current, "add": add, "tasks": allocations}
    return plan


def load_seed_store(path: Path) -> tuple[list[dict], dict[str, dict], dict[tuple[str, str], dict], int]:
    sources: list[dict] = []
    source_by_id: dict[str, dict] = {}
    translations: dict[tuple[str, str], dict] = {}
    max_seq = 0
    if path.exists():
        for line in path.open(encoding="utf-8"):
            if not line.strip():
                continue
            row = json.loads(line)
            match = re.search(r"(\d+)$", row.get("row_id", ""))
            if match:
                max_seq = max(max_seq, int(match.group(1)))
            if row.get("kind") == "source":
                sources.append(row)
                source_by_id[row["source_id"]] = row
            elif row.get("kind") == "translation":
                translations[(row["source_id"], row["lang"])] = row
    return sources, source_by_id, translations, max_seq


class Store:
    def __init__(self, path: Path, translation_path: Path | None = None):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.sources, self.source_by_id, self.translations, self.seq = load_seed_store(path)
        if translation_path is not None and translation_path.exists():
            for line in translation_path.open(encoding="utf-8"):
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("kind") == "translation":
                    self.translations[(row["source_id"], row["lang"])] = row
        self.counts = Counter()
        for row in self.sources:
            self.counts[(row["category"], row["task"], row["lang"])] += 1

    def next_id(self) -> str:
        with WRITE_LOCK:
            self.seq += 1
            return f"topup-{self.seq:06d}"

    def add_source(self, row: dict) -> None:
        with WRITE_LOCK:
            with self.path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            self.sources.append(row)
            self.source_by_id[row["source_id"]] = row
            self.counts[(row["category"], row["task"], row["lang"])] += 1

    def add_translation(self, row: dict) -> None:
        with WRITE_LOCK:
            with self.path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            self.translations[(row["source_id"], row["lang"])] = row


def validate_state(state: str, lang: str, task: dict) -> tuple[bool, str]:
    if not isinstance(state, str):
        return False, "type"
    text = state.strip()
    if lang == SOURCE:
        if not 140 <= len(text) <= 1000:
            return False, "length"
    elif lang == "zh-Hant":
        if not 100 <= len(text) <= 1600:
            return False, "length"
    else:
        if not 300 <= len(text) <= 4000:
            return False, "length"
    braces = ANY_BRACE.findall(text)
    if any(not PLACEHOLDER.fullmatch(item) for item in braces):
        return False, "placeholder"
    if lang == "en":
        if CJK.search(text):
            return False, "cjk-in-en"
    elif lang == "zh-Hans":
        if not is_simplified(text):
            return False, "not-simplified"
    else:
        if T2S.convert(text) == text:
            return False, "not-traditional"
    for name in task["questions"]:
        if re.search(rf"(?<![A-Za-z]){re.escape(name)}(?![A-Za-z])", text):
            return False, "field-leak"
    return True, "ok"


def validate_labels(labels: object, task: dict) -> tuple[bool, str]:
    if not isinstance(labels, dict) or set(labels) != set(task["questions"]):
        return False, "keys"
    for name, question in task["questions"].items():
        value = labels[name]
        if question["type"] == "noul":
            if value not in (0, 1) or isinstance(value, bool):
                return False, "noul"
        elif question["type"] == "choice":
            if value not in question["criteria"]["zh-Hans"]:
                return False, "choice"
        else:
            levels = question["criteria"]["zh-Hans"]
            if not isinstance(value, int) or isinstance(value, bool) or not 0 <= value < len(levels):
                return False, "score"
    return True, "ok"


def parse_json_records(text: str) -> list[dict]:
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
    return [item for item in payloads if isinstance(item, dict)]


def client() -> OpenAI:
    global CLIENT
    if CLIENT is None:
        base = os.environ.get("LLM_API_BASE", "").rstrip("/")
        key = os.environ.get("LLM_API_KEY", "").strip()
        if not base or not key:
            raise RuntimeError("LLM_API_BASE / LLM_API_KEY 未设置")
        CLIENT = OpenAI(api_key=key, base_url=base, max_retries=0, timeout=int(os.environ.get("LLM_TIMEOUT", "300")))
    return CLIENT


def call_api(messages: list[dict], *, temperature: float, max_tokens: int) -> str:
    model = os.environ.get("LLM_MODEL", "")
    if not model:
        raise RuntimeError("LLM_MODEL 未设置")
    last: Exception | None = None
    for attempt in range(4):
        if STOP.is_set():
            raise RuntimeError("stop requested")
        try:
            kwargs = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "response_format": {"type": "json_object"},
                "timeout": int(os.environ.get("LLM_TIMEOUT", "300")),
            }
            effort = os.environ.get("LLM_REASONING_EFFORT", "").strip()
            if effort:
                kwargs["reasoning_effort"] = effort
            response = client().chat.completions.create(**kwargs)
            return response.choices[0].message.content or ""
        except Exception as exc:  # noqa: BLE001
            last = exc
            name = type(exc).__name__
            message = str(exc)
            if any(marker in name for marker in ("Authentication", "PermissionDenied")) or "401" in message or "403" in message:
                STOP.set()
                log(f"鉴权失败，停止：{name} {message[:120]}")
                raise
            time.sleep(min(2 ** attempt, 30))
    raise RuntimeError(f"API 连续失败：{type(last).__name__} {str(last)[:120]}")


def question_block(task: dict, lang: str) -> str:
    lines = []
    for name, question in task["questions"].items():
        instruction = question["instructions"][lang][0]
        criteria = question["criteria"][lang]
        if question["type"] == "noul":
            options = f"0 = {criteria.get('false')} / 1 = {criteria.get('true')}"
        elif question["type"] == "choice":
            options = " | ".join(f"{key} = {label}" for key, label in criteria.items())
        else:
            options = " | ".join(f"{index} = {label}" for index, label in enumerate(criteria))
        lines.append(f"- {name}（{question['type']}）：{instruction}\n  取值：{options}")
    return "\n".join(lines)


def build_generation_prompt(category: str, task: dict, examples: list[dict], batch_no: int) -> str:
    rng = random.Random(f"{category}:{task['id']}:{batch_no}")
    context = CONTEXT[category]
    topics = context["topics"]
    personas = context["personas"]
    topic = rng.choice(topics)
    persona = rng.choice(personas)
    styles = rng.sample(STYLES, k=6)
    sample = examples[:]
    rng.shuffle(sample)
    sample = sample[:2]
    labels_shape = ", ".join(f'"{name}": <取值>' for name in task["questions"])
    user = f"""业务场景：{category} / {task['id']}（{task['title']}）

三个待判定问题：
{question_block(task, SOURCE)}

请写 {BATCH} 条该场景下的真实简体中文业务输入。

硬性要求：
1. 每条 180-450 字，像真实工单、内部邮件、群聊、系统导出、审批备注、值班记录等；4 条文体与场景要明显不同。
2. 每条 state 必须写清推出三个标签所需的全部事实：分值、阈值、分档、累计分、条件是否满足等，数字写死；标签必须能由 state 中的显式事实和 state 里写明的规则唯一推出。
3. 每条加入 1-2 句与判定无关的干扰信息（别的工单、设备、时间、人物等），不得改变关键信号。
4. 非关键实体用占位符：{{case_id}} {{date}} {{time}} {{ticket}} {{customer}} {{order_no}} {{site}} {{device}} {{batch}}；不得出现真实人名、公司名、机构名。
5. 不得出现字段名（如 {', '.join(task['questions'])}），不得写“答案是”。
6. 4 条不得套模板，场景、数字、句式都要不同。
7. 主题参考：{topic}；涉事方参考：{persona}；文体参考：{', '.join(styles)}。

只输出 JSON 对象，格式：
{{"records": [{{"state": "<输入原文>", "labels": {{{labels_shape}}}}}, ...]}}
records 必须正好有 {BATCH} 个元素。
标签键：{', '.join(task['questions'])}。

已有样例（只示范写法与判定粒度，禁止照抄其场景、数字或规则）：
{json.dumps(sample, ensure_ascii=False)}"""
    return user


def generation_job(job: tuple[str, str, int, int], store: Store, specs: dict, examples: dict, seen: dict) -> int:
    category, task_id, batch_no, target = job
    if STOP.is_set():
        return 0
    current = store.counts[(category, task_id, SOURCE)]
    if current >= target:
        return 0
    spec = specs[category]
    task = next(item for item in spec["tasks"] if item["id"] == task_id)
    prompt = build_generation_prompt(category, task, examples.get((category, task_id), []), batch_no)
    try:
        text = call_api(
            [
                {"role": "system", "content": "你是资深业务数据撰写员。只输出 JSON，不要解释，不要 Markdown 代码块。"},
                {"role": "user", "content": prompt},
            ],
            temperature=1.0,
            max_tokens=12000,
        )
    except Exception as exc:  # noqa: BLE001
        log(f"生成失败 {category}/{task_id}：{type(exc).__name__} {str(exc)[:100]}")
        return 0
    accepted = 0
    for payload in parse_json_records(text):
        if accepted >= BATCH:
            break
        state = payload.get("state")
        labels = payload.get("labels")
        ok, reason = validate_state(str(state or ""), SOURCE, task)
        if not ok:
            continue
        ok, reason = validate_labels(labels, task)
        if not ok:
            continue
        digest = fingerprint(state)
        with WRITE_LOCK:
            if digest in seen[(category, SOURCE)]:
                continue
            seen[(category, SOURCE)].add(digest)
        source_id = store.next_id()
        row = {
            "kind": "source",
            "row_id": source_id,
            "source_id": source_id,
            "category": category,
            "task": task_id,
            "lang": SOURCE,
            "state": str(state).strip(),
            "labels": labels,
        }
        store.add_source(row)
        accepted += 1
    return accepted


def build_translation_prompt(rows: list[dict], target: str) -> str:
    lang_name = {"zh-Hant": "繁体中文（台湾用语，不得出现简体字）", "en": "英文（不得出现任何中文字符）"}[target]
    items = [{"i": index, "text": row["state"]} for index, row in enumerate(rows)]
    required = sorted({
        *PLACEHOLDER.findall(" ".join(row["state"] for row in rows)),
        *NUMBER.findall(" ".join(row["state"] for row in rows)),
    })
    return f"""把下面 {len(rows)} 条简体中文业务语料逐条翻译成{lang_name}。

要求：
1. 逐条一一对应，条数不变，顺序不变。
2. 保留所有占位符原样（如 {{ticket}}、{{date}}）与所有数字、单位、编号，不得改写或省略。
3. 只翻译语言，不改变事实、结论、程度与判定规则；不要添加解释。
4. 中文的“以上 / 以下 / 及以上 / 及以下”包含边界：英文写成 “N or more / N or fewer / at least N / at most N”，
   不要写成 “above N / below N”（不含边界），避免阈值含义变化。
5. 下列数字与占位符必须在译文中原样出现（可多不可少）：{', '.join(required)}
6. 输出 JSON 对象：{{"records": [{{"i": 0, "text": "<译文>"}}, ...]}}

输入：
{json.dumps(items, ensure_ascii=False)}"""


def translation_validate(text: str, source: str, lang: str) -> tuple[bool, str]:
    if not isinstance(text, str) or not text.strip():
        return False, "empty"
    text = text.strip()
    if set(PLACEHOLDER.findall(text)) != set(PLACEHOLDER.findall(source)):
        return False, "placeholder"
    remaining = list(NUMBER.findall(text))
    for token in NUMBER.findall(source):
        if token in remaining:
            remaining.remove(token)
            continue
        units = re.search(re.escape(token) + r"\s*(万|亿)", source)
        hit = None
        if units:
            base = int(float(token.replace(",", "")))
            scaled = base * (10000 if units.group(1) == "万" else 100000000)
            hit = next((item for item in (str(scaled), f"{scaled:,}") if item in remaining), None)
        if hit is not None:
            remaining.remove(hit)
            continue
        return False, "number"
    return validate_state(text, lang, {"questions": {}})


def translate_raw(rows: list[dict], target: str) -> list[str | None]:
    """调用一次翻译，返回未校验的原始译文（缺失项为 None）。"""
    if not rows:
        return []
    try:
        text = call_api(
            [
                {"role": "system", "content": "你是专业本地化译者，只输出 JSON。"},
                {"role": "user", "content": build_translation_prompt(rows, target)},
            ],
            temperature=0.3,
            max_tokens=10000,
        )
    except Exception as exc:  # noqa: BLE001
        log(f"翻译失败 {target}：{type(exc).__name__} {str(exc)[:100]}")
        return [None] * len(rows)
    parsed = {}
    for item in parse_json_records(text):
        index = item.get("i")
        if isinstance(index, int) and 0 <= index < len(rows):
            parsed[index] = item.get("text")
    return [parsed.get(index) if isinstance(parsed.get(index), str) else None for index in range(len(rows))]


def translate_rows(rows: list[dict], target: str) -> list[str | None]:
    """批量翻译并逐条校验，不合格的返回 None。"""
    if not rows:
        return []
    out: list[str | None] = []
    for row, candidate in zip(rows, translate_raw(rows, target)):
        if candidate is None:
            out.append(None)
            continue
        ok, _ = translation_validate(candidate, row["state"], target)
        out.append(candidate.strip() if ok else None)
    return out


def missing_tokens(source: str, translated: str) -> list[str]:
    """列出译文里缺失的占位符与数字（含“万/亿”换算后的等价写法）。"""
    missing: list[str] = []
    if set(PLACEHOLDER.findall(translated)) != set(PLACEHOLDER.findall(source)):
        missing.extend(sorted(set(PLACEHOLDER.findall(source)) - set(PLACEHOLDER.findall(translated))))
    remaining = list(NUMBER.findall(translated))
    for token in NUMBER.findall(source):
        if token in remaining:
            remaining.remove(token)
            continue
        units = re.search(re.escape(token) + r"\s*(万|亿)", source)
        hit = None
        if units:
            base = int(float(token.replace(",", "")))
            scaled = base * (10000 if units.group(1) == "万" else 100000000)
            hit = next((item for item in (str(scaled), f"{scaled:,}") if item in remaining), None)
        if hit is not None:
            remaining.remove(hit)
            continue
        missing.append(token)
    return missing


def repair_text(row: dict, target: str, previous: str, tokens: list[str]) -> str | None:
    """把上一版译文与缺失的 token 一起回灌，要求模型只补全这些 token。"""
    lang_name = {"zh-Hant": "繁体中文（台湾用语）", "en": "英文（不得出现任何中文字符）"}[target]
    prompt = f"""下面是一条简体中文业务记录，以及上一版{lang_name}译文。
上一版译文遗漏或改写了这些必须原样保留的内容：{', '.join(tokens)}。

请重译整条记录，确保：
1. 逐句对应，事实、结论、分档规则不变；
2. 上面列出的每一项都按原样出现在译文中（数字不得改写、不得换算、不得省略）；
3. 占位符 {{...}} 原样保留；
4. 只输出 JSON：{{"text": "<完整译文>"}}。

简体原文：
{row['state']}

上一版译文：
{previous}"""
    try:
        text = call_api(
            [
                {"role": "system", "content": "你是专业本地化译者，只输出 JSON。"},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=10000,
        )
    except Exception as exc:  # noqa: BLE001
        log(f"修复失败 {target} {row['source_id']}：{type(exc).__name__} {str(exc)[:80]}")
        return None
    for item in parse_json_records(text):
        candidate = item.get("text")
        if isinstance(candidate, str) and candidate.strip():
            ok, _ = translation_validate(candidate, row["state"], target)
            if ok:
                return candidate.strip()
    return None


def translate_with_repair(row: dict, target: str, attempts: int = 5) -> str | None:
    """单条翻译：先直译，失败则带着缺失 token 回灌修复，最多 attempts 轮。"""
    last_text = ""
    for _ in range(attempts):
        if last_text:
            tokens = missing_tokens(row["state"], last_text)
            if tokens:
                fixed = repair_text(row, target, last_text, tokens)
                if fixed:
                    return fixed
        raw = translate_raw([row], target)
        candidate = raw[0] if raw else None
        if candidate is None:
            continue
        ok, _ = translation_validate(candidate, row["state"], target)
        if ok:
            return candidate.strip()
        last_text = candidate
    return None




def translation_job(job: tuple[str, list[dict]], store: Store) -> int:
    target, rows = job
    if STOP.is_set():
        return 0
    pending = [row for row in rows if (row["source_id"], target) not in store.translations]
    accepted = 0
    attempts = 0

    def save(row: dict, text: str) -> bool:
        if (row["source_id"], target) in store.translations:
            return False
        store.add_translation({
            "kind": "translation",
            "row_id": f"{row['source_id']}-{target}",
            "source_id": row["source_id"],
            "category": row["category"],
            "task": row["task"],
            "lang": target,
            "state": text,
            "labels": row["labels"],
        })
        return True

    while pending and attempts < 4:
        attempts += 1
        texts = translate_rows(pending, target)
        failed = []
        for row, text in zip(pending, texts):
            if text is None:
                failed.append(row)
            elif save(row, text):
                accepted += 1
        if not failed:
            break
        pending = failed
        if len(pending) == 1:
            row = pending[0]
            if target == "zh-Hant":
                text = S2T.convert(row["state"])
                ok, _ = translation_validate(text, row["state"], target)
                if ok and save(row, text):
                    accepted += 1
                    break
            log(f"翻译跳过 {target} {row['source_id']}")
            break
        # 整批失败后改为逐条重试，避免一条坏样本拖垮整批。
        for row in list(pending):
            single = translate_rows([row], target)
            text = single[0] if single else None
            if text is not None and save(row, text):
                accepted += 1
                pending.remove(row)
        if not pending:
            break
    return accepted


def worker_loop(jobs: queue.Queue, handler, stop_after: int = 0) -> None:
    processed = 0
    while not STOP.is_set():
        try:
            job = jobs.get(timeout=5)
        except queue.Empty:
            return
        try:
            handler(job)
        except Exception as exc:  # noqa: BLE001
            log(f"任务异常 {type(exc).__name__} {str(exc)[:100]}")
        finally:
            jobs.task_done()
        processed += 1
        if stop_after and processed >= stop_after:
            return


def run_jobs(jobs: list, handler, workers: int = WORKERS) -> None:
    work: queue.Queue = queue.Queue()
    for job in jobs:
        work.put(job)
    threads = [threading.Thread(target=worker_loop, args=(work, handler), daemon=True) for _ in range(workers)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if not work.empty():
        log(f"队列未清空：剩余 {work.qsize()}")


def generate(plan: dict, specs: dict, store: Store, seen: dict, examples: dict, work: Path) -> dict[str, int]:
    total_target = sum(entry["add"] for entry in plan["categories"].values())
    log(f"生成目标：{total_target} 条源记录（{WORKERS} 并发 × {BATCH} 条/请求）")

    def handler(job):
        accepted = generation_job(job, store, specs, examples, seen)
        if accepted:
            category, task_id, _, _ = job
            total = store.counts[(category, task_id, SOURCE)]
            log(f"生成 +{accepted} {category}/{task_id} -> {total}")

    for pass_no in range(1, 4):
        jobs = []
        for category, entry in plan["categories"].items():
            for task_id, target in entry["tasks"].items():
                done = store.counts[(category, task_id, SOURCE)]
                need = max(0, target - done)
                if need <= 0:
                    continue
                # 10% 超发，抵消拒收与重复；并发下每任务最多多出 1 批。
                batches = (need * 11 + BATCH * 10 - 1) // (BATCH * 10) + 1
                for batch_no in range(batches):
                    jobs.append((category, task_id, batch_no + pass_no * 1000, target))
        if not jobs:
            break
        random.Random(42 + pass_no).shuffle(jobs)
        log(f"生成第 {pass_no} 轮：{len(jobs)} 批")
        run_jobs(jobs, handler)

    missing = {}
    for category, entry in plan["categories"].items():
        for task_id, target in entry["tasks"].items():
            current = store.counts[(category, task_id, SOURCE)]
            if current < target:
                missing[f"{category}/{task_id}"] = target - current
    if missing:
        log(f"生成缺口：{missing}")
    else:
        log("生成完成")
    return missing


def translate(plan: dict, store: Store, work: Path) -> list[str]:
    sources = store.sources
    log(f"翻译目标：{len(sources) * len(TARGETS)} 条译文（{WORKERS} 并发 × {BATCH} 条/请求，两个目标语言共用并发）")

    def handler(job):
        target, rows = job
        accepted = translation_job(job, store)
        if accepted:
            log(f"翻译 +{accepted} {target}")

    for pass_no in range(1, 4):
        jobs = []
        for target in TARGETS:
            missing = [row for row in sources if (row["source_id"], target) not in store.translations]
            random.Random(43 + pass_no).shuffle(missing)
            for index in range(0, len(missing), BATCH):
                jobs.append((target, missing[index : index + BATCH]))
        if not jobs:
            break
        random.Random(44 + pass_no).shuffle(jobs)
        log(f"翻译第 {pass_no} 轮：{len(jobs)} 批")
        run_jobs(jobs, handler)

    missing = []
    for row in sources:
        for target in TARGETS:
            if (row["source_id"], target) not in store.translations:
                missing.append(f"{row['source_id']}:{target}")
    log(f"翻译完成：{len(sources) * len(TARGETS) - len(missing)}/{len(sources) * len(TARGETS)}，缺失 {len(missing)}")
    if missing:
        (work / "missing_translations.json").write_text(json.dumps(missing, ensure_ascii=False, indent=2), encoding="utf-8")
    return missing


def merge(plan: dict, specs: dict, store: Store, dry_run: bool = False) -> dict:
    report = {"added": {}, "skipped": {}, "languages": {}}
    # 按 category/lang 分组新记录；只合并既有源又有两个译文的完整三元组。
    complete_sources = [
        row for row in store.sources
        if all((row["source_id"], target) in store.translations for target in TARGETS)
    ]
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for source in complete_sources:
        grouped[(source["category"], source["lang"])].append(source)
        for target in TARGETS:
            grouped[(source["category"], target)].append(store.translations[(source["source_id"], target)])

    for category, entry in plan["categories"].items():
        spec = specs[category]
        tasks = {task["id"]: task for task in spec["tasks"]}
        for lang in LANGS:
            path = TRAIN / f"00-llm.{category}.{lang}.jsonl"
            existing_lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
            existing_states = set()
            blocked_states = set()
            max_index = 0
            for line in existing_lines:
                if not line.strip():
                    continue
                record = json.loads(line)
                existing_states.add(fingerprint(record["state"]))
                match = re.search(r"-(\d+)$", record["id"])
                if match:
                    max_index = max(max_index, int(match.group(1)))
            for relative in ("test", "dev", "train/posttrain"):
                other = ROOT / relative / f"00-llm.{category}.{lang}.jsonl"
                if not other.exists():
                    continue
                for line in other.open(encoding="utf-8"):
                    if line.strip():
                        blocked_states.add(fingerprint(json.loads(line)["state"]))
            new_rows = []
            skipped = 0
            for row in grouped.get((category, lang), []):
                if row["task"] not in tasks:
                    skipped += 1
                    continue
                task = tasks[row["task"]]
                ok, _ = validate_state(row["state"], lang, task)
                if not ok:
                    skipped += 1
                    continue
                ok, _ = validate_labels(row["labels"], task)
                if not ok:
                    skipped += 1
                    continue
                if fingerprint(row["state"]) in existing_states or fingerprint(row["state"]) in blocked_states:
                    skipped += 1
                    continue
                digest = hashlib.blake2b(row["state"].encode(), digest_size=2).digest()
                variant = digest[0] % 2
                questions = question_schema(spec, task, lang, variant)
                max_index += 1
                record = {
                    "id": f"train-llm-{category}-{lang.lower()}-{max_index:06d}",
                    "split": "train",
                    "task": row["task"],
                    "category": category,
                    "lang": lang,
                    "style": "llm",
                    "state": row["state"],
                    "questions": questions,
                    "labels": row["labels"],
                }
                new_rows.append(record)
                existing_states.add(fingerprint(record["state"]))
            report["added"][f"{category}/{lang}"] = len(new_rows)
            report["skipped"][f"{category}/{lang}"] = skipped
            report["languages"].setdefault(lang, 0)
            report["languages"][lang] += len(new_rows)
            if dry_run:
                continue
            with path.open("w", encoding="utf-8", newline="\n") as handle:
                for line in existing_lines:
                    if line.strip():
                        handle.write(line + "\n")
                for record in new_rows:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            log(f"合并 {category}/{lang} +{len(new_rows)}（跳过 {skipped}）")
    return report


def report_counts(specs: dict, target: int) -> dict:
    counts: Counter = Counter()
    for path in sorted(TRAIN.glob("*.jsonl")):
        category, lang = path.stem.split(".")[1], path.stem.split(".")[2]
        for line in path.open(encoding="utf-8"):
            if line.strip():
                counts[(category, lang)] += 1
    result = {}
    ok = True
    for category in sorted(specs):
        row = {lang: counts[(category, lang)] for lang in LANGS}
        result[category] = row
        if min(row.values()) < target:
            ok = False
    return {"ok": ok, "counts": result}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="补齐预训练集到每语言每垂类 >=3000 条")
    parser.add_argument("--target", type=int, default=TARGET_TOTAL)
    parser.add_argument("--categories", nargs="*", default=None)
    parser.add_argument("--work", default=str(DEFAULT_WORK))
    parser.add_argument("--translations", default="", help="独立的译文文件；默认为 <work>/translations.jsonl")
    parser.add_argument("--plan", action="store_true", help="只打印计划")
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--translate", action="store_true")
    parser.add_argument("--merge", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--rounds", type=int, default=1, help="生成/翻译循环轮数，直到无缺口或达到轮数")
    args = parser.parse_args(argv)

    specs = load_specs()
    counts, seen, examples = load_existing()
    plan = compute_plan(specs, counts, args.target, args.categories)
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    global LOG_PATH
    LOG_PATH = work / "topup.log"
    (work / "plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")

    total_add = sum(entry["add"] for entry in plan["categories"].values())
    log(f"目标 {args.target}：{len(plan['categories'])} 个垂类，需补 {total_add} 条源记录")
    for category, entry in plan["categories"].items():
        log(f"  {category}: {entry['current']} +{entry['add']} -> {entry['tasks']}")
    if args.plan:
        return 0

    store = Store(work / "seeds.jsonl", Path(args.translations) if args.translations else work / "translations.jsonl")
    log(f"已有种子：source={len(store.sources)} translation={len(store.translations)}")
    if args.generate or args.translate:
        last_missing: dict[str, int] = {}
        for round_no in range(1, max(1, args.rounds) + 1):
            missing: dict[str, int] = {}
            if args.generate:
                missing = generate(plan, specs, store, seen, examples, work)
            if args.translate:
                translate(plan, store, work)
            last_missing = missing
            if not missing:
                log(f"第 {round_no} 轮结束：生成已达标")
                break
            log(f"第 {round_no} 轮结束：仍有缺口 {sum(missing.values())} 条，继续下一轮")
        if args.generate and not last_missing:
            (work / "generation.done").write_text("done\n", encoding="utf-8")
            log(f"已写入完成标记 {work / 'generation.done'}")
    if args.merge:
        report = merge(plan, specs, store, dry_run=args.dry_run)
        (work / "merge_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        log(f"合并完成：{report['languages']}")
    if args.report or args.merge:
        result = report_counts(specs, args.target)
        (work / "count_report.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        log(f"计数达标：{result['ok']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
