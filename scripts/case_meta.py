"""案情元数据：占位符渲染、case_id / question_lang / source_lang 字段。

背景（nano-jev docs/dataset-mixed-language-requirements.md §2、§4）：
- 每条记录要有 ``case_id``（同案情跨语言一致）、``question_lang``（问题语言）、
  ``source_lang``（该案情的源语言）；
- 交付数据里不允许残留 ``{date}`` / ``{ticket}`` 这类模板占位符。

渲染策略：占位符按 ``(case_id, 占位符名, 第几次出现)`` 确定性映射成**语言中立**的真实值
（日期、时间、编号、站点码等），同一案情在三种语言里渲染出完全相同的值，因此翻译对齐
关系与判定依据都不变。历史数据里还有 ``{客户}``、``{supplier}``、``{site2}`` 这类写法，
统一兜底成“前缀-编号”的引用码；英文记录里的中文占位符按角色词表转成英文角色名，
保证英文分片不出现中文字符。
"""

from __future__ import annotations

import hashlib
import re

# 模板占位符：不含空白、引号、冒号、逗号、中文标点，避免把 state 里内嵌的 JSON 对象误判成占位符。
PLACEHOLDER = re.compile(r"\{([^{}\s\"':,：，。“”‘’、；]{1,40})\}")
CASE_ID = re.compile(r"^[a-z_]+-(?:legacy-)?\d{6}$")
CJK = re.compile(r"[\u4e00-\u9fff]")

# 常见中文占位符 → 英文角色词（仅用于英文分片，避免中文字符残留）。
EN_ROLE = {
    "客户": "Customer",
    "客戶": "Customer",
    "顾客": "Customer",
    "客户昵称": "Customer handle",
    "客服专员": "Support agent",
    "客服專員": "Support agent",
    "某平台客服专员": "Platform support agent",
    "客户经理": "Account manager",
    "客戶經理": "Account manager",
    "供应商": "Supplier",
    "供應商": "Supplier",
    "批次": "Batch",
    "区域": "Region",
    "區域": "Region",
    "组长": "Team lead",
    "組長": "Team lead",
    "研发甲": "Engineer A",
    "研发乙": "Engineer B",
    "运维甲": "Operator A",
    "运维乙": "Operator B",
    "某实验室技术员": "Lab technician",
    "某医院行政值班人员": "Hospital duty officer",
    "计量班": "Metering crew",
    "計量班": "Metering crew",
    "员工": "Staff",
    "員工": "Staff",
}


def _digest(*parts: str) -> int:
    return int.from_bytes(hashlib.blake2b("\x1f".join(parts).encode(), digest_size=8).digest(), "big")


def _date(seed: int) -> str:
    day = _digest(str(seed), "date")
    month = 9 + day % 3            # 2026-09 ~ 2026-11
    d = 1 + (day >> 8) % 28
    return f"2026-{month:02d}-{d:02d}"


def _time(seed: int) -> str:
    value = _digest(str(seed), "time")
    return f"{(value >> 3) % 24:02d}:{(value >> 11) % 60:02d}"


def render_token(name: str, case_id: str, index: int, lang: str = "zh-Hans") -> str | None:
    """把一个占位符渲染成真实值（语言中立）；未知写法兜底成引用码。"""
    base = re.sub(r"\d+$", "", name)
    seed = _digest(case_id, name, str(index))
    if base == "date":
        return _date(seed)
    if base == "time":
        return _time(seed)
    if base == "ticket":
        return f"TK-{_date(seed).replace('-', '')}-{seed % 10000:04d}"
    if base == "order_no":
        return f"SO-{_date(seed).replace('-', '')}-{seed % 1000000:06d}"
    if base == "case_id":
        suffix = case_id.rsplit("-", 1)[-1]
        return f"CS-{suffix}"
    if base == "customer":
        return f"C-{seed % 100000:05d}"
    if base == "site":
        return f"S-{seed % 1000:03d}"
    if base == "device":
        return f"DEV-{seed % 10000:04d}"
    if base == "batch":
        return f"B-{seed % 100000:05d}"
    if base == "amount":
        return f"{1 + seed % 9900}.{seed % 100:02d}"
    if base == "severity":
        return str(1 + seed % 4)
    if base == "priority":
        return str(1 + seed % 3)
    if CJK.search(name):
        role = EN_ROLE.get(name) or EN_ROLE.get(base)
        if lang == "en":
            if "甲" in name:
                role = role or "Contact A"
            elif "乙" in name:
                role = role or "Contact B"
            elif "丁" in name:
                role = role or "Contact D"
            elif "丙" in name:
                role = role or "Contact C"
            role = role or "Contact"
            return f"{role} {seed % 1000:03d}"
        return f"{name}{seed % 1000:03d}"
    # ASCII 历史写法（{supplier} / {merchant} / {phone} …）：前缀 + 编号。
    prefix = re.sub(r"[^A-Za-z]", "", base).upper()[:4] or "REF"
    return f"{prefix}-{seed % 10000:04d}"


def render_state(text: str, case_id: str, lang: str = "zh-Hans") -> tuple[str, list[str]]:
    """渲染模板占位符；返回 (新文本, 未识别写法列表)。"""
    counter: dict[str, int] = {}

    def replace(match: re.Match) -> str:
        name = match.group(1)
        index = counter.get(name, 0)
        counter[name] = index + 1
        value = render_token(name, case_id, index, lang)
        return value if value is not None else match.group(0)

    return PLACEHOLDER.sub(replace, text), []


def residual_placeholders(text: str) -> list[str]:
    return PLACEHOLDER.findall(text)


def legacy_case_id(category: str, index: int) -> str:
    """旧数据（三语言独立生成、无跨语言对应）分配互不相同的 case_id。"""
    return f"{category}-legacy-{index:06d}"


def is_case_id(value: object) -> bool:
    return isinstance(value, str) and bool(CASE_ID.match(value))
