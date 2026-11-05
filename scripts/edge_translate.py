"""Edge 翻译 API 客户端（免费、无需密钥）。

实测端点（2026-11-05）::

    POST https://edge.microsoft.com/translate/translatetext?to=<lang>&isEnterpriseClient=false
    content-type: application/json
    body: ["text1", "text2", ...]

响应：``[{"detectedLanguage": {...}, "translations": [{"text": "...", "to": "en"}]}, ...]``

用于把简体中文语料批量翻译成英文：单请求可带多条文本，实测 4 条约 0.7 秒，
且能原样保留 ``{date}`` / ``{ticket}`` 这类占位符与数字。
"""

from __future__ import annotations

import json
import os
import threading
import time
import urllib.error
import urllib.request

ENDPOINT = "https://edge.microsoft.com/translate/translatetext?to={lang}&isEnterpriseClient=false"
SUPPORTED = {"en", "zh-Hans", "zh-Hant"}

# 限流：所有线程共用同一个最小请求间隔，避免高频调用被服务端限流/封禁。
MIN_INTERVAL = float(os.environ.get("EDGE_MIN_INTERVAL", "1.2"))
MAX_CONSECUTIVE_FAILURES = int(os.environ.get("EDGE_MAX_FAILURES", "5"))
COOLDOWN_SECONDS = float(os.environ.get("EDGE_COOLDOWN", "300"))

_THROTTLE_LOCK = threading.Lock()
_LAST_CALL = [0.0]
_FAILURE_LOCK = threading.Lock()
_CONSECUTIVE_FAILURES = [0]


class EdgeTranslateError(RuntimeError):
    pass


def _throttle() -> None:
    with _THROTTLE_LOCK:
        wait = MIN_INTERVAL - (time.time() - _LAST_CALL[0])
        if wait > 0:
            time.sleep(wait)
        _LAST_CALL[0] = time.time()


def _note_failure() -> None:
    with _FAILURE_LOCK:
        _CONSECUTIVE_FAILURES[0] += 1
        count = _CONSECUTIVE_FAILURES[0]
    if count >= MAX_CONSECUTIVE_FAILURES:
        with _FAILURE_LOCK:
            _CONSECUTIVE_FAILURES[0] = 0
        # 连续失败：整体冷却，避免继续撞墙。
        time.sleep(COOLDOWN_SECONDS)


def _note_success() -> None:
    with _FAILURE_LOCK:
        _CONSECUTIVE_FAILURES[0] = 0


def _post(lang: str, texts: list[str], timeout: float = 40.0) -> list[str]:
    payload = json.dumps(texts, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        ENDPOINT.format(lang=lang),
        data=payload,
        headers={"content-type": "application/json"},
        method="POST",
    )
    _throttle()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw = response.read().decode("utf-8")
    _note_success()
    data = json.loads(raw)
    if not isinstance(data, list) or len(data) != len(texts):
        raise EdgeTranslateError(f"响应条数不匹配：{len(data) if isinstance(data, list) else 'n/a'} != {len(texts)}")
    out: list[str] = []
    for item in data:
        translations = item.get("translations") or []
        if not translations:
            raise EdgeTranslateError("响应缺少 translations")
        out.append(str(translations[0].get("text", "")))
    return out


def translate(texts: list[str], lang: str = "en", retries: int = 4) -> list[str]:
    """批量翻译；失败按指数退避重试，仍失败则抛 EdgeTranslateError。"""
    if not texts:
        return []
    if lang not in SUPPORTED:
        raise EdgeTranslateError(f"不支持的目标语言 {lang}")
    last: Exception | None = None
    for attempt in range(retries):
        try:
            return _post(lang, texts)
        except urllib.error.HTTPError as exc:
            last = exc
            _note_failure()
            # 429 / 403 / 5xx：长时间退避，避免被判为滥用。
            if exc.code in (429, 403) or 500 <= exc.code < 600:
                time.sleep(min(15 * (attempt + 1), 90))
                continue
            time.sleep(min(2 ** attempt, 8))
        except Exception as exc:  # noqa: BLE001
            last = exc
            _note_failure()
            time.sleep(min(2 ** attempt, 8))
    raise EdgeTranslateError(f"Edge 翻译失败：{type(last).__name__} {str(last)[:120]}")


if __name__ == "__main__":
    print(translate(["这是一次翻译测试。", "另一条带 {date} 与 12.5 的文本。"], "en"))
