"""导出 LLM 撰写所需的任务规格：问题定义、候选集、标签域、真实样例（临时脚本，不提交）。"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from inject_translations import inject  # noqa: E402
from nano_jev.quotas import LANGUAGE_PLAN  # noqa: E402
from nano_jev.synth import MODULE_NAMES  # noqa: E402

LANG_KEY = {
    "zh-Hans": "zh", "en": "en", "zh-Hant": "zh-Hant", "ru": "ru",
    "ko": "ko", "ja": "ja", "ar": "ar", "bo": "bo",
}

OUT = ROOT / "artifacts" / "llm" / "specs"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    index = {}
    for name in MODULE_NAMES:
        try:
            mod = importlib.import_module(f"nano_jev.synth.{name}")
        except ImportError:
            continue
        category = getattr(mod, "CATEGORY", None)
        if not category:
            continue
        stats = inject(category, apply=True)
        tasks = []
        for task in mod.TASKS:
            questions = {}
            for qname, question in task.questions.items():
                questions[qname] = {
                    "type": question.qtype,
                    "instructions": {lang: list(question.instructions.get(key, ())) for lang, key in LANG_KEY.items()},
                    "criteria": (
                        {lang: question.criteria.get(key) for lang, key in LANG_KEY.items()}
                        if question.criteria else None
                    ),
                }
            tasks.append({"id": task.id, "title": task.title, "questions": questions})
        payload = {"category": category, "languages": list(LANGUAGE_PLAN), "tasks": tasks,
                   "inject": stats}
        (OUT / f"{category}.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        index[category] = [t["id"] for t in tasks]
    (OUT / "_index.json").write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(index, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
