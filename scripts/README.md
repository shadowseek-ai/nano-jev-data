# 生成与验证脚本

本目录是从 `nano-jev` 仓库的 `artifacts/llm/` 迁移过来的数据集生成/校验流水线。
所有脚本只从环境变量读取凭据，**仓库内不保存任何密钥**。

## 依赖

```bash
pip install openai opencc-python-reimplemented
```

## 环境变量

| 变量 | 说明 |
|---|---|
| `LLM_API_KEY` / `LLM_API_KEYS` | 必填。前者单个 key，后者逗号分隔多个 key 轮换 |
| `LLM_API_BASE` | 必填。OpenAI 兼容网关地址（脚本不写死，缺失即报错退出） |
| `LLM_MODEL` | 必填。模型名（脚本不写死，缺失即报错退出） |
| `LLM_MAX_TOKENS` | 单请求输出上限（生成默认 128000，翻译建议 16000） |
| `LLM_TIMEOUT` | 单请求超时秒数（翻译默认 300） |
| `LLM_GEN_LANGS` | 源语言（默认 `zh-Hans`，即只生成简体，其余语言靠翻译） |
| `LLM_PER_TASK` | 每（垂类, 任务, 语言）的目标条数；不设则按配额自动折算 |

> 凭据、网关与模型名只走环境变量：仓库内不写死任何网关地址、模型名或 API key，`.gitignore` 已排除
> `.env`、`*.log`、`*_state.json`、`seeds*/`、`artifacts/`。

## 流水线顺序

```bash
# 1. 生成源语言（简体中文）语料
python scripts/api_gen.py --workers 4 --batch 16

# 2. 逐条对齐翻译（每种目标语言共享并发）
python scripts/translate_seeds.py --targets zh-Hant --workers 2 --batch 8
python scripts/translate_seeds.py --targets en     --workers 2 --batch 8

# 3. （可选）重试被跳过的硬样本
python scripts/retry_skipped.py --lang en --workers 4 --batch 8

# 4. 去重（精确 + MinHash 近似）
python scripts/dedupe_audit.py            # 只报告
python scripts/dedupe_audit.py --apply    # 删除重复（保留首次出现）

# 5. 装配为 data/ 分片
python scripts/assemble.py --write

# 6. 繁简/语言纯净度校正与质量审计
python scripts/fix_variants.py
python scripts/quality_gate.py

# 7. 逐垂类逐语言核验配额
python scripts/check_quota.py --dir data

# 8. 冻结（在 nano-jev 仓库内执行，生成 MANIFEST.json）
nano-jev data freeze --dir data --note "..."
```

## 脚本职责

| 脚本 | 作用 |
|---|---|
| `api_gen.py` | 调用 LLM 按任务规格生成 `state` + `labels`，写入 `artifacts/llm/seeds_api/`；按 (垂类, 任务, 语言) 记录进度，可断点续跑 |
| `translate_seeds.py` | 以简体为源逐条翻译成繁体/英文，校验占位符、数字（含万/亿换算）、繁简与英文纯净度；游标与跳过记录持久化 |
| `retry_skipped.py` | 只重试 `skipped` 里的记录，避免整组重跑 |
| `aligned_translate.py` | 子代理协作翻译用的 `next` / `merge` 小工具，校验口径与 `translate_seeds.py` 一致 |
| `dedupe_audit.py` | 精确指纹 + MinHash（5-gram / 8 哈希 / 4 段）+ Jaccard ≥0.80 近似去重 |
| `assemble.py` | 按内容指纹装配，输出 `train/`、`train/posttrain/`、`test/`、`dev/` 每垂类每语言一个文件 |
| `fix_variants.py` | 简繁变体校正，报告英文中混入的中文 |
| `quality_gate.py` | schema / 标签域 / 长度 / 语言纯度 / 重复 / 标签均衡审计 |
| `verify.py` | 装配结果结构校验，输出 `verify_report.json` |
| `progress.py` | 统计每语言已译量、逐组缺口与预计剩余时间 |
| `check_quota.py` | 逐垂类逐语言的 train/test/post 配额核验（本数据集口径） |
| `export_specs.py` | 导出每个垂类的任务规格到 `specs/`，供生成与审计引用 |

## 目录约定

- 数据在仓库根目录：`train/`、`train/posttrain/`、`test/`、`dev/`，文件名 `00-llm.<category>.<lang>.jsonl`。
- 生成中间产物（种子、日志、状态文件）默认写在 `artifacts/llm/`，该目录不应提交。
