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

## 敏感信息

- 凭据、网关与模型名只从环境变量读取：不写死、不落盘、不打印，缺失时报错只提示变量名。
- 错误日志只记录异常类型与消息（截断 120 字符），不输出请求头、key 或完整响应体。
- `.gitignore` 已排除 `.env*`、`secrets/`、`*.key`、`*.pem`、`*.log`、`*_state.json`、`*.cursor.json`、`seeds*/`、`artifacts/`。

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

## 预训练集补齐到 3000（2026-11-05 口径）

当前仓库的预训练集按“每垂类每语言 >= 3000 条记录”补齐，使用独立脚本，
不依赖 nano-jev 的 `src/`。硬性并发口径：

| 阶段 | 并发 | 每请求条数 |
|---|---:|---:|
| 简体中文源语言生成 | 2 | 4 |
| 对齐翻译（zh-Hant + en 共用） | 2 | 4 |

```bash
# 生成源语言（可断点续跑）
python scripts/topup_pretrain.py --generate --work artifacts/llm/topup

# 同时增量翻译（繁中走 OpenCC 精确转换，英文走 LLM，2 并发 × 4 条）
python scripts/topup_translate.py --work artifacts/llm/topup --workers 2 --batch 4

# 生成完成后写入标记，翻译进程收尾退出
New-Item artifacts/llm/topup/generation.done

# 合并进 train/，并核验 3000 配额
python scripts/topup_pretrain.py --merge --work artifacts/llm/topup

# 审计种子（标签域 / 语言 / 占位符 / 数字 / 三语对齐）
python scripts/topup_audit.py artifacts/llm/topup

# 重新冻结 MANIFEST.json（只扫描 train/test/dev）
python scripts/freeze_manifest.py
python scripts/check_quota.py --dir .
```

`topup_pretrain.py` 只从环境变量读取 `LLM_API_BASE` / `LLM_API_KEY` / `LLM_MODEL`，
密钥不落盘；生成、翻译、合并都按内容指纹去重，并保留 `artifacts/llm/splits/` 作为
可恢复的中间产物（该目录不入库）。

## 案情中心三 split 流水线（当前在用）

```bash
# 1. 按“文件缺口”生成源语言（含 dev），2 并发 × 4 条
python scripts/topup_splits.py --generate --from-files --rounds 8 --work artifacts/llm/splits

# 2. 繁中本地转换（OpenCC，零成本）
python scripts/convert_hant.py

# 3. 英文走 Edge 翻译（免费，2 并发 × 4 条，全局限流）
python scripts/topup_splits.py --translate --work artifacts/llm/splits

# 4. 补译被校验拒绝的样本（Edge 放宽 -> LLM 兜底）
python scripts/repair_en.py --work artifacts/llm/splits

# 5. 合并进 train/test/dev/posttrain（三语同进同出，写入 case_id/question_lang/source_lang）
python scripts/topup_splits.py --merge --work artifacts/llm/splits

# 6. 渲染残留占位符 + 去重 + 简体纯度
python scripts/render_placeholders.py
python scripts/clean_splits.py

# 7. 派生混合分片与语言对抗子集（零 LLM 成本）
python scripts/build_mixed.py --adversarial

# 8. 校验、冻结、更新 README
python scripts/topup_verify.py
python scripts/check_quota.py --dir .
python scripts/freeze_manifest.py
python scripts/update_readme_stats.py
```

旧数据改造（一次性）：`rewrite_legacy.py` 渲染占位符并补 `case_id`；
`rebuild_legacy_aligned.py` 以 zh-Hans 为源重建 zh-Hant/en，使三语条数严格一致。


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
| `topup_pretrain.py` | 按 3000 配额生成源语言、翻译、合并与计数报告 |
| `topup_translate.py` | 与生成并行的增量翻译（英文 LLM、繁中 OpenCC） |
| `topup_audit.py` | 审计 topup 种子的标签域、语言、占位符/数字与三语对齐 |
| `freeze_manifest.py` | 只扫描 `train/test/dev` 重新生成 `MANIFEST.json`（含 `per_combo`） |
| `topup_splits.py` | 三 split（train/test/post/dev）统一补齐：生成、翻译、合并，三语同进同出 |
| `convert_hant.py` | 把种子里的简体源转成繁体（OpenCC `s2twp`），不占用翻译 API |
| `edge_translate.py` | Edge 翻译客户端（免费），全局限流 + 连续失败冷却 |
| `repair_en.py` | 补译被校验拒绝的英文：Edge 放宽校验 → LLM 批量兜底 |
| `repair_en_final.py` | 逐条 LLM 修复（带缺失 token 回灌），作为最后一档 |
| `build_mixed.py` | 由 `case_id` 对齐派生混合分片（`train` / `post` / `test` / `dev` 各 6 组合） |
| `build_adversarial.py` | 类型 B 语言对抗：把判定句换成另一语言原文，`adv_type="lang"` |
| `build_distractor.py` | 类型 A 干扰项：注入错误选项旁注，事实与标签不变，`adv_type="distractor"` |
| `build_polarity.py` | 类型 C 极性对照：C1 只否结论 / C2 只改事实 / C3 同向改，`adv_type="polarity"` |
| `gen_nearmiss_flip.py` | 用 LLM 找「改一个数字就翻盘」的位置并给出翻转后标签（带规则校验） |
| `build_nearmiss_flip.py` | 把翻盘样本落成近似反例子集（`case_id` 加 `-nm`，标签翻转） |
| `audit_derived.py` | **派生/对抗子集门禁**：键交集、规模、标签关系、指纹唯一、case_id 交叉、占位符 |
| `dedupe_files.py` | 文件内去重（按 `case_id`+`lang`+`style`+`pair_role`） |
| `drop_leaks.py` | 删除跨 split 重复的案情（整案三语一起删） |
| `move_cases.py` | 把案情从 train 整体搬到 test，补缺口且不产生泄漏 |
| `scan_secrets.py` | 扫描**全 git 历史对象**，确认无密钥值/令牌/私钥入库 |
| `release_notes.py` | 从 `MANIFEST.json` 生成 Release 说明，避免手写数字漂移 |
| `render_placeholders.py` | 把 `{date}`/`{ticket}` 等模板占位符按 `case_id` 确定性渲染 |
| `clean_splits.py` | 跨 split 去重（整案三语一起删）+ 简体纯度规整 |
| `rewrite_legacy.py` | 旧数据一次性改造：渲染占位符、补 `case_id`/`question_lang`/`source_lang` |
| `rebuild_legacy_aligned.py` | 以 zh-Hans 为源重建 zh-Hant/en，使旧数据三语条数严格一致 |
| `case_meta.py` | 占位符渲染与 `case_id` 规则的公共库 |
| `update_readme_stats.py` | 刷新 README 的版本/规模/语言×split/组合/对抗统计表 |

## CI（GitHub Actions）

| 工作流 | 触发 | 作用 |
|---|---|---|
| `.github/workflows/dataset-gate.yml` | push `master`、PR、手动 | 跑 `topup_verify` / `check_quota` / `audit_derived` / `freeze_manifest --dry-run`，并校验 README 统计表与数据一致；另有全历史密钥扫描 job |
| `.github/workflows/release.yml` | 手动 | 用当前 master 删除并重建 `nano-jev-data` release，说明由 `release_notes.py` 生成 |

CI 依赖见 `requirements-dev.txt`（`openai`、`opencc-python-reimplemented`）。CI 不需要任何密钥：
`topup_pretrain.client()` 是延迟创建的，只在校验脚本不调用 LLM 时不会触发。

## 目录约定

- 数据在仓库根目录：`train/`、`train/posttrain/`、`test/`、`dev/`，文件名 `00-llm.<category>.<lang>.jsonl`。
- 生成中间产物（种子、日志、状态文件）默认写在 `artifacts/llm/`，该目录不应提交。
