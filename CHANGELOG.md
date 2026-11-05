# CHANGELOG

本仓库的数据版本与 `MANIFEST.json` 的 `dataset_version` 保持一致。

## 2026.11.05.x（案情中心改造 + 混合/对抗分片）

### 口径变化

- **案情中心**：简体中文（`zh-Hans`）为唯一真源生成案情，`labels` 只算一次；
  繁体与英文由对齐翻译得到，翻译不改判定。
- **新增字段**：每条记录带 `case_id`（同案情跨语言一致）、`question_lang`（问题语言）、
  `source_lang`（案情源语言，新数据固定 `zh-Hans`）。
- **配额改为按记录数**：每垂类每语言
  `train ≥ 3000` / `test ≥ 500` / `train/posttrain ≥ 300` / `dev ≥ 100` 条记录
  （每条记录 = 1 个 `state` + 3 道该语言原文题）。
- **占位符清零**：`{date}` `{ticket}` `{case_id}` 等模板占位符按 `case_id` 确定性渲染为
  真实值（日期、单号、编号、站点码等），三种语言渲染结果一致；交付数据残留为 0。
- **混合分片**：`train/mix/`、`test/mix/` 下按 `state 语言 × 问题语言` 派生 6 种组合，
  零 LLM 成本。
- **语言对抗子集**：`train/adv/`、`test/adv/` 下把 state 中带数字的关键句替换成另一语言的
  同句，迫使模型跨语言对齐语义。
- **MANIFEST**：新增 `per_combo`（每个 split 下各语言组合的条数）。
- **README**：新增「各语言题目数」「每组合条数」两张表，垂类改用中文名。

### 生成与翻译

- 源语言生成：`deepseek-flash`，2 并发 × 4 条/请求。
- 对齐翻译：`zh-Hant` 用 OpenCC `s2twp` 确定性转换；`en` 优先用 Edge 翻译 API
  （免费，2 并发 × 4 条/请求，全局限流 1.2 s/请求），个别样本回落到 LLM 修复。
- 凭据只从 `artifacts/llm/.env.local`（gitignored）或环境变量读取，仓库内不保存密钥。

## 2026.11.04.3

- 三语对齐数据集：每垂类每语言 `train ≥ 2000` 条 / `test ≥ 500` 题 / `post ≥ 200` 题。
- MIT 许可，数据全部由 LLM 生成，不含真实个人信息。
