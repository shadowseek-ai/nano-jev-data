# nano-jev-data

[nano-jev](https://github.com/shadowseek-ai/nano-jev) 的三语数据集（LLM 生成），用于评测与训练。

| 项目 | 内容 |
|---|---|
| 版本 | `2026.11.06.2` |
| 更新时间 | 2026-11-06 00:22 (+08:00) |
| 语言 | `zh-Hans` 简体中文、`zh-Hant` 繁体中文、`en` 英文 |
| 规模 | **154,983 条记录 / 464,949 题**，13 个垂类，312 个文件，约 1081 MB |
| 许可 | MIT |

## 数据量

| split | 目录 | 记录 | 题 |
|---|---|---:|---:|
| 预训练 | `train/` | 118,377 | 355,131 |
| 后训练 | `train/posttrain/` | 11,898 | 35,694 |
| 测试 | `test/` | 20,013 | 60,039 |
| 开发 | `dev/` | 4,695 | 14,085 |

每条记录 = 1 个 `state` + 3 道题（`noul` / `choice` / `score` 各一），三题共用同一 `state`，题目文本按该语言原文给出。
语言分布（记录）：`zh-Hans` 51,661、`zh-Hant` 51,661、`en` 51,661。

## 各语言题目数

<!-- lang-stats:start -->
| 语言 | 预训练 | 后训练 | 测试 | 开发 | 合计 |
|---|---:|---:|---:|---:|---:|
| `zh-Hans` | 118,377 | 11,898 | 20,013 | 4,695 | 154,983 |
| `zh-Hant` | 118,377 | 11,898 | 20,013 | 4,695 | 154,983 |
| `en` | 118,377 | 11,898 | 20,013 | 4,695 | 154,983 |
| **合计** | **355,131** | **35,694** | **60,039** | **14,085** | **464,949** |
<!-- lang-stats:end -->

## 每组合条数（state 语言 × 问题语言）

同一案情在三种语言下共用 `case_id`，因此可零成本派生跨语言组合样本；
`train/mix/`、`test/mix/` 存放混合分片，`train/adv/`、`test/adv/` 存放语言对抗子集。
表中每格为 **混合条数 / 对抗条数**（含 train、test、dev 三个 split 的合计）。

<!-- combo-stats:start -->
| state 语言 \ 问题语言 | `zh-Hans` | `zh-Hant` | `en` |
|---|---:|---:|---:|
| `zh-Hans` | 47,695 / 0 | 46,130 / 46,127 | 46,130 / 46,127 |
| `zh-Hant` | 46,130 / 46,127 | 47,695 / 0 | 46,130 / 46,127 |
| `en` | 46,130 / 46,127 | 46,130 / 46,127 | 47,695 / 0 |
<!-- combo-stats:end -->

## 目录

```
train/               00-llm.<category>.<lang>.jsonl   预训练
train/posttrain/     00-llm.<category>.<lang>.jsonl   后训练
test/                00-llm.<category>.<lang>.jsonl   测试
dev/                 00-llm.<category>.<lang>.jsonl   开发
```

13 个垂类：数据治理与 Agent 运维（`agent_tool_governance`）、内容安全与合规（`content_safety_ops`）、游戏与具身智能（`embodied_game_ops`）、能源与公用事业（`energy_utilities`）、金融与保险（`finance_insurance`）、医疗后勤（探索性）（`healthcare_logistics`）、人力资源与招聘（`hr_recruiting`）、生命科学（实验自动化）（`life_sciences_lab`）、制造业与工业质检（`manufacturing_quality`）、公共部门与政务服务（`public_sector_services`）、零售与电商（`retail_ecommerce`）、供应链与物流（`supply_chain_logistics`）、电信与网络运维（`telecom_network_ops`）。

## 校验

```bash
nano-jev data verify --dir .              # 校验 MANIFEST.json 与文件哈希
python scripts/check_quota.py --dir .     # 逐垂类逐语言配额
```

## 说明

- 数据全部由 LLM 生成，不含真实个人信息；场景中的编号已渲染为虚构值（单号、设备码、站点码等）。
- **案情中心**：同一 `case_id` 在三种语言下是同一个案子，`labels` 只由简体中文案情算一次，
  翻译不改判定；`question_lang` 是问题语言，`source_lang` 是案情源语言。
- 混合样本无需新生成：`train/mix/`、`test/mix/` 由 `state 语言 × 问题语言` 派生 6 种组合。
- 任何 `.jsonl` 改动后需重新生成 `MANIFEST.json`：
  `python scripts/freeze_manifest.py`（只扫描 `train/`、`test/`、`dev/`）。
- 生成、翻译与校验脚本见 [scripts/README.md](scripts/README.md)，版本变更见 [CHANGELOG.md](CHANGELOG.md)；
  凭据只从环境变量或 `artifacts/llm/.env.local` 读取，仓库内不保存任何密钥。
