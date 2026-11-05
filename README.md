# nano-jev-data

[nano-jev](https://github.com/shadowseek-ai/nano-jev) 的三语数据集（LLM 生成），用于评测与训练。

| 项目 | 内容 |
|---|---|
| 版本 | `2026.11.05.2` |
| 更新时间 | 2026-11-05 12:45 (+08:00) |
| 语言 | `zh-Hans` 简体中文、`zh-Hant` 繁体中文、`en` 英文 |
| 规模 | **114,246 条记录 / 342,738 题**，13 个垂类，156 个文件，约 168 MB |
| 许可 | MIT |

## 数据量

| split | 目录 | 记录 | 题 |
|---|---|---:|---:|
| 预训练 | `train/` | 98,803 | 296,409 |
| 后训练 | `train/posttrain/` | 4,838 | 14,514 |
| 测试 | `test/` | 9,905 | 29,715 |
| 开发 | `dev/` | 700 | 2,100 |

每条记录 = 1 个 `state` + 3 道题（`noul` / `choice` / `score` 各一），三题共用同一 `state`，题目文本按该语言原文给出。
语言分布（记录）：`zh-Hans` 35,726、`zh-Hant` 41,909、`en` 36,611。

## 各语言题目数

<!-- lang-stats:start -->
| 语言 | 预训练 | 后训练 | 测试 | 开发 | 合计 |
|---|---:|---:|---:|---:|---:|
| `zh-Hans` | 92,595 | 4,527 | 9,417 | 639 | 107,178 |
| `zh-Hant` | 108,720 | 5,241 | 10,956 | 810 | 125,727 |
| `en` | 95,094 | 4,746 | 9,342 | 651 | 109,833 |
| **合计** | **296,409** | **14,514** | **29,715** | **2,100** | **342,738** |
<!-- lang-stats:end -->

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

- 数据全部由 LLM 生成，不含真实个人信息；场景中的编号均为占位符（如 `{ticket}`）。
- 任何 `.jsonl` 改动后需 `nano-jev data freeze` 重新生成 `MANIFEST.json`，否则 `verify` 失败。
- 生成、翻译与校验脚本见 [scripts/README.md](scripts/README.md)，凭据只从环境变量读取，仓库内不保存任何密钥。
