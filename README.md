# nano-jev-data

[nano-jev](https://github.com/shadowseek-ai/nano-jev) 的三语数据集（LLM 生成），用于评测与训练。

| 项目 | 内容 |
|---|---|
| 版本 | `2026.11.04.3` |
| 更新时间 | 2026-11-04 18:46 (+08:00) |
| 语言 | `zh-Hans` 简体中文、`zh-Hant` 繁体中文、`en` 英文 |
| 规模 | **114,189 条记录 / 342,567 题**，13 个垂类，156 个文件，约 168 MB |
| 许可 | MIT |

## 数据量

| split | 目录 | 记录 | 题 |
|---|---|---:|---:|
| 预训练 | `train/` | 98,803 | 296,409 |
| 后训练 | `train/posttrain/` | 4,781 | 14,343 |
| 测试 | `test/` | 9,905 | 29,715 |
| 开发 | `dev/` | 700 | 2,100 |

每条记录 = 1 个 `state` + 3 道题（`noul` / `choice` / `score` 各一），三题共用同一 `state`。
语言分布（记录）：`zh-Hans` 35,707、`zh-Hant` 41,890、`en` 36,592。

## 目录

```
train/               00-llm.<category>.<lang>.jsonl   预训练
train/posttrain/     00-llm.<category>.<lang>.jsonl   后训练
test/                00-llm.<category>.<lang>.jsonl   测试
dev/                 00-llm.<category>.<lang>.jsonl   开发
```

13 个垂类：`agent_tool_governance`、`content_safety_ops`、`embodied_game_ops`、
`energy_utilities`、`finance_insurance`、`healthcare_logistics`、`hr_recruiting`、
`life_sciences_lab`、`manufacturing_quality`、`public_sector_services`、`retail_ecommerce`、
`supply_chain_logistics`、`telecom_network_ops`。

## 校验

```bash
nano-jev data verify --dir .              # 校验 MANIFEST.json 与文件哈希
python scripts/check_quota.py --dir .     # 逐垂类逐语言配额
```

## 说明

- 数据全部由 LLM 生成，不含真实个人信息；场景中的编号均为占位符（如 `{ticket}`）。
- 任何 `.jsonl` 改动后需 `nano-jev data freeze` 重新生成 `MANIFEST.json`，否则 `verify` 失败。
- 生成、翻译与校验脚本见 [scripts/README.md](scripts/README.md)，凭据只从环境变量读取，仓库内不保存任何密钥。
