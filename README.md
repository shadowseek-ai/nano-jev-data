# nano-jev-data

[nano-jev](https://github.com/shadowseek-ai/nano-jev) 的三语数据集（LLM 生成），用于评测与训练。

| 项目 | 内容 |
|---|---|
| 版本 | `2026.11.06.6` |
| 更新时间 | 2026-11-06 12:43 (+08:00) |
| 语言 | `zh-Hans` 简体中文、`zh-Hant` 繁体中文、`en` 英文 |
| 规模 | **154,932 条记录 / 464,796 题**（三语合计），13 个垂类，819 个文件，约 930 MB |
| 许可 | MIT |

## 数据量

| split | 目录 | 记录（三语合计） | 题目（三语合计） |
|---|---|---:|---:|
| 预训练 | `train/` | 118,371 | 355,113 |
| 后训练 | `train/posttrain/` | 11,898 | 35,694 |
| 测试 | `test/` | 20,007 | 60,021 |
| 开发 | `dev/` | 4,656 | 13,968 |
| 派生分片 | `train/mix`、`train/adv`、`test/mix`、`test/adv` | 553,542 | 1,660,626 |

每条记录 = 1 个 `state` + 3 道题（`noul` / `choice` / `score` 各一），三题共用同一 `state`，题目文本按该语言原文给出。
单语记录每语言 51,661 条；派生分片由同一批案情按语言组合复用而来，不额外生成。

## 各语言题目数

<!-- lang-stats:start -->
| 语言 | 预训练 | 后训练 | 测试 | 开发 | 合计 |
|---|---:|---:|---:|---:|---:|
| `zh-Hans` | 118,371 | 11,898 | 20,007 | 4,656 | 154,932 |
| `zh-Hant` | 118,371 | 11,898 | 20,007 | 4,656 | 154,932 |
| `en` | 118,371 | 11,898 | 20,007 | 4,656 | 154,932 |
| **合计** | **355,113** | **35,694** | **60,021** | **13,968** | **464,796** |
<!-- lang-stats:end -->

## 每组合条数（state 语言 × 问题语言）

同一案情在三种语言下共用 `case_id`，因此可零成本派生跨语言组合样本；
`train/mix/`、`test/mix/` 存放混合分片，`train/adv/`、`test/adv/` 存放语言对抗子集。
表中每格为 **混合条数 / 对抗条数**（含 train、test、dev 三个 split 的合计）。

<!-- combo-stats:start -->
| state 语言 \ 问题语言 | `zh-Hans` | `zh-Hant` | `en` |
|---|---:|---:|---:|
| `zh-Hans` | 47,678 / 14,768 | 47,678 / 6,962 | 47,678 / 7,466 |
| `zh-Hant` | 47,678 / 6,814 | 47,678 / 14,768 | 47,678 / 7,452 |
| `en` | 47,678 / 6,735 | 47,678 / 6,842 | 47,678 / 14,768 |
<!-- combo-stats:end -->

## 对抗与派生子集

所有子集都由**同一批案情**派生，`case_id` 可追溯（`derived_from` 指回原案），
`labels` 是否随事实变化严格区分，详见 `dataset-adversarial-requirements.md` 的类型 A/B/C。

| 目录 / 文件 | 类型 | `style` | `adv_type` | `labels` | 条数 |
|---|---|---|---|---:|---:|
| `train/mix/`、`test/mix/`、`dev/mix/` | 跨语言混合 | `mix` | — | 不变 | 286,068 |
| `train/adv/10-advdist.*`、`test/adv/10-advdist.*` | A 干扰项 | `adv-distractor` | `distractor` | 不变 | 19,500 |
| `train/adv/20-adv-lang.*`、`test/adv/20-adv-lang.*`、`dev/adv/` | B 语言对抗 | `lang-adversarial` | `lang` | 不变 | 42,271 |
| `train/adv/12-advpol.*`、`test/adv/12-advpol.*` | C 极性对照 | `adv-polarity` | `polarity` | C1 不变 / C2・C3 翻转 | 24,804 |
| `train/nearmiss/` | 近似反例 | `near-miss` | — | **翻转** | 11,439 |

### 类型 A：干扰项（`10-advdist`）

在 state 里注入 1–3 句「提到**错误选项**关键词」的旁注，案情事实与 `labels` 一律不变：

- 近义混淆占比 **40%**（干扰项取与正确选项用词最相似的错误选项）；
- 注入位置 句首 / 句中 / 句末 各约 1/3，用于检验位置不敏感；
- `case_id` = `<原案>-dist<NNNN>`，另有 `derived_from`、`adv_position`、`adv_injected`、`adv_similarity`、`adv_near_synonym`。

### 类型 B：语言对抗（`20-adv-lang`）

同一案情、同一 `labels`，把 state 里**带数字的判定句**替换成另一种语言的同句原文，
判定线索只出现在另一种语言里。`case_id` = `<原案>-adv`，每组合 397–400 条。

### 类型 C：极性对照（`12-advpol`）

对每个已验证的「改一个数字跨过阈值 → 标签翻转」的案情产出三态，`pair_id` 串起同一组：

| `pair_role` | 事实 | 结论句 | `labels` |
|---|---|---|---|
| `C1` | 原样 | 追加**否定**结论 | 与原案相同 |
| `C2` | 跨阈值 | 仍维持旧结论 | 翻转 |
| `C3` | 跨阈值 | 与事实一致的结论 | 翻转 |

共 **8,268** 个完整组（每垂类每语言 ≥120 组），按文档口径为诊断集，不建议直接进训练。

### 近似反例（`train/nearmiss/`）

每条与原案**同案不同标签**：把案情里一个关键数字改掉，正好跨过阈值，三题标签随之翻转。
`case_id` = `<原案>-nm`，`derived_from` 指回原案；每垂类 ≥500 条记录。

## 目录

```
train/               00-llm.<category>.<lang>.jsonl   预训练
train/posttrain/     00-llm.<category>.<lang>.jsonl   后训练
test/                00-llm.<category>.<lang>.jsonl   测试
dev/                 00-llm.<category>.<lang>.jsonl   开发
train/mix/           10-mix.<category>.<stateLang>_<questionLang>.jsonl      混合（6 组合）
test/mix/            10-mix.<category>.<stateLang>_<questionLang>.jsonl      混合（6 组合）
dev/mix/             10-mix.<category>.<stateLang>_<questionLang>.jsonl      混合（6 组合，早停用）
train/adv/           10-advdist.<category>.<lang>.jsonl                      类型 A 干扰项
                     20-adv-lang.<category>.<stateLang>_<questionLang>.jsonl  类型 B 语言对抗
                     12-advpol.<category>.<lang>.jsonl                       类型 C 极性对照
test/adv/            同上三类的 test 同构分片
dev/adv/             20-adv-lang.*（早停用语言对抗）
train/nearmiss/      30-nearmiss.<category>.<lang>.jsonl                     近似反例（标签翻转）
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
