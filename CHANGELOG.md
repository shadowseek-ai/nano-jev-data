# CHANGELOG

本仓库的数据版本与 `MANIFEST.json` 的 `dataset_version` 保持一致。

## 2026.11.06.8（后训练混合分片 + CI 门禁）

- **新增 `train/posttrain/mix/`**：后训练集此前只有单语分片，导致无法按「混合口径」做后训练；
  现与 `train/mix/` 同构，6 组合 × 13 垂类，每个组合 **3,966** 条，共 23,796 条。
- **新增 CI**：`.github/workflows/dataset-gate.yml`（push/PR 自动跑结构、配额、对抗子集、
  MANIFEST 与 README 一致性校验，外带全历史密钥扫描）；
  `.github/workflows/release.yml`（手动删除并重建 `nano-jev-data` release）。
- **新增门禁脚本**：`audit_derived.py`（派生/对抗子集全部验收项）、`scan_secrets.py`（全历史扫描）、
  `release_notes.py`（Release 说明由 MANIFEST 生成）、`dedupe_files.py`、`drop_leaks.py`、`move_cases.py`。
- **移除无效文件**：删除旧实现 `scripts/gen_nearmiss.py`、`scripts/build_nearmiss.py`——
  它们生成的是「选项级干扰项当作反例」的副本，已被审计判定为无效数据，避免误用回归。
- README 的「每组合条数」拆成三张表：混合矩阵、语言对抗矩阵、按语言的对抗子集；
  对角线标「—」（同语言不适用），修正此前把单语记录与 A/C 类混算进对角格的问题。
- 冻结版本 `2026.11.06.8`。

## 2026.11.06.5（按对抗需求重做 adv / nearmiss，新增 dev/mix）

对应 `dataset-2026.11.06.3-audit.md` 与 `dataset-adversarial-requirements.md`。

### 修掉的问题

| 项 | 修前 | 修后 |
|---|---|---|
| `adv` 与 `mix` 键交集 | 236,736（完全重合） | **0** |
| `adv` 内容 | 是 `mix` 的副本 | 三类真改写，内容指纹与 `mix` 无交集 |
| `nearmiss` 同 case 标签 | 19,500 条全部相同 | **0**（全部翻转） |
| 内容指纹重复（全库） | 1,570 | **0** |
| train/test `case_id` 交叉 | 1（另含 posttrain/dev 共 14） | **0** |
| `dev/mix` | 缺失 | **9,312 条**（6 组合 × 13 垂类） |

### 对抗子集（按 `adv_type` 区分）

| 类型 | 文件 | 规模 | 说明 |
|---|---|---|---|
| A 干扰项 | `10-advdist.*` | 19,500（train 15,600 + test 3,900） | 注入 1–3 句错误选项旁注，事实与标签不变；近义混淆 40%，位置三档均分 |
| B 语言对抗 | `20-adv-lang.*` | 42,271（train 31,189 + test 7,796 + dev 3,286） | 判定线索换成另一语言原文 |
| C 极性对照 | `12-advpol.*` | 24,804（**8,268** 个完整组） | C1 只否结论 / C2 只改事实 / C3 同向改，`pair_id` 串联 |
| 近似反例 | `train/nearmiss/` | 11,439 | 改一个数字跨阈值，标签翻转 |

新增字段：`adv_type`、`derived_from`、`pair_id`、`pair_role`、`adv_position`、`adv_injected`、`adv_similarity`、`adv_near_synonym`。

### 其它

- `MANIFEST.json` 增加 `adversarial` 段（按 split × `adv_type` × 语言/垂类给条数）。
- 新增门禁脚本 `scripts/audit_derived.py`，一条命令校验上述全部验收项。
- README 增加「对抗与派生子集」表与目录说明。
- 语言对抗（B）与干扰项（A）建议进训练；极性对照（C）按文档作为诊断集，先不直接进训练。

## 2026.11.06.x（近似反例子集 §3.2）

- 新增 `train/nearmiss/`：13 垂类 × 500 案例 × 3 语言 = **19,500 条**记录。
- 每条与原记录**同案同标签**，仅在 choice 题 `criteria` 增加一个 `near_miss` 干扰项：
  语气与正确选项相近，但案情里有明确相反证据；生成时要求模型给出该证据原文，
  校验「证据必须是 state 原文子串」通过才收录。
- 正确选项与 `labels` 完全不变，标签不变性不受影响。

## 2026.11.06.2（混合/对抗分片落地，配额达标）

- 三语单语记录：**154,983** 条（train 118,377 / post 11,898 / test 20,013 / dev 3,978 附近），
  每条 3 题，共 464,949 题。
- 每垂类每语言记录数全部达标：`train 3001+`、`test 501+`、`post 300+`、`dev 102+`。
- 派生分片：`train/mix/`、`test/mix/` 各 78 个文件（13 垂类 × 6 组合，39,456 / 6,667 条每组合），
  `train/adv/`、`test/adv/` 同构；混合 46,130 条/组合、对抗 46,127 条/组合（train+test+dev 合计）。
- 模板占位符残留 **0**（含派生分片）；`verify_manifest` 通过，跨 split 重复 **0**。
- `MANIFEST.json` 增加 `per_combo`，README 增加「各语言题目数」「每组合条数」两张表。

### 丢弃与补齐

- 生成种子中约 2.5%（482/19,018）因翻译校验不过（数字缺失/占位符丢失）被排除，
  对应缺口由后续增量生成补齐；交付数据只包含三语齐全的案情。

## 2026.11.05.x（案情中心改造）

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
