# 数据集目录

本目录是**冻结的 LLM 生成数据集**：文件内容、sha256、版本与每个大类的题量都记录在
[MANIFEST.json](MANIFEST.json)。校验/门禁默认在统一训练镜像内执行（`make gate`），
宿主机命令只用于快速回路。

数据来源与旧版 `00-synth.*` 合成模板不同：本版全部由 LLM 生成（简体中文为源语言，
繁体中文与英文由逐条对齐翻译产出），因此也包含更长的真实风格输入与干扰句。

## 当前版本

- `dataset_version`：见 `MANIFEST.json`（当前 `2026.11.04.3`）
- 规模：**train 103,584 条记录 / 310,752 题；test 9,905 条 / 29,715 题；dev 700 条 / 2,100 题**
- 覆盖：**13 个行业大类 × 3 个任务 × 3 种原语（noul / choice / score）× 3 种语言**
  （`zh-Hans` 简体中文 / `zh-Hant` 繁体中文 / `en` 英文）

### 分项规模（记录数）

| split | zh-Hans | zh-Hant | en | 每垂类最小值 |
|---|---|---|---|---|
| 预训练 train | 30,865 | 36,240 | 31,698 | 2,206 / 2,214 / 2,147 |
| 测试 test | 3,139 | 3,652 | 3,114 | 223 / 203 / 198 条（669 / 609 / 594 题） |
| 后训练 post | 1,490 | 1,728 | 1,563 | 99 / 98 / 98 条（297 / 294 / 294 题） |
| 开发 dev | 213 | 270 | 217 | 8 / 16 / 12 |

### 配额口径（本版验收标准）

- 每个垂类每种语言 **train ≥ 2,000 条记录**（≈6,000 题）
- 每个垂类每种语言 **test ≥ 500 题**（≈167 条记录）
- 每个垂类每种语言 **post ≥ 200 题**（≈67 条记录）
- 三种语言总量 **dev ≥ 100 条**

当前 39 个（13 垂类 × 3 语言）组合全部达标，由 `check_quota.py` 逐组核验。

## 目录

```
data/
├── MANIFEST.json                    # 版本、sha256、题量、配额（由 data freeze 生成）
├── README.md                        # 本文件
├── train/                           # 预训练：00-llm.<category>.<lang>.jsonl
│   └── posttrain/                   # 后训练：00-llm.<category>.<lang>.jsonl
├── test/                            # 测试：00-llm.<category>.<lang>.jsonl
└── dev/                             # 开发：00-llm.<category>.<lang>.jsonl
```

- 配置与 CLI 使用目录路径 `data/train`、`data/dev`、`data/test`（不是单个 `data/train.jsonl`）。
- 文件名格式 `00-llm.<category>.<lang>.jsonl`，一个垂类一种语言一个文件；`train/posttrain/`
  下的同名文件属于后训练集。
- 每条记录 = 1 个 `state` + 3 个问题（1 noul + 1 choice + 1 score），三题共用同一 `state`。

## 常用命令

```bash
nano-jev data stats   --dir data                 # 查看规模（不写文件）
nano-jev data verify  --dir data                 # 校验 MANIFEST 与文件哈希
nano-jev data freeze  --note "..."               # 刷新 MANIFEST 与版本号
python check_quota.py --dir data                 # 逐垂类逐语言核验 train/test/post 配额
python progress.py                               # 查看对齐翻译进度与缺口
```

> `nano-jev data verify --dir data --quota` 使用的是旧口径（每大类训练题 ≥2,000、测试题 ≥100，
> 按**垂类**而非按语言统计），与本版配额不同，会失败；本版请用 `check_quota.py`。

## 去重

- 生成阶段：按归一化 `state` 做精确指纹去重（`blake2b`，同一垂类同一语言内）。
- 装配阶段：按 `state + questions` 内容指纹跨 split 去重。
- 审计阶段：`dedupe_audit.py` 做精确去重 + MinHash 分桶（5-gram，8 哈希 4 段）+ Jaccard ≥0.80 近似去重。
- 本版复检结果：**114,189 条记录中重复数为 0**。

## 注意

- 数据全部由 LLM 生成，不含真实个人信息；医疗、政务等场景中的编号均为占位符（如 `{ticket}`）。
- 改动任何 `.jsonl` 后必须 `data freeze`，否则 `data verify`/CI 会失败（有意设计）。
- 数据文件与 `MANIFEST.json` 必须一起提交。
