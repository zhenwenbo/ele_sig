# SGCC 原始 1,034 个日用电量字段单表数据

SGCC 的原始文件、固定协议、生成数据与专用脚本集中保存在本目录。

## 两种使用方式

1. **直接推理**：冻结预训练参数，使用 `data/context_2048.parquet` 中固定的 2,048 条训练用户作为带标签上下文，不计算梯度、不执行优化器更新；验证 F1 选分类阈值后评估测试集。
2. **全参数微调**：完整 29,659 条训练用户无放回训练一轮，每步训练上下文为 1,024 条、查询最多 128 条，共 232 个优化步；训练前与一轮结束后按完整验证集 ROC-AUC 选择权重，再用同一份固定 2,048 条训练用户上下文进行验证／测试推理。分类阈值仍按所选检查点的验证 F1 确定。

两种方式共用固定划分、特征白名单与四个必测指标。方法对比见[根目录 README](../README.md)，通用微调协议见 [finetuning/README.md](../finetuning/README.md)。

## 数据来源与任务

原作者发布地址：[henryRDlab/ElectricityTheftDetection](https://github.com/henryRDlab/ElectricityTheftDetection)。作者说明数据由国家电网提供，为真实用户用电记录。

- 一行代表一名用户，共 **42,372** 名用户。
- 原始 CSV 有 1,036 列：用户编号 `CONS_NO`、标签 `FLAG` 和 1,034 个日用电量字段。
- `FLAG=0`：正常，38,757 人；`FLAG=1`：窃电，3,615 人，正类约占 8.53%。
- 处理后标签名为 `target`，数值与对应原始用户的 `FLAG` 完全一致，**窃电为正类 1**。
- 任务是用户级回溯窃电分类，数据没有用户窃电开始时间，不能据此解释为未来窃电预测或电力设备故障检测。

引用：Zibin Zheng, Yatao Yang, Xiangdong Niu, Hong-Ning Dai, Yuren Zhou. *Wide and Deep Convolutional Neural Networks for Electricity-Theft Detection to Secure Smart Grids.* IEEE Transactions on Industrial Informatics, 14(4):1606–1615, 2018. DOI: [10.1109/TII.2017.2785963](https://doi.org/10.1109/TII.2017.2785963)。原作者说明保存在 `raw/AUTHOR_README.md`。

## 目录

```text
SGCC/
├── README.md
├── rebuild_database.py
├── evaluate.py
├── raw/
│   ├── data.csv
│   └── AUTHOR_README.md
├── reference/
│   ├── protocol.json
│   ├── feature_columns.json
│   ├── split_indices.json
│   ├── context_indices.json
│   └── source_download_manifest.json
└── data/
    ├── sgcc.sqlite3
    ├── train.parquet / train.csv
    ├── val.parquet / val.csv
    ├── test.parquet / test.csv
    ├── context_2048.parquet / context_2048.csv
    ├── customer_splits.csv
    ├── context_indices.json
    ├── feature_columns.json
    ├── manifest.json
    └── validation_report.json
```

以下相对路径均以本 `SGCC/` 目录为基准。

`raw/data.csv` 的 SHA256 固定为：

```text
99f8fd315626b1f729a9a03a97cb52ed097ab4d43e5771e21554c9e0c369b9b7
```

## 1,034 个输入字段的处理方式

原始日期列为字符串排序，准备时按真实日历日期升序排列，重命名为 `day_YYYY_MM_DD`。完整白名单在 `reference/feature_columns.json`，生成后也复制到 `data/feature_columns.json`。

保留原始数值与缺失值，不在全数据上插值、补零或标准化。缺失值约占日用电量单元格的 25.64%，其中 5 名用户的全部读数缺失；全部保留。SQLite 中缺失值对应 `NULL`。

**模型只能读取白名单中的 1,034 个字段。** `target` 是标签；`row_id`、`CONS_NO`／`customer_id` 是追踪标识；`split`、`feature_group`、`context_order` 是数据管理信息，均不能作为输入。原始行顺序与标签有关，把行号当输入会产生泄漏。

## 划分方式

采用按标签分层、同时按完全相同的原始用电模式分组的划分，约为 **70%／15%／15%**。

### 分组依据

对每名用户按日期排序后的全部 1,034 个原始用电量及缺失模式，用：

```python
groups = pd.util.hash_pandas_object(x, index=False).to_numpy()
```

生成分组标识。原始数据有 878 条额外重复用电模式记录；不同用户若用电模式完全相同，则放在同一划分。训练、验证、测试之间既无相同用户，也无完全相同的原始用电模式。

### 确切算法

1. `StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=42)` 生成十折。
2. 将第 0、1、2 折的测试位置合并并排序，作为约 30% 的留出池；其余用户为训练集。
3. 在留出池上使用 `StratifiedGroupKFold(n_splits=2, shuffle=True, random_state=43)`，取首次 `split` 返回的两个位置数组，分别作为验证集和测试集。
4. 固定的原始行号保存在 `reference/split_indices.json`。`row_id` 为原始 CSV 中从 0 开始、不计表头的行位置。

| 划分 | 用户总数 | 正常 | 窃电 |
| --- | ---: | ---: | ---: |
| 训练 | **29,659** | 27,129 | 2,530 |
| 验证 | **6,356** | 5,814 | 542 |
| 测试 | **6,357** | 5,814 | 543 |

分组约束使人数与精确的 70%／15%／15% 略有差异。这是按用户及重复模式划分，**不是按日期切分**。验证与测试完整保留原始类别分布。

## 2,048 条上下文的含义与抽取

上下文是从训练集抽出的 **2,048 名带标签用户**，用于上下文推理。它不是查询 batch size，也不是训练集总人数。

抽取规则精确为：

```python
context_ids, _ = train_test_split(
    train_row_ids,
    train_size=2048,
    random_state=42,
    stratify=y[train_row_ids],
)
```

- 从 29,659 名训练用户中按类别分层、无放回抽取。
- 组成：**正常 1,873 人，窃电 175 人**，总计 **2,048 人**。
- 全部来自训练集，验证和测试用户及其标签不进入上下文。
- 对比实验应共用这同一份上下文，不为不同模型或不同查询重新抽样。
- **保持 `context_indices.json` 中 `row_ids` 的返回顺序**，不再按用户编号或原始行号排序。
- 其余 27,611 名训练用户仍保存在训练表中；固定上下文推理不使用这些额外训练标签。
- **全量微调会使用完整 29,659 人的训练集更新参数。** 2,048 条限制仅用于验证／测试时的固定推理上下文，不限制参与梯度训练的用户人数。
- `data/context_2048.parquet` 与 `data/context_2048.csv` 已按固定上下文顺序物化，可直接读取。

上下文输入形状为 `(2048, 1034)`，上下文标签形状为 `(2048,)`。

## SQLite 数据库

数据库为 `data/sgcc.sqlite3`，只有一张保存样本的基础表 `samples`；训练、验证、测试及上下文是该表上的视图。

| 表／视图 | 内容与顺序 |
| --- | --- |
| `samples` | 42,372 名用户、1,034 个原始读数字段及追踪信息 |
| `train` | 29,659 行，按原始 `row_id` 排序 |
| `val` | 6,356 行，按原始 `row_id` 排序 |
| `test` | 6,357 行，按原始 `row_id` 排序 |
| `context_2048` | 2,048 行，按冻结的 `context_order` 排序 |

`samples` 的额外字段为 `row_id`、`customer_id`、`target`、`split`、`context_order`、`feature_group`。`feature_group` 为避免 SQLite 有符号整数溢出，保存为十进制文本；非上下文用户的 `context_order` 为 `NULL`。

## 重建数据库

建议 Python 3.12，依赖版本写在上一级 [`requirements.txt`](../requirements.txt)；本机已有对应环境。重建仅依赖本目录下的原始 CSV 和固定协议文件。

```bash
cd /raid5/wenbozhen/ele_sig/SGCC
PYTHON=/raid5/wenbozhen/relational_foundation_models/.venv/bin/python

# 首次重建：用固定行号与上下文，适合稳定复用
$PYTHON rebuild_database.py

# 已存在数据时重建，并重新执行原划分／上下文抽样算法逐项核对
$PYTHON rebuild_database.py --overwrite --recompute-splits

# 单独验证当前数据库及所有导出表
$PYTHON rebuild_database.py --verify-only

# 在另一目录重建，不覆盖当前数据
$PYTHON rebuild_database.py --output-dir /tmp/sgcc_rebuild --recompute-splits
```

默认使用冻结行号，可避免库升级造成随机划分漂移；`--recompute-splits` 会重新执行上述算法，并要求三份划分及上下文的**成员和顺序**与冻结版本完全一致。

脚本先在临时目录构建并核对全部数据，再发布到输出目录。它检查原始 CSV SHA256、原始标签、全部 42,372 行和 1,034 个字段的值及缺失位置、跨划分分组互斥、上下文来源和顺序、SQLite 完整性与各视图人数。检查记录在 `data/validation_report.json`，文件校验值在 `data/manifest.json`。

## 必须测量的四个指标

**SGCC 的验证主指标固定为 ROC-AUC，越大越好。** 如果比较不同配置、检查点或上下文方案，以固定验证集上的 ROC-AUC 选择；不能按测试指标选择。验证集成员由前述固定划分产生，不按指标挑选。

| 展示名称 | 代码字段 | 定义与输入 |
| --- | --- | --- |
| **ROC-AUC** | `ROC_AUC` | 使用窃电连续概率 `p_theft` 评估排序能力 |
| **PR-AUC（AP）** | `PR_AUC_AP` | 使用 `sklearn.metrics.average_precision_score`；这里是 AP，不是梯形积分 |
| **F1** | `F1` | 窃电为正类，`2 × precision × recall / (precision + recall)` |
| **窃电召回率** | `recall_theft` | 窃电正类的 `TP / (TP + FN)` |

ROC-AUC 和 AP 使用概率，不先二值化。**配置选择与分类阈值选择是两件事：ROC-AUC 不依赖分类阈值，不能用它选择阈值。** F1 和窃电召回率的阈值规则沿用独立约定：默认只在验证集最大化窃电 F1，然后固定该阈值测完整测试集。若报告固定阈值 0.5 的对照，必须标明阈值。不能按测试指标调阈值。

`evaluate.py` 接收外部生成的验证／测试概率文件，要求每个文件包含 `row_id,p_theft` 两列；概率必须在 `[0,1]`，每个划分必须完整覆盖且行号不重复。输入文件可为 CSV 或 Parquet。真实标签始终从本目录的固定数据读取。

```bash
# 配置选择阶段：只读取验证集，用 ROC-AUC 选择配置，不读取测试标签
$PYTHON evaluate.py --validation-only \
  --validation-predictions /path/to/val_probabilities.csv

# 验证集选阈值，输出四个测试指标
$PYTHON evaluate.py \
  --validation-predictions /path/to/val_probabilities.csv \
  --test-predictions /path/to/test_probabilities.csv

# 固定阈值 0.5 的对照
$PYTHON evaluate.py \
  --test-predictions /path/to/test_probabilities.csv \
  --threshold-mode fixed --fixed-threshold 0.5
```

## Python 读取

```python
import json
from pathlib import Path
import pandas as pd

root = Path('/raid5/wenbozhen/ele_sig/SGCC')
features = json.loads((root / 'data/feature_columns.json').read_text())
context = pd.read_parquet(root / 'data/context_2048.parquet')
val = pd.read_parquet(root / 'data/val.parquet')
test = pd.read_parquet(root / 'data/test.parquet')

X_context, y_context = context[features], context['target']
X_val, y_val = val[features], val['target']
X_test, y_test = test[features], test['target']
assert X_context.shape == (2048, 1034)
```

通过 SQLite 读取相同顺序的上下文：

```python
import sqlite3
with sqlite3.connect(root / 'data/sgcc.sqlite3') as connection:
    context = pd.read_sql_query('SELECT * FROM context_2048', connection)
X_context, y_context = context[features], context['target']
```
