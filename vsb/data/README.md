# VSB 电力线路局部放电检测：9 特征单表数据

整理时间：2026-11-16 11:24 CST。

## 来源与任务

竞赛：[VSB Power Line Fault Detection](https://www.kaggle.com/competitions/vsb-power-line-fault-detection)。
特征来源：[mark4h 第一名公开 notebook](https://www.kaggle.com/code/mark4h/vsb-1st-place-solution)，版本 3、公开运行 ID 12111254。
方案说明：[第一名 writeup](https://www.kaggle.com/competitions/vsb-power-line-fault-detection/writeups/mark4h-overview-of-1st-place-solution)。

原始信号来自实际线路测量。公开有标签部分为 8,712 条相位信号，属于 2,904 次三相测量，每条波形有 800,000 个采样点。原始相位阳性数为 525。

本表沿用作者方案，**一行对应一次三相测量**，不是一个波形采样点。用同一次测量的三相信号构造 9 个特征；若任一相位原始 target=1，测量标签 target=1，否则为 0。共 2,904 行，其中 194 行为阳性。
正类表示有局部放电／线路绝缘故障隐患，不能理解为所有类型的电力设备故障。

## 特征复现方式

复用作者公开输出的 train_peaks.pkl（1,355,697 条峰值记录），从中按作者非简化版 process() 聚合 9 个特征。作者源 notebook、提取代码和下载清单保存于实验目录。
原始竞赛波形下载需要 Kaggle 登录，本次没有下载原始 train.parquet，也没有重新从波形识别峰值。特征聚合结果已与作者 notebook 中的两个聚合函数逐值核对，检查通过。
峰值文件没有 target；聚合只关联 signal_id 与 id_measurement，不读取标签。原始标签从单独的 metadata_train.csv 获得，该公开副本来自 RoyMachineLearning/VSB-Power-Line-Fault-Detection，SHA256 在 metadata.json 中。

作者会移除 ratio_next > 0.33333 且 height > 50 的峰值。没有峰值的分组及不能计算的统计量保持 NaN，未擅自填零。所有三相聚合均在划分前按单次测量独立计算，不跨测量使用其他样本信息。

## 字段

| 字段 | 含义 |
| --- | --- |
| `peak_count_Q13` | 相位对齐后第 1、3 象限的峰值总数。 |
| `abs_small_dist_to_min_mean_Q02` | 第 0、2 象限中，峰值与附近反极性极值之间采样点距离绝对值的均值。 |
| `height_mean_Q02` | 第 0、2 象限的峰高均值。 |
| `height_std_Q02` | 第 0、2 象限的峰高样本标准差。 |
| `sawtooth_rmse_mean_Q02` | 第 0、2 象限中，作者定义的锯齿模板误差特征均值；保留作者字段名及原计算定义。 |
| `peak_count_Q02` | 相位对齐后第 0、2 象限的峰值总数。 |
| `ratio_next_mean_Q02` | 第 0、2 象限中，后一采样点幅值绝对值与峰高之比的均值。 |
| `ratio_prev_mean_Q02` | 第 0、2 象限中，前一采样点幅值绝对值与峰高之比的均值。 |
| `peak_count_total` | 三相全部象限的峰值总数。 |
| `row_id` | 原始 id_measurement，仅用于追踪，不能作为模型特征。 |
| `target` | 测量级标签：任一相位阳性为 1，三相均阴性为 0，不能作为模型特征。 |

**输入严格使用 metadata.json 的 feature_columns，只包含以上 9 个数值特征。** 排除 signal_id、id_measurement、row_id、phase 和 target。

## 文件与划分

- all_measurements.csv：完整 2,904 行测量级单表。
- train.csv / val.csv / test.csv 及同名 Parquet：固定划分，可直接读取。
- original_phase_labels.csv：8,712 行原始相位元数据与真实标签。
- metadata.json：来源 SHA256、特征白名单、标签定义、划分行号和 Parquet 校验值。

seed=42，按测量标签分层，70%／15%／15%。同一次三相测量不会跨划分。

| 划分 | 测量数 | 阳性数 |
| --- | ---: | ---: |
| 训练 | 2,032 | 136 |
| 验证 | 436 | 29 |
| 测试 | 436 | 29 |

验证与测试推理使用完整 2,032 条训练测量作固定上下文。只在验证集最大化 MCC 选择阈值，测试标签只用于最终评分；同时保留固定阈值 0.5 的参照。全参数微调使用完整训练表，说明见 ../../finetuning/README.md。
主要指标为测量级 MCC，辅助为 ROC-AUC、PR-AUC（AP）、F1、召回率与精确率。

**本地内部留出集结果不能直接与 Kaggle 私榜 MCC=0.71899 比较。** 原竞赛测试部分有 20,337 条相位信号，其真实标签不公开，本次未用它计算指标。
测试集中只有 29 个阳性测量，单 seed 结果可能受划分影响。公开 VSB 数据及冠军特征是否进入基础模型预训练语料尚未核实。

本数据包包含已提取的单表特征、标签、划分、说明及上一级 raw/ 中的公开峰值文件，**不包含原始波形，也不把 80 万个采样点当成独立分类样本**。
原始数据许可与竞赛条款以 Kaggle 来源页面为准。作者 notebook 页面标注代码为 Apache 2.0，本包保留其作者与来源，不为原始竞赛数据另行授予许可。

## Python 读取

```python
import json
import pandas as pd
from pathlib import Path
root = Path('.')
cols = json.loads((root / 'metadata.json').read_text())['feature_columns']
train = pd.read_csv(root / 'train.csv')
test = pd.read_csv(root / 'test.csv')
X_train, y_train = train[cols], train['target']
X_test, y_test = test[cols], test['target']
```
