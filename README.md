# 电力单表数据包：SGCC 与 VSB

本数据包包含 SGCC 原始 **1,034 个日用电量字段**版，以及 VSB 作者 **9 特征**版；两个数据集各有独立目录，共享直接推理与微调方法说明。

## 从 GitHub 获取完整数据

数据文件使用 Git LFS 保存。先安装 [Git LFS](https://git-lfs.com/)，然后执行：

```bash
git lfs install
git clone git@github.com:zhenwenbo/ele_sig.git
cd ele_sig
git lfs pull
python verify_package.py
```

最后一条命令核对本地真实数据文件的 SHA256。仅下载不含 LFS 对象的源码 ZIP，可能只得到数据指针文件；请使用上述方式获取完整数据。

## 数据与两种使用方式

SGCC 集中放在 [`SGCC/`](SGCC/README.md)，包含原始数据、固定协议、划分、SQLite 数据库及重建／评估脚本。VSB 的作者 **9 特征版**放在 [`vsb/`](vsb/README.md)，包含原始公开峰值文件、标签、单表及重建脚本。

本 README 并列保留 **直接推理（冻结参数）**和 **全参数微调**两种方式，均用于 SGCC 与 VSB。微调协议见 [`finetuning/README.md`](finetuning/README.md)。数据包仅包含数据、协议、代码和方法说明；模型权重、预测、日志与成绩输出到本目录之外。

微调只保留一套设置：**完整训练集训练一轮，每步训练上下文为 1,024 条**。查询样本每条恰好参与一次监督损失；训练前与一轮结束后评估完整验证集，SGCC 按 ROC-AUC、VSB 按 MCC 选择权重。

## 方式一：直接推理（冻结参数）

加载预训练模型后保持所有权重不变，通过带标签的训练上下文预测验证／测试样本。这是上下文推理（ICL），**不计算训练损失，不反向传播，不执行优化器更新**。

1. SGCC 读取 `SGCC/data/context_2048.parquet` 中固定的 2,048 条带标签训练用户；VSB 读取 `vsb/data/train.parquet` 中全部 2,032 条带标签训练测量。保持各自保存的上下文顺序。
2. 按所用模型的推理接口输入带标签上下文和查询特征，关闭梯度与参数更新，获得正类概率。如果模型接口包含准备上下文缓存的过程，该过程也不属于参数微调。
3. 先预测完整验证集。SGCC 的 F1／窃电召回率阈值按验证 F1 最大选择；VSB 按验证 MCC 最大选择阈值。若比较多个预先确定的推理配置，SGCC 按验证 ROC-AUC、VSB 按验证 MCC 选择配置；单一固定配置不需要选训练检查点。
4. 固定配置与分类阈值后预测完整测试集。验证和测试样本及标签均不加入上下文，测试指标不参与配置或阈值选择。

直接推理的具体模型、权重与设备设置由实际实验记录。SGCC 使用固定上下文中的 2,048 条训练标签，其余训练用户保留在训练表中；VSB 的全部训练标签均作为上下文。

## 方式二：全参数微调

从同一预训练模型出发，用完整训练池的真实标签更新全部模型参数，然后用固定训练上下文进行验证／测试推理。

1. SGCC 使用完整 29,659 条训练用户，VSB 使用完整 2,032 条训练测量。训练集打乱后无放回取查询批，每批最多 128 条；另从其余训练样本中抽取 **1,024 条训练上下文**，以查询真实标签计算交叉熵并更新全部参数。遍历完完整训练集一次即结束，保留最后不足 128 条的批次。
2. 默认 AdamW 学习率 `1e-6`，只训练 **1 epoch**；按查询批大小 128 计算，SGCC 为 232 个优化步，VSB 为 16 个优化步。训练前及这一轮结束后评估完整验证集；SGCC 按验证 ROC-AUC、VSB 按验证 MCC 选择权重，不重新搜索超参数。
3. 在初始预训练权重与单轮训练结束权重之间按验证主指标选择，同分保留初始权重。固定所选检查点及其验证阈值后评估测试集一次。训练上下文的 1,024 条与验证／测试的固定推理上下文分别定义，后者仍与直接推理一致。详细协议见 [`finetuning/README.md`](finetuning/README.md)。

| 对比项 | 直接推理 | 全参数微调 |
| --- | --- | --- |
| 预训练模型参数 | 全部冻结 | 全部参与梯度更新 |
| SGCC 用于参数训练的行数 | 0 | 完整训练池 29,659 行 |
| VSB 用于参数训练的行数 | 0 | 完整训练池 2,032 行 |
| 训练预算 | 0 | 完整训练集 1 轮；SGCC 232 步，VSB 16 步 |
| 训练上下文 | 无梯度训练阶段 | 每步 1,024 条 |
| 验证／测试上下文 | SGCC 固定 2,048；VSB 全部训练 2,032 | 与直接推理相同 |
| 模型权重选择 | 使用原始预训练权重 | 按验证主指标比较初始与单轮结束权重 |

两种方式共用相同特征白名单、数据划分、推理上下文、正类定义及指标口径。SGCC 均必测 **ROC-AUC、PR-AUC（AP）、F1、窃电召回率**；VSB 以 **MCC** 为主，另报 ROC-AUC、AP、F1 和阳性召回率。两者的分类阈值分别由各自验证概率选择，不必相同；测试标签只用于最终评分。

## 数据重建与核对

两数据集统一重建／核对入口：

```bash
cd /raid5/wenbozhen/ele_sig
PYTHON=/raid5/wenbozhen/relational_foundation_models/.venv/bin/python
$PYTHON rebuild_all.py --verify-only
$PYTHON rebuild_all.py --overwrite --recompute-splits
```

第二条命令会覆盖生成的数据表并重建 SGCC 数据库；保留原始文件，重新核对固定划分。VSB 重建还会与作者的九特征聚合实现逐值对照。

`$PYTHON verify_package.py` 可按 `PACKAGE_MANIFEST.json` 核对包内全部文件的 SHA256；该清单不包含模型结果。通用微调协议、单轮查询调度与核对工具位于 `finetuning/`，具体模型训练接口由实际实验代码提供。

## 目录

```text
ele_sig/
├── README.md
├── requirements.txt
├── rebuild_all.py                 # 统一重建／核对两套数据
├── verify_package.py
├── PACKAGE_MANIFEST.json
├── SGCC/
│   ├── README.md
│   ├── rebuild_database.py         # SGCC 数据库重建
│   ├── evaluate.py                 # SGCC 四指标评估
│   ├── raw/                       # 原始用电 CSV
│   ├── reference/                 # 固定协议、划分与上下文行号
│   └── data/                      # 划分单表、2,048 条上下文、SQLite
├── vsb/
│   ├── README.md
│   ├── prepare_vsb.py
│   ├── author_solution.py
│   ├── evaluation_protocol.json
│   ├── raw/                       # 作者公开峰值与原始相位标签
│   └── data/                      # 九特征单表与固定划分
└── finetuning/
    ├── README.md
    ├── training_schedule.py
    ├── audit_run.py
    └── protocol.json
```

## 数据集详细说明

- [SGCC：来源、1,034 个字段、划分、2,048 条上下文、数据库重建与四指标评估](SGCC/README.md)。
- [VSB：作者九特征版、三相测量标签、固定划分及重建](vsb/README.md)。
- [全参数微调：完整训练集一轮、训练上下文 1,024 条](finetuning/README.md)。
