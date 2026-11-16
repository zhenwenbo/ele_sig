# VSB：作者 9 特征单表版

一行是一次三相线路测量。`target=1` 表示三个相位中至少一相的原始标签为局部放电阳性，属于线路绝缘隐患检测。共 2,904 次测量、194 次阳性；不是 SGCC 窃电检测。

来源与九个字段的具体含义见 [`data/README.md`](data/README.md)，源文件哈希和固定样本行号见 [`data/metadata.json`](data/metadata.json)。特征白名单为该文件的 `feature_columns`。

| 划分 | 测量数 | 阳性数 |
| --- | ---: | ---: |
| 训练 | 2,032 | 136 |
| 验证 | 436 | 29 |
| 测试 | 436 | 29 |

先按一次三相测量聚合，随后按标签分层划分约 70%／15%／15%，两次 `train_test_split` 均为种子 42。同次测量的三相不会落入不同划分。不能把 `row_id`、`signal_id`、`id_measurement`、`phase` 或标签作为特征。

验证主指标为 **MCC**：每个检查点在完整验证集上最大化 MCC 选阈值，再按验证 MCC 选检查点；平分保留更早检查点。最终阈值与检查点冻结后才评估测试集，另报 ROC-AUC、PR-AUC（AP）、F1 和阳性召回率。推理上下文固定为全部 2,032 条训练测量，保持训练表顺序。

## 重建

```bash
cd /raid5/wenbozhen/ele_sig/vsb
/raid5/wenbozhen/relational_foundation_models/.venv/bin/python prepare_vsb.py
```

脚本从 `raw/train_peaks.pkl` 及 `raw/metadata_train.csv` 重建 `data/`，并逐值对照所附作者 `author_solution.py` 的聚合函数。数据包复用了作者公开的峰值文件，**未包含原始波形，也未从波形重新提峰**。重建覆盖生成的单表文件，原始文件不变。

## 两种使用方式

1. **直接推理**：冻结预训练模型参数，用全部 2,032 条训练测量作为固定带标签上下文，先预测验证集并按验证 MCC 选阈值，再预测测试集；没有梯度更新，也不选择训练检查点。
2. **全参数微调**：完整 2,032 条训练测量无放回训练 **一轮**，每步训练上下文 **1,024 条**、查询最多 128 条，共 16 个优化步；训练前及一轮结束后按完整验证集 MCC 选择权重，再用相同的 2,032 条固定训练测量上下文推理。

两种方式共用相同划分、九特征白名单、真实标签与指标定义。完整对比见[根目录 README](../README.md)，通用微调协议见 [`../finetuning/README.md`](../finetuning/README.md)。本目录不存放模型运行结果。
