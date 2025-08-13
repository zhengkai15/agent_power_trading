# 电力交易收益最大化 Agent

本项目旨在基于历史真实电价、日前电价和气象资料，构建一个 Agent 实现电力交易收益最大化。方案包含电价预测模型和强化学习交易策略，并实现训练、推理和评估流程。

## 项目结构

```
agent_power_trading/
├── .DS_Store
├── .gitignore
├── GEMINI.md
├── README.md
├── requirements.txt
├── aexp/                 # 实验数据
├── data/                 # 数据目录
│   ├── raw/              # 原始数据
│   └── processed/        # 处理后的数据
├── discuss/              # 讨论和评审文档
├── docs/                 # 正式文档
├── logs/                 # 日志文件
├── models/               # 训练好的模型
├── script/               # Shell 脚本
│   ├── install_dependencies.sh
│   ├── run_data_processing.sh
│   ├── run_inference.sh
│   └── run_training.sh
└── src/                  # Python 源代码
    ├── agent/            # 交易 Agent 相关代码
    ├── data_processing/  # 数据处理模块
    ├── inference/        # 推理模块
    ├── model/            # 模型定义
    ├── training/         # 训练模块
    └── utils/            # 工具函数
```

## 功能概述

- **数据处理**: 从原始数据中提取、清洗、对齐和聚合电价与气象数据，生成模型训练所需的特征和标签。
- **电价预测模型**: 基于深度学习技术，利用历史数据和气象特征预测未来电价。
- **交易 Agent**: 基于强化学习构建交易策略，在模拟环境中学习如何最大化交易收益。
- **训练流程**: 包含电价预测模型的监督学习训练和交易 Agent 的强化学习训练。
- **推理流程**: 独立的推理模块，用于实时电价预测和交易决策。
- **评估**: 提供电价预测准确性和交易策略收益的评估指标。

## 快速开始

### 1. 安装依赖

```bash
bash script/install_dependencies.sh
```

### 2. 数据处理

```bash
bash script/run_data_processing.sh
```

### 3. 模型训练

```bash
bash script/run_training.sh
```

### 4. 运行推理

```bash
bash script/run_inference.sh
```

## 文档

- **解决方案计划**: [docs/solution_plan.md](docs/solution_plan.md)
- **方案评价与讨论**: [discuss/solution_discussion.md](discuss/solution_discussion.md)

## 贡献

欢迎贡献！请遵循 `GEMINI.md` 中的代码规范和贡献指南。