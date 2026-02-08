# FedAnchor: 基于语义锚点与解耦架构的联邦知识图谱对齐框架

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)
[![Status](https://img.shields.io/badge/Status-Active-success)](https://github.com/)

**FedAnchor** 是一个隐私保护下的联邦知识图谱对齐（Federated Knowledge Graph Alignment）框架。它利用预训练语言模型（SBERT）构建语义锚点，通过 **解耦图神经网络（Decoupled GCN）** 架构，在不共享原始图结构和实体数据的前提下，实现了跨图实体的自动对齐。

---

## 数据集来源 (Data Source)
本项目默认配置使用 DBP15K (ZH-EN) 数据集进行实验。

数据来源: DBP15K (Chinese-English)

下载地址: [HuggingFace - MatchBench DBP15K](https://huggingface.co/datasets/matchbench/dbp15k-zh-en/tree/main)

---

## 核心特性 (Key Features)

* **隐私保护联邦学习**: 数据不出本地，仅交换公共映射层的模型参数，保留私有图结构特征.
* **语义锚点机制 (SBERT)**: 利用 SBERT 生成的文本嵌入作为冻结的“绝对坐标系”，引导异构图结构特征的对齐，无需人工标注种子对齐.
* **解耦模型架构 (Decoupled Architecture)**:
    * **Private (私有层)**: GCN 编码器，适应本地特有的图拓扑结构，**不参与聚合**。
    * **Shared (公共层)**: MLP 投影层，学习统一的语义映射规则，**参与联邦聚合**。
* **迭代自训练 (Iterative Self-Training)**: 基于互为最近邻 (RNN) 生成高置信度伪标签，动态更新锚点，逐步扩大对齐规模.
* **安全困难负采样 (Safe Hard Mining)**: 自动挖掘难区分的负样本，提升模型判别能力.

---

## 📂 项目结构

```text
FedAnchor/
├── main.py             # 🚀 项目入口：主循环、伪标签生成、流程控制
├── config.py           # ⚙️ 配置文件：参数设定 (DBP15K/Demo)、模型选择、超参数
├── environment.yml     # 📦 环境依赖文件
│
├── src/                # 📦 核心模块包
│   ├── data/           # 📚 数据处理
│   │   ├── dataset.py       # 数据集抽象类 (AlignmentTaskData)
│   │   └── loader.py        # 数据加载工具函数
│   │
│   ├── federation/     # 🤝 联邦学习核心
│   │   ├── server.py        # Server：全局模型聚合策略
│   │   ├── client_sbert.py  # ClientSBERT：语义微调客户端
│   │   ├── client_structure.py  # ClientStructure：结构训练客户端 (带 Adaptive Fusion)
│   │   └── strategy.py       # PseudoLabelGenerator：伪标签生成策略
│   │
│   ├── models/         # 🧠 模型架构
│   │   ├── decoupled.py     # [核心] 解耦模型 (Private + Shared 双层)
│   │   ├── encoders/        # 图编码器库
│   │   │   ├── gcn.py           # 标准 GCN 编码器
│   │   │   ├── gat.py           # GAT 编码器
│   │   │   ├── rgat.py          # Relational GAT (多关系)
│   │   │   └── sage.py          # GraphSAGE 编码器
│   │   └── projectors/       # 投影层库
│   │       └── mlp.py            # MLP 投影层 (可学习的门控)
│   │
│   └── utils/          # ⚙️ 工具函数
│       ├── device_manager.py  # GPU/MPS/CPU 设备管理与显存优化
│       ├── graph.py           # 图构建：邻接矩阵、边特征处理
│       ├── metrics.py         # 评估指标：Hits@K, MRR, eval_alignment
│       ├── logger.py          # 日志记录：实验结果持久化
│       └── tuning.py          # 超参数搜索：Alpha 融合权重自适应调整
│
├── configs/            # 📋 Hydra 配置文件 (YAML)
│   └── config.yaml        # 主配置文件 (数据路径、超参数、实验设置)
│
├── checkpoints/        # 💾 保存的模型权重
│
├── data/               # 📁 数据集目录
│   ├── demo/               # 小规模演示数据
│   └── dbp15k/             # DBP15K 官方数据集 (zh_en)
│
├── figures/            # 📈 生成的可视化图表
│
├── logs/               # 📝 运行日志
│
└── models_old/         # 🗂️ 旧版本模型 (备份)
    ├── decoupled.py
    ├── gcn.py
    ├── projection.py
    └── transe.py
```

### 关键模块说明

| 模块 | 功能 | 状态 |
|------|------|------|
| **data.dataset** | 数据集抽象，统一 ID/Triple/Alignment 三元组 | ✅ 生产 |
| **federation.client_sbert** | SBERT 微调：Description + Polish 混合训练 | ✅ 生产 |
| **federation.client_structure** | 结构训练：解耦 GCN + 自适应融合门控 | ✅ 生产 |
| **federation.strategy** | 伪标签生成：互为最近邻 (RNN) + 阈值课程学习 | ✅ 生产 |
| **models.encoders.rgat** | 多关系 GAT：处理异构图关系类型 | ✅ 新增 |
| **utils.device_manager** | 设备管理：GPU/MPS 自动切换 + 显存优化 | ✅ 新增 |
| **utils.tuning** | 融合权重搜索：自动找最优 Alpha | ✅ 新增 |
```