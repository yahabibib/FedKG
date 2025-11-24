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
├── fl_core.py          # 🤝 联邦核心：定义 Client (本地训练) 和 Server (聚合策略)
├── precompute.py       # ⚡️ 预计算：SBERT 嵌入生成 (带缓存)、邻接矩阵构建
├── data_loader.py      # 📚 数据加载：解析 Triples, IDs, Attributes (.pkl/txt)
├── evaluate.py         # 📊 评估模块：计算 Hits@K, MRR，包含错误案例分析
├── environment.yml     # 📦 环境依赖文件
└── models/             # 🧠 模型定义
    ├── decoupled.py    # [核心] 解耦模型 (Private GCN + Shared MLP)
    ├── gcn.py          # 基础 GCN 层
    ├── projection.py   # 简单的投影模型
    └── transe.py       # TransE 模型实现