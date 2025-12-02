# FedAnchor: 基于语义锚点与解耦架构的联邦知识图谱对齐框架

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)
[![Status](https://img.shields.io/badge/Status-Active-success)](https://github.com/)

**FedAnchor** 是一个隐私保护下的联邦知识图谱对齐（Federated Knowledge Graph Alignment）框架。它结合了 **大语言模型（LLM）本地增强 与 解耦联邦图神经网络（Decoupled GCN）**，在不共享原始图结构和实体数据的前提下，实现了跨图实体的自动对齐。

---

## 数据集来源 (Data Source)
本项目默认配置使用 DBP15K (ZH-EN) 数据集进行实验。

数据来源: DBP15K (Chinese-English)

下载地址: [HuggingFace - MatchBench DBP15K](https://huggingface.co/datasets/matchbench/dbp15k-zh-en/tree/main)

---

## 核心特性 (Key Features)

1.  **两阶段增强架构 (Two-Stage Enhancement)**:

      * **Stage 1: 语义注入**: 利用本地部署的 LLM (Qwen-2.5) 将结构化三元组转化为自然语言描述，并通过对比学习（Contrastive Learning）微调 SBERT，构建强大的语义地基。
      * **Stage 2: 结构对齐**: 使用微调后的 SBERT 初始化联邦训练，利用 GCN 捕获拓扑结构一致性。

2.  **隐私保护联邦学习 (Privacy-Preserving FL)**:

      * **解耦架构 (Decoupled Model)**:
          * **Private (私有层)**: GCN 编码器，保留本地特有的图拓扑特征，**参数不上传**。
          * **Shared (公共层)**: MLP 投影层，学习统一的语义映射规则，**参与联邦聚合**。
      * **数据不出域**: 原始三元组、文本描述、实体名称均保留在客户端本地。

3.  **迭代自训练与融合挖掘 (Iterative Fusion Mining)**:

      * **双模融合**: 在伪标签挖掘阶段，结合 **GCN（结构）** 与 **SBERT（语义）** 的双重特征，利用结构信息纠正语义偏差（如消歧、别名识别）。
      * **课程学习**: 动态阈值策略（Curriculum Learning），随着迭代轮次增加逐步提高置信度要求，防止噪音传播。

---

## 项目结构

```text
FedAnchor/
├── config/                 # (可选) 配置文件存放
├── data/                   # 数据集存放 (DBP15K)
├── output/                 # 输出目录 (模型 Checkpoints, Logs, Figures)
├── scripts/                # 执行脚本
│   ├── run_stage1.py       # 第一阶段：LLM 润色 + SBERT 微调
│   ├── run_stage2.py       # 第二阶段：联邦训练主程序 (含消融实验模式)
│   ├── plot_results.py     # 结果可视化绘图
│   └── test.py             # 单元测试与 Demo
├── src/                    # 核心源码包
│   ├── core/               # 联邦核心 (Client, Server)
│   ├── data/               # 数据处理 (Loader, Preprocessor)
│   ├── llm/                # LLM 相关 (Polisher, Finetuner)
│   ├── models/             # 模型定义 (Decoupled, GCN)
│   └── utils/              # 工具类 (Config, Logger, Metrics)
├── environment.yml         # 环境依赖
└── README.md               # 项目说明
```

-----

## 🚀 快速开始 (Quick Start)

### 1\. 环境准备

推荐使用 Conda 管理环境：

```bash
conda env create -f environment.yml
conda activate fl-anchor-mac
```

### 2\. 数据准备

请确保数据集 (DBP15K ZH-EN) 位于 `data/dbp15k/zh_en/` 目录下。该目录应包含 `ent_ids_1`, `triples_1` 等标准文件。

### 3\. 第一阶段：本地语义增强 (Local Semantic Enhancement)

此步骤在客户端本地运行，利用 LLM 润色数据并微调 SBERT。

```bash
# 自动加载 Qwen 模型 -> 生成文本 -> 微调 SBERT
python scripts/run_stage1.py
```

  * **产出**: `output/fine_tuned_models/exp4_finetuned` (结构感知 SBERT 模型)。

### 4\. 第二阶段：联邦迭代对齐 (Federated Iterative Alignment)

启动联邦训练循环。脚本支持多种模式，方便进行消融实验。

**模式 A: 完整形态 (Full Model) - 推荐**
包含 Stage 1 增强、解耦架构、自训练循环（5 Iterations）。

```bash
python scripts/run_stage2.py --mode full
```

**模式 B: 消融实验 - 无 LLM (No LLM)**
使用原始 BERT 模型，验证语义增强的有效性。

```bash
python scripts/run_stage2.py --mode no_llm
```

**模式 C: 消融实验 - 无挖掘 (No Mining)**
仅训练一轮（Iter 1），不进行伪标签挖掘，验证自训练机制的有效性。

```bash
python scripts/run_stage2.py --mode no_mining
```

-----

## ⚙️ 配置说明 (Configuration)

所有核心参数均在 `src/utils/config.py` 中集中管理。

| 参数名 | 默认值 | 说明 |
| :--- | :--- | :--- |
| `model_arch` | `decoupled` | 模型架构：`decoupled` (推荐) 或 `gcn` |
| `fl_rounds` | `100` | 首轮联邦迭代的通信次数 |
| `sbert_batch_size` | `32` | SBERT 推理批量大小 (针对 Mac MPS 优化) |
| `eval_fusion_alpha` | `0.42` | 评估时的结构/语义融合权重 |
| `llm_model_id` | `Qwen/Qwen2.5...` | 用于润色的 LLM 模型路径 |

-----

## 📊 实验结果 (Performance)

在 DBP15K (ZH-EN) 数据集上的实验表现：

| Method | Hits@1 (%) | Hits@10 (%) | 说明 |
| :--- | :---: | :---: | :--- |
| **No LLM (Raw SBERT)** | 61.13 | - | 基准：原始语义 + GCN |
| **No Mining (Iter=1)** | 68.87 | - | 进阶：LLM 增强 + GCN |
| **FedAnchor (Full)** | **71.34** | **83.5** | **最终：LLM + 迭代自训练** |

-----