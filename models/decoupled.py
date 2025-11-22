# 📄 AiStudy/models/decoupled.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from .gcn import GCNLayer  # 复用之前的 GCNLayer


class DecoupledModel(nn.Module):
    """
    【解耦模型架构】
    - struct_encoder (GCN): 私有层，负责提取本地结构特征 (不聚合)
    - semantic_projector (MLP): 公共层，负责映射到语义空间 (聚合)
    """

    def __init__(self, num_entities, feature_dim, hidden_dim, output_dim, dropout=0.3):
        super().__init__()

        print(f"    [Model Init] Decoupled: GCN(Private) -> MLP(Shared)")

        # --- 1. 私有结构编码器 (Private) ---
        # 可学习的初始节点特征
        self.initial_features = nn.Parameter(
            torch.randn(num_entities, feature_dim))
        nn.init.xavier_uniform_(self.initial_features)

        # GCN 层 (只负责提取结构，不负责对齐)
        self.struct_encoder = nn.ModuleList([
            GCNLayer(feature_dim, hidden_dim),
            # 可以加更多层，这里保持双层结构
        ])

        # --- 2. 公共语义映射器 (Shared) ---
        # 这是一个 MLP，负责把结构特征翻译成 SBERT 语义
        # 它的输入是 GCN 的输出 (hidden_dim)，输出是 SBERT 维度 (output_dim)
        self.semantic_projector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )

        self.dropout = dropout

    def forward(self, adj):
        x = self.initial_features

        # 1. 经过私有 GCN 编码
        for layer in self.struct_encoder:
            x = layer(x, adj)
            x = F.relu(x)
            x = F.dropout(x, self.dropout, training=self.training)

        # 2. 经过公共 MLP 映射
        # 注意：GCN 的输出经过 MLP 调整后，才去和 SBERT 做 Loss
        x = self.semantic_projector(x)

        return x
