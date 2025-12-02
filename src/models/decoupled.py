# src/models/decoupled.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.gcn import GCNLayer


class DecoupledModel(nn.Module):
    """
    FedAnchor 核心架构: Decoupled (解耦)
    - struct_encoder (Private): 捕捉本地特有的图拓扑结构，参数不上传。
    - semantic_projector (Shared): 学习通用的语义映射，参数参与联邦聚合。
    """

    def __init__(self, num_entities: int, feature_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.3):
        super().__init__()

        # --- Private Component (Local) ---
        self.initial_features = nn.Parameter(
            torch.randn(num_entities, feature_dim))
        nn.init.xavier_uniform_(self.initial_features)

        # 结构编码器 (私有): 只在本地更新
        self.struct_encoder = nn.ModuleList([
            GCNLayer(feature_dim, hidden_dim),
            # 可以根据需要添加更多层
        ])

        # --- Shared Component (Global) ---
        # 语义投影器 (公共): 负责将结构特征对齐到 SBERT 空间
        # 这是一个 MLP
        self.semantic_projector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )

        self.dropout = dropout

    def forward(self, adj: torch.sparse.Tensor) -> torch.Tensor:
        x = self.initial_features

        # 1. Private Structure Encoding (GCN)
        for layer in self.struct_encoder:
            x = layer(x, adj)
            x = F.relu(x)
            x = F.dropout(x, self.dropout, training=self.training)

        # 2. Shared Semantic Projection (MLP)
        # 输入是 GCN 提取的 hidden_dim 特征，输出是 BERT_DIM
        x = self.semantic_projector(x)

        return x
