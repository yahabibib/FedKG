# src/models/gcn.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class GCNLayer(nn.Module):
    """
    基础图卷积层: Linear -> SparseMM -> Activation
    """

    def __init__(self, input_dim: int, output_dim: int, use_bias: bool = False):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias=use_bias)

    def forward(self, x: torch.Tensor, adj: torch.sparse.Tensor) -> torch.Tensor:
        # 1. 线性变换 (Feature Transformation)
        x = self.linear(x)

        # 2. 邻居聚合 (Message Passing)
        # 兼容性处理: MPS 目前对稀疏矩阵乘法的支持有限，如果报错或处于 MPS 环境，
        # 有时需要将稀疏矩阵乘法回退到 CPU 执行，或者确保 adj 也在 MPS 上。
        # 这里保留你原有的稳健逻辑：如果 x 在 MPS 但 adj 在 CPU，则在 CPU 计算后转回。
        if x.device.type == 'mps' and adj.device.type == 'cpu':
            out = torch.sparse.mm(adj, x.cpu()).to(x.device)
        else:
            out = torch.sparse.mm(adj, x)

        return out


class GCN(nn.Module):
    """
    标准两层 GCN
    """

    def __init__(self, num_entities: int, feature_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.3):
        super().__init__()
        # 1. 可学习的初始节点特征 (Node Embeddings)
        # 也可以选择传入预训练特征，这里默认随机初始化
        self.initial_features = nn.Parameter(
            torch.randn(num_entities, feature_dim))
        nn.init.xavier_uniform_(self.initial_features)

        # 2. GCN 层
        self.gc1 = GCNLayer(feature_dim, hidden_dim)
        self.gc2 = GCNLayer(hidden_dim, output_dim)

        self.dropout = dropout

    def forward(self, adj: torch.sparse.Tensor) -> torch.Tensor:
        x = self.initial_features

        # Layer 1
        x = self.gc1(x, adj)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)

        # Layer 2
        x = self.gc2(x, adj)

        # 通常最后一层输出 embeddings，不需要再过激活函数
        return x
