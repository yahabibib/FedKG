# 📄 models/gcn.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class GCNLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias=False)

    def forward(self, x, adj):
        # 1. 线性变换 (GPU 加速)
        x = self.linear(x)

        # 2. 邻居聚合 (CPU/GPU 混合计算兼容)
        # 如果 x 在 MPS (GPU) 上，但 adj 在 CPU 上，则将 x 临时转到 CPU 进行稀疏乘法
        if x.device.type == 'mps' and adj.device.type == 'cpu':
            out = torch.sparse.mm(adj, x.cpu()).to(x.device)
        else:
            out = torch.sparse.mm(adj, x)

        return out


class GCN(nn.Module):
    def __init__(self, num_entities, feature_dim, hidden_dim, output_dim, dropout=0.3):
        super().__init__()

        print(
            f"    [Model Init] GCN: {feature_dim} -> {hidden_dim} -> {output_dim} (Dropout: {dropout})")

        # 初始结构特征 (Node Embeddings)
        self.initial_features = nn.Parameter(
            torch.randn(num_entities, feature_dim))
        nn.init.xavier_uniform_(self.initial_features)

        self.gc1 = GCNLayer(feature_dim, hidden_dim)
        self.gc2 = GCNLayer(hidden_dim, output_dim)

        self.dropout = dropout

    def forward(self, adj):
        x = self.initial_features

        # Layer 1
        x = self.gc1(x, adj)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)

        # Layer 2
        x = self.gc2(x, adj)

        return x
