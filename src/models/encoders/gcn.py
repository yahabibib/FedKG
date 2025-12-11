# src/models/encoders/gcn.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class GCNLayer(nn.Module):
    def __init__(self, input_dim, output_dim, use_bias=False):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias=use_bias)

    def forward(self, x, adj):
        """
        x: Node Features [N, In_Dim]
        adj: Sparse Adjacency Matrix [N, N]
        """
        # 1. 线性变换 (Dense Compute) -> GPU/MPS Friendly
        x = self.linear(x)

        # 2. 邻居聚合 (Sparse Compute) -> MPS Compatibility Fix
        # 如果 x 在 MPS 上，但 adj 只能在 CPU 上运算 (MPS 对 sparse_mm 支持有限)
        # 以 adj 的设备为准进行计算
        if adj.device != x.device:
            target_device = x.device
            # 临时把 x 搬去 adj 所在的设备 (通常是 CPU)
            out = torch.sparse.mm(adj, x.to(adj.device))
            # 算完搬回来
            out = out.to(target_device)
        else:
            # 正常情况
            out = torch.sparse.mm(adj, x)

        return out


class GCNEncoder(nn.Module):
    """
    [Private Component]
    负责提取图的拓扑结构特征。这部分参数在联邦学习中保留在本地，不上传。
    """

    def __init__(self, num_entities, feature_dim, hidden_dim, dropout=0.3):
        super(GCNEncoder, self).__init__()

        # 1. 初始节点特征 (可学习的 Embedding)
        # 这是每个 Client 独有的，包含了这一侧图谱的实体信息
        self.initial_features = nn.Parameter(
            torch.randn(num_entities, feature_dim))
        nn.init.xavier_uniform_(self.initial_features)

        # 2. GCN 层
        # 目前是双层结构：Input -> Hidden
        self.layers = nn.ModuleList([
            GCNLayer(feature_dim, hidden_dim)
            # 如果需要更深，可以继续 append
        ])

        self.dropout = dropout

    def forward(self, adj):
        x = self.initial_features

        for layer in self.layers:
            x = layer(x, adj)
            x = F.relu(x)
            x = F.dropout(x, self.dropout, training=self.training)

        return x
