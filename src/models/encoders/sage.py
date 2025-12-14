# src/models/encoders/sage.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphSAGELayer(nn.Module):
    """
    GraphSAGE (Mean Aggregator) 层实现
    核心逻辑: Output = Linear( Concat( Input, Mean(Neighbors) ) )
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphSAGELayer, self).__init__()

        # GraphSAGE 的特征拼接：输入维度变成 2倍
        # [Self_Dim + Neighbor_Dim] -> Out_Dim
        self.linear = nn.Linear(2 * in_features, out_features, bias=bias)

    def forward(self, x, adj):
        """
        x: [N, In_Dim]
        adj: Sparse Adjacency Matrix
        """
        # --- 1. 邻居聚合 (Neighbor Aggregation) ---
        # [MPS 兼容性修复]: 检查设备一致性
        if adj.device != x.device:
            target_device = x.device
            # 将 x 搬运到 adj 所在的设备 (通常是 CPU) 进行稀疏乘法
            # adj (CPU) @ x (CPU) -> neigh_feat (CPU)
            neigh_feat = torch.sparse.mm(adj, x.to(adj.device))
            # 结果搬回 GPU/MPS
            neigh_feat = neigh_feat.to(target_device)
        else:
            # 常规情况
            neigh_feat = torch.sparse.mm(adj, x)

        # --- 2. 特征拼接 (Concatenation) ---
        # 显式保留自身信息 (Skip Connection 的一种强形式)
        # [N, In] || [N, In] -> [N, 2*In]
        combined = torch.cat([x, neigh_feat], dim=1)

        # --- 3. 线性变换 ---
        out = self.linear(combined)
        return out


class SAGEEncoder(nn.Module):
    """
    [Private Component]
    GraphSAGE 编码器，用于提取结构特征。
    """

    def __init__(self, num_entities, feature_dim, hidden_dim, dropout=0.3):
        super(SAGEEncoder, self).__init__()

        # 1. 初始节点特征 (Learnable Embeddings)
        self.initial_features = nn.Parameter(
            torch.randn(num_entities, feature_dim)
        )
        nn.init.xavier_uniform_(self.initial_features)

        # 2. 定义层结构
        # 这里演示单层结构，如果需要多层，注意中间层的输入输出维度匹配
        self.layers = nn.ModuleList([
            GraphSAGELayer(feature_dim, hidden_dim)
        ])

        self.dropout = dropout

    def forward(self, adj):
        x = self.initial_features

        for layer in self.layers:
            x = layer(x, adj)
            x = F.relu(x)
            x = F.dropout(x, self.dropout, training=self.training)

        return x
