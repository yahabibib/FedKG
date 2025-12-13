# src/models/encoders/gat.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class SparseGATLayer(nn.Module):
    """
    支持稀疏矩阵输入的 GAT 层 (Single Head)
    核心逻辑：学习边权重 alpha_ij，实现各向异性聚合，过滤噪声邻居。
    """

    def __init__(self, in_features, out_features, dropout=0.6, alpha=0.2):
        super(SparseGATLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.leakyrelu = nn.LeakyReLU(alpha)

        # 变换矩阵 W [In, Out]
        self.W = nn.Linear(in_features, out_features, bias=False)
        # 注意力向量 a [2*Out, 1]
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

    def forward(self, h, adj):
        # 1. 线性变换
        Wh = self.W(h)  # [N, Out]

        # 2. 准备注意力计算 (基于稀疏索引)
        indices = adj.indices()  # [2, E]
        rows, cols = indices[0], indices[1]

        # 3. 计算每条边的 Attention Score
        # e_ij = LeakyReLU(a^T [Wh_i || Wh_j])
        Wh_i = Wh[rows]
        Wh_j = Wh[cols]
        a_input = torch.cat([Wh_i, Wh_j], dim=1)  # [E, 2*Out]

        # [E, 1] -> [E]
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(1))

        # 4. Sparse Softmax
        # 为了数值稳定性，减去最大值
        e = e - e.max()
        exp_e = torch.exp(e)

        N = h.size(0)
        # 分母：对每个目标节点(row)的所有入边进行求和
        sum_exp = torch.zeros(N, device=h.device)
        sum_exp.index_add_(0, rows, exp_e)

        # 归一化 alpha_ij = exp(e_ij) / sum(exp(e_ik))
        alpha = exp_e / (sum_exp[rows] + 1e-10)

        # Dropout on Attention Weights
        alpha = F.dropout(alpha, self.dropout, training=self.training)

        # 5. 聚合
        # 构造带有 Attention 权重的稀疏矩阵
        adj_att = torch.sparse_coo_tensor(
            indices,
            alpha,
            torch.Size([N, N]),
            device=h.device
        )

        # H_new = Adj_att * Wh
        h_prime = torch.sparse.mm(adj_att, Wh)

        return F.elu(h_prime)


class GATEncoder(nn.Module):
    """
    [Private Component]
    使用 GAT 替代 GCN 进行结构特征提取。
    """

    def __init__(self, num_entities, feature_dim, hidden_dim, dropout=0.3):
        super(GATEncoder, self).__init__()

        # 1. 初始节点特征 (可学习的 Embedding)
        self.initial_features = nn.Parameter(
            torch.randn(num_entities, feature_dim))
        nn.init.xavier_uniform_(self.initial_features)

        # 2. GAT 层
        # 目前使用单层结构，如果需要更深网络，可以在此叠加
        self.layer = SparseGATLayer(feature_dim, hidden_dim, dropout=dropout)

        self.dropout = dropout

    def forward(self, adj):
        x = self.initial_features
        x = F.dropout(x, self.dropout, training=self.training)

        # GAT Forward
        x = self.layer(x, adj)

        return x
