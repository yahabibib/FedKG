# src/models/encoders/rgat.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class RelationalGATLayer(nn.Module):
    """
    [Pseudo-RGCN] 关系感知图注意力层 (R-GAT)

    Standard GAT: e_ij = LeakyReLU(a^T [Wh_i || Wh_j])
    Relational GAT: e_ij = LeakyReLU(a^T [Wh_i || Wh_j || Emb(r_ij)])
    """

    def __init__(self, in_features, out_features, num_relations, dropout=0.6, alpha=0.2):
        super(RelationalGATLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha_val = alpha

        # 1. 节点特征变换 (共享参数)
        self.W = nn.Linear(in_features, out_features, bias=False)

        # 2. 关系嵌入 (Relation Embeddings)
        # 维度设为 out_features，方便拼接
        self.rel_dim = out_features
        self.rel_emb = nn.Embedding(num_relations, self.rel_dim)
        nn.init.xavier_uniform_(self.rel_emb.weight)

        # 3. 注意力向量 a
        # 输入维度：[Node_Out + Node_Out + Rel_Dim]
        self.attn_dim = 2 * out_features + self.rel_dim
        self.a = nn.Parameter(torch.zeros(size=(self.attn_dim, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

    def forward(self, h, adj, edge_types):
        """
        h: [N, D] 节点特征
        adj: Sparse Matrix 邻接矩阵
        edge_types: [E] 边的关系类型 ID (与 adj.indices 对应)
        """
        Wh = self.W(h)

        # --- 设备一致性检查 (混合计算支持) ---
        if adj.device != h.device:
            target_device = h.device
            compute_device = adj.device

            # 搬运数据到计算设备 (通常是 CPU)
            Wh_comp = Wh.to(compute_device)
            a_comp = self.a.to(compute_device)
            et_comp = edge_types.to(compute_device)

            # Embedding 查找在计算设备进行
            rel_emb_weight = self.rel_emb.weight.to(compute_device)
            r_vecs = F.embedding(et_comp, rel_emb_weight)

            out = self._compute_attention(
                Wh_comp, adj, a_comp, r_vecs, compute_device)
            return out.to(target_device)
        else:
            # 常规模式
            r_vecs = self.rel_emb(edge_types)
            return self._compute_attention(Wh, adj, self.a, r_vecs, h.device)

    def _compute_attention(self, Wh, adj, a_param, r_vecs, device):
        indices = adj.indices()
        rows, cols = indices[0], indices[1]

        # 准备 Attention 输入
        Wh_i = Wh[rows]
        Wh_j = Wh[cols]

        # [核心] 拼接：[源 || 目标 || 关系]
        a_input = torch.cat([Wh_i, Wh_j, r_vecs], dim=1)

        # 计算分数 LeakyReLU
        e = F.leaky_relu(torch.matmul(a_input, a_param).squeeze(
            1), negative_slope=self.alpha_val)

        # Sparse Softmax
        e = e - e.max()
        exp_e = torch.exp(e)

        N = Wh.size(0)
        sum_exp = torch.zeros(N, device=device)
        sum_exp.index_add_(0, rows, exp_e)

        alpha = exp_e / (sum_exp[rows] + 1e-10)
        alpha = F.dropout(alpha, self.dropout, training=self.training)

        # 聚合 H' = Adj_att * Wh
        adj_att = torch.sparse_coo_tensor(
            indices, alpha, torch.Size([N, N]), device=device
        )
        h_prime = torch.sparse.mm(adj_att, Wh)

        return F.elu(h_prime)


class RGATEncoder(nn.Module):
    """
    [Private Component] Pseudo-RGCN Encoder 包装器
    """

    def __init__(self, num_entities, num_relations, feature_dim, hidden_dim, dropout=0.3):
        super(RGATEncoder, self).__init__()

        # 1. 初始节点特征
        self.initial_features = nn.Parameter(
            torch.randn(num_entities, feature_dim)
        )
        nn.init.xavier_uniform_(self.initial_features)

        # 2. R-GAT 层 (传入关系总数)
        self.layer = RelationalGATLayer(
            feature_dim, hidden_dim, num_relations, dropout=dropout
        )
        self.dropout = dropout

    def forward(self, adj, edge_types):
        x = self.initial_features
        x = F.dropout(x, self.dropout, training=self.training)

        # 必须传入 edge_types
        x = self.layer(x, adj, edge_types)
        return x
