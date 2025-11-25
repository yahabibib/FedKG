# 📄 models/gcn.py
# 【RREA 安全版】移除输入强制归一化，保留关系归一化，增强数值稳定性

import torch
import torch.nn as nn
import torch.nn.functional as F


class ReflectionLayer(nn.Module):
    def __init__(self, in_channels, output_dim, activation=F.relu):
        super().__init__()
        self.in_channels = in_channels
        self.activation = activation

        # Shape Builder 变换矩阵
        self.W = nn.Linear(in_channels, output_dim, bias=False)
        nn.init.xavier_uniform_(self.W.weight)

        # 内置 LayerNorm (这是防爆的关键，必须保留)
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, x, edge_index, edge_type, rel_emb):
        """
        x: [N, D]
        rel_emb: [TotalRels, D]
        """
        src_idx, tgt_idx = edge_index

        # 1. 准备关系向量
        rel_emb = rel_emb.to(x.device)

        # ⚠️ 必须归一化关系向量 (||r||=1)，否则反射公式不成立
        # 增加 eps 防止除零异常
        rel_emb = F.normalize(rel_emb, p=2, dim=1, eps=1e-6)

        # 2. 准备节点特征
        # ❌ 移除对 x 的强制归一化，避免梯度问题，让 LayerNorm 去处理尺度
        h_src = x[src_idx]
        h_rel = rel_emb[edge_type]

        # 3. 关系反射变换 (Relational Reflection)
        # 公式: h' = h - 2 * (h . r) * r

        # 计算点积
        dot_prod = torch.sum(h_src * h_rel, dim=1, keepdim=True)

        # 执行反射
        h_reflected = h_src - 2 * dot_prod * h_rel

        # 4. 聚合 (Mean Aggregation)
        out = torch.zeros(x.shape[0], h_reflected.shape[1], device=x.device)
        out.index_add_(0, tgt_idx, h_reflected)

        # 度归一化
        ones = torch.ones(tgt_idx.size(0), 1, device=x.device)
        deg = torch.zeros(x.shape[0], 1, device=x.device)
        deg.index_add_(0, tgt_idx, ones)
        out = out / deg.clamp(min=1.0)

        # 5. 线性变换 (Shape Building)
        out = self.W(out)

        # 6. 残差连接 (Residual)
        # 只有维度匹配时才加残差
        if out.shape == x.shape:
            out = out + x

        # 7. 输出稳压 (LayerNorm)
        out = self.norm(out)

        if self.activation:
            out = self.activation(out)

        return out


class RelationGCN(nn.Module):
    def __init__(self, num_entities, num_relations, feature_dim, hidden_dim, output_dim, dropout=0.3):
        super().__init__()

        self.num_base_relations = num_relations
        total_rels = 2 * num_relations + 1

        print(
            f"    [Model Init] RREA (Safe): {num_entities} Ents, {total_rels} Rels")

        self.initial_features = nn.Parameter(
            torch.randn(num_entities, feature_dim))
        nn.init.xavier_uniform_(self.initial_features)

        self.relation_embeddings = nn.Parameter(
            torch.randn(total_rels, feature_dim))
        nn.init.xavier_uniform_(self.relation_embeddings)

        # 定义两层
        self.gc1 = ReflectionLayer(feature_dim, hidden_dim)
        self.gc2 = ReflectionLayer(hidden_dim, output_dim, activation=None)

        self.dropout = dropout

    def init_relation_embeddings(self, sbert_rel_emb):
        pass

    def forward(self, edge_index, edge_type):
        x = self.initial_features

        x = self.gc1(x, edge_index, edge_type, self.relation_embeddings)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, edge_index, edge_type, self.relation_embeddings)

        return x
