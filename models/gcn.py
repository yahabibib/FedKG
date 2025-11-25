# 📄 models/gcn.py
# 【防爆版】集成 LayerNorm + 关系门控 + 平均聚合

import torch
import torch.nn as nn
import torch.nn.functional as F


class RelationGCNLayer(nn.Module):
    def __init__(self, input_dim, output_dim, activation=F.relu):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias=False)
        self.activation = activation

        # 门控网络
        self.gate_linear = nn.Linear(input_dim, output_dim, bias=True)
        nn.init.xavier_uniform_(self.gate_linear.weight)
        # 偏置初始化为 2.0，让门默认开启
        nn.init.constant_(self.gate_linear.bias, 2.0)

        # 🔥 [关键新增] LayerNorm (防止数值爆炸的稳压器)
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, x, edge_index, edge_type, rel_emb):
        """
        x: [N, D]
        rel_emb: [TotalRels, D]
        """
        x_trans = self.linear(x)
        src_idx, tgt_idx = edge_index

        rel_emb = rel_emb.to(x.device)
        h_rel = rel_emb[edge_type]

        # 关系门控
        gate = torch.sigmoid(self.gate_linear(h_rel))
        msg = x_trans[src_idx] * gate

        # 平均聚合
        out = torch.zeros(x.shape[0], x_trans.shape[1], device=x.device)
        out.index_add_(0, tgt_idx, msg)

        ones = torch.ones(tgt_idx.size(0), 1, device=x.device)
        deg = torch.zeros(x.shape[0], 1, device=x.device)
        deg.index_add_(0, tgt_idx, ones)
        out = out / deg.clamp(min=1.0)

        # 🔥 [关键应用] 先 Norm 再激活
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
            f"    [Model Init] RelationGCN (LayerNorm): {num_entities} Ents, {total_rels} Rels")

        self.initial_features = nn.Parameter(
            torch.randn(num_entities, feature_dim))
        nn.init.xavier_uniform_(self.initial_features)

        self.relation_embeddings = nn.Parameter(
            torch.randn(total_rels, feature_dim))
        nn.init.xavier_uniform_(self.relation_embeddings)

        self.gc1 = RelationGCNLayer(feature_dim, hidden_dim)
        self.gc2 = RelationGCNLayer(hidden_dim, output_dim, activation=None)

        self.dropout = dropout

    def init_relation_embeddings(self, sbert_rel_emb):
        # 保留接口，防止报错，但这里不执行任何操作，依靠随机初始化
        pass

    def forward(self, edge_index, edge_type):
        x = self.initial_features

        x = self.gc1(x, edge_index, edge_type, self.relation_embeddings)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, edge_index, edge_type, self.relation_embeddings)

        return x
