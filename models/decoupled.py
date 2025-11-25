# 📄 models/decoupled.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from .gcn import RelationGCNLayer


class DecoupledModel(nn.Module):
    def __init__(self, num_entities, num_relations, feature_dim, hidden_dim, output_dim, dropout=0.3):
        super().__init__()

        self.num_base_relations = num_relations
        total_rels = 2 * num_relations + 1
        print(
            f"    [Model Init] Decoupled (Gating): {num_entities} Ents, {total_rels} Rels")

        self.initial_features = nn.Parameter(
            torch.randn(num_entities, feature_dim))
        nn.init.xavier_uniform_(self.initial_features)

        # 关系嵌入 (随机初始化)
        self.relation_embeddings = nn.Parameter(
            torch.randn(total_rels, feature_dim))
        nn.init.xavier_uniform_(self.relation_embeddings)

        self.struct_encoder = nn.ModuleList([
            RelationGCNLayer(feature_dim, hidden_dim),
        ])

        self.activation = nn.ReLU()

        self.semantic_projector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        self.dropout = dropout

    def init_relation_embeddings(self, sbert_rel_emb):
        # 同步保留该方法
        with torch.no_grad():
            count = 0
            for rid, emb in sbert_rel_emb.items():
                rid = int(rid)
                if rid < self.relation_embeddings.shape[0]:
                    dim = min(self.relation_embeddings.shape[1], emb.shape[0])
                    self.relation_embeddings.data[rid, :dim] = emb[:dim]
                    inv_rid = rid + self.num_base_relations
                    if inv_rid < self.relation_embeddings.shape[0]:
                        self.relation_embeddings.data[inv_rid,
                                                      :dim] = emb[:dim]
                    count += 1
            print(f"    [Decoupled] Initialized {count} relation pairs.")

    def forward(self, edge_index, edge_type):
        x = self.initial_features

        for i, layer in enumerate(self.struct_encoder):
            x = layer(x, edge_index, edge_type, self.relation_embeddings)
            if i < len(self.struct_encoder) - 1:
                x = self.activation(x)
                x = F.dropout(x, self.dropout, training=self.training)

        x = self.semantic_projector(x)
        return x
