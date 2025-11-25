# 📄 models/decoupled.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from .gcn import ReflectionLayer


class DecoupledModel(nn.Module):
    def __init__(self, num_entities, num_relations, feature_dim, hidden_dim, output_dim, dropout=0.3):
        super().__init__()

        self.num_base_relations = num_relations
        total_rels = 2 * num_relations + 1
        print(
            f"    [Model Init] Decoupled (RREA-Safe): {num_entities} Ents, {total_rels} Rels")

        self.initial_features = nn.Parameter(
            torch.randn(num_entities, feature_dim))
        nn.init.xavier_uniform_(self.initial_features)

        self.relation_embeddings = nn.Parameter(
            torch.randn(total_rels, feature_dim))
        nn.init.xavier_uniform_(self.relation_embeddings)

        # 两层 RREA 结构
        self.struct_encoder = nn.ModuleList([
            ReflectionLayer(feature_dim, hidden_dim),
            ReflectionLayer(hidden_dim, hidden_dim)
        ])

        # MLP 投射层
        self.semantic_projector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        self.dropout = dropout

    def init_relation_embeddings(self, sbert_rel_emb):
        pass

    def forward(self, edge_index, edge_type):
        x = self.initial_features

        for i, layer in enumerate(self.struct_encoder):
            x = layer(x, edge_index, edge_type, self.relation_embeddings)
            # Layer 内部已经有 Norm 和 Activation 了
            # 这里只需要处理层间 Dropout
            if i < len(self.struct_encoder) - 1:
                x = F.dropout(x, self.dropout, training=self.training)

        x = self.semantic_projector(x)
        return x
