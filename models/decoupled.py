# 📄 models/decoupled.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from .gcn import RelationGCNLayer  # <--- 升级导入 RelationGCNLayer


class DecoupledModel(nn.Module):
    """
    【关系感知解耦模型】
    - struct_encoder (RelationGCN): 私有层，提取关系感知的结构特征
    - semantic_projector (MLP): 公共层，映射到语义空间
    """

    def __init__(self, num_entities, num_relations, feature_dim, hidden_dim, output_dim, dropout=0.3):
        super().__init__()

        # 注意：num_relations 是原始关系数，内部层会处理反向和自环，
        # 但我们需要在这里维护 Relation Embeddings，因为它是结构编码器的一部分
        total_rels = 2 * num_relations + 1

        print(
            f"    [Model Init] Decoupled (Relation-Aware): {num_entities} Ents, {total_rels} Rels")

        # --- 1. 私有结构编码器 (Private) ---
        # 初始节点特征
        self.initial_features = nn.Parameter(
            torch.randn(num_entities, feature_dim))
        nn.init.xavier_uniform_(self.initial_features)

        # 初始关系特征 (私有参数)
        self.relation_embeddings = nn.Parameter(
            torch.randn(total_rels, feature_dim))
        nn.init.xavier_uniform_(self.relation_embeddings)

        # 使用 RelationGCNLayer 替代原来的 GCNLayer
        self.struct_encoder = nn.ModuleList([
            RelationGCNLayer(feature_dim, hidden_dim),
            # 可以加更多层，这里保持双层结构
            # 第二层输入 hidden, 输出 hidden (因为最后还要过 MLP)
            # 或者像 GCN 那样第二层直接输出 output_dim?
            # 这里的 Decoupled 架构通常是 GCN 负责提取特征，MLP 负责对齐
            # 所以我们让 GCN 输出 hidden_dim
        ])

        # 为了加深网络，可以加一个层间激活
        self.activation = nn.ReLU()

        # --- 2. 公共语义映射器 (Shared) ---
        # MLP: Hidden -> Output (SBERT Dim)
        self.semantic_projector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )

        self.dropout = dropout

    def init_relation_embeddings(self, sbert_rel_emb):
        """ 用 SBERT 初始化关系嵌入 """
        with torch.no_grad():
            count = 0
            for rid, emb in sbert_rel_emb.items():
                if rid < self.relation_embeddings.shape[0]:
                    dim = min(self.relation_embeddings.shape[1], emb.shape[0])
                    self.relation_embeddings.data[rid, :dim] = emb[:dim]
                    count += 1
            print(f"    [Decoupled] Initialized {count} relation embeddings.")

    def forward(self, edge_index, edge_type):
        x = self.initial_features

        # 1. 经过私有 RelationGCN 编码
        # 注意：现在需要传入 edge_index, edge_type, relation_embeddings
        for i, layer in enumerate(self.struct_encoder):
            x = layer(x, edge_index, edge_type, self.relation_embeddings)
            if i < len(self.struct_encoder) - 1:  # 如果有多层
                x = self.activation(x)
                x = F.dropout(x, self.dropout, training=self.training)

        # 2. 经过公共 MLP 映射
        x = self.semantic_projector(x)

        return x
