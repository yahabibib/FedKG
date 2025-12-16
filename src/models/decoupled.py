# src/models/decoupled.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from .encoders.gcn import GCNEncoder
from .encoders.gat import GATEncoder
from .encoders.rgat import RGATEncoder
from .encoders.sage import SAGEEncoder
from .projectors.mlp import MLPProjector


class DecoupledModel(nn.Module):
    # [修改] 增加 num_relations 参数
    def __init__(self, cfg, num_entities, num_relations=0):
        super(DecoupledModel, self).__init__()

        feature_dim = cfg.gcn_dim
        hidden_dim = cfg.gcn_hidden
        output_dim = 768
        dropout = cfg.dropout

        encoder_type = getattr(cfg, 'encoder_name', 'gcn').lower()

        # --- Encoder 初始化逻辑 ---
        if encoder_type == 'gat':
            self.encoder = GATEncoder(
                num_entities=num_entities,
                feature_dim=feature_dim,
                hidden_dim=hidden_dim,
                dropout=dropout
            )
        elif encoder_type == 'rgat':
            self.encoder = RGATEncoder(
                num_entities=num_entities,
                num_relations=num_relations,  # 必须传
                feature_dim=feature_dim,
                hidden_dim=hidden_dim,
                dropout=dropout
            )
        elif encoder_type == 'sage':
            self.encoder = SAGEEncoder(
                num_entities=num_entities,
                feature_dim=feature_dim,
                hidden_dim=hidden_dim,
                dropout=dropout
            )
        else:
            self.encoder = GCNEncoder(
                num_entities=num_entities,
                feature_dim=feature_dim,
                hidden_dim=hidden_dim,
                dropout=dropout
            )

        self.projector = MLPProjector(
            input_dim=hidden_dim,
            output_dim=output_dim,
            dropout=dropout
        )

    def forward(self, adj, edge_types=None):  # [修改] 增加 edge_types 参数
        """
        前向传播，根据 Encoder 类型自动决定是否使用 edge_types
        """
        if isinstance(self.encoder, RGATEncoder):
            if edge_types is None:
                raise ValueError("RGATEncoder requires 'edge_types' input!")
            struct_features = self.encoder(adj, edge_types)
        else:
            # GCN, GAT, SAGE 忽略 edge_types
            struct_features = self.encoder(adj)

        semantic_embeddings = self.projector(struct_features)
        return F.normalize(semantic_embeddings, p=2, dim=1)

    # ... get/load state dict 方法保持不变 ...
    def get_shared_state_dict(self):
        return self.projector.state_dict()

    def load_shared_state_dict(self, shared_state):
        self.projector.load_state_dict(shared_state)

    def get_private_state_dict(self):
        return self.encoder.state_dict()
