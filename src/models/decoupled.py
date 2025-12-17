# src/models/decoupled.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from .encoders.gcn import GCNEncoder
from .encoders.gat import GATEncoder
from .encoders.rgat import RGATEncoder
from .encoders.sage import SAGEEncoder
from .projectors.mlp import MLPProjector


class AdaptiveGate(nn.Module):
    """
    [新增] 自适应门控模块
    """

    def __init__(self, input_dim):
        super(AdaptiveGate, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim * 2, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, struct_emb, sbert_emb):
        combined = torch.cat([struct_emb, sbert_emb], dim=1)
        alpha = self.net(combined)
        return alpha


class SharedArchitecture(nn.Module):
    """
    [核心修复] 共享架构容器
    将 Projector 和 Gate 封装在一个 Module 里，
    这样调用 state_dict() 时会自动生成扁平化的 key (e.g., 'projector.net.0.weight')，
    解决 Server 端聚合时的 AttributeError。
    """

    def __init__(self, hidden_dim, output_dim, dropout=0.3):
        super(SharedArchitecture, self).__init__()
        self.projector = MLPProjector(hidden_dim, output_dim, dropout)
        self.gate = AdaptiveGate(output_dim)


class DecoupledModel(nn.Module):
    def __init__(self, cfg, num_entities, num_relations=0):
        super(DecoupledModel, self).__init__()

        feature_dim = cfg.gcn_dim
        hidden_dim = cfg.gcn_hidden
        output_dim = 768
        dropout = cfg.dropout

        encoder_type = getattr(cfg, 'encoder_name', 'gcn').lower()

        # 1. Encoder (Private)
        if encoder_type == 'gat':
            self.encoder = GATEncoder(
                num_entities, feature_dim, hidden_dim, dropout)
        elif encoder_type == 'rgat':
            self.encoder = RGATEncoder(
                num_entities, num_relations, feature_dim, hidden_dim, dropout)
        elif encoder_type == 'sage':
            self.encoder = SAGEEncoder(
                num_entities, feature_dim, hidden_dim, dropout)
        else:
            self.encoder = GCNEncoder(
                num_entities, feature_dim, hidden_dim, dropout)

        # 2. Shared Architecture (Projector + Gate)
        # [修改] 使用容器初始化
        self.shared = SharedArchitecture(hidden_dim, output_dim, dropout)

        # 方便快捷访问 (指针引用)
        self.projector = self.shared.projector
        self.gate = self.shared.gate

    def forward(self, adj, edge_types=None):
        if isinstance(self.encoder, RGATEncoder):
            if edge_types is None:
                raise ValueError("RGATEncoder requires 'edge_types' input!")
            struct_features = self.encoder(adj, edge_types)
        else:
            struct_features = self.encoder(adj)

        semantic_embeddings = self.projector(struct_features)
        return F.normalize(semantic_embeddings, p=2, dim=1)

    def fuse(self, struct_emb, sbert_emb):
        alpha = self.gate(struct_emb, sbert_emb)
        fused = alpha * struct_emb + (1 - alpha) * sbert_emb
        return F.normalize(fused, p=2, dim=1), alpha

    def get_shared_state_dict(self):
        # [修复] 直接返回 shared 容器的 state_dict，它是扁平的 Tensor 字典
        return self.shared.state_dict()

    def load_shared_state_dict(self, shared_state):
        # [修复] 直接加载
        self.shared.load_state_dict(shared_state)

    def get_private_state_dict(self):
        return self.encoder.state_dict()
