# src/models/decoupled.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from .encoders.gcn import GCNEncoder
from .encoders.gat import GATEncoder
from .encoders.sage import SAGEEncoder
from .projectors.mlp import MLPProjector


class DecoupledModel(nn.Module):
    """
    FedAnchor 核心模型架构：
    组合 Private Encoder (GCN/GAT) 和 Shared Projector (MLP)。
    """

    def __init__(self, cfg, num_entities):
        """
        :param cfg: Hydra 配置对象 (cfg.task.model)
        :param num_entities: 实体数量
        """
        super(DecoupledModel, self).__init__()

        # 参数解包
        feature_dim = cfg.gcn_dim
        hidden_dim = cfg.gcn_hidden
        output_dim = 768
        dropout = cfg.dropout

        # [核心修改] 根据配置选择 Encoder 类型
        # 默认为 'gcn' 以兼容旧配置
        encoder_type = getattr(cfg, 'encoder_name', 'gcn').lower()

        if encoder_type == 'gat':
            self.encoder = GATEncoder(
                num_entities=num_entities,
                feature_dim=feature_dim,
                hidden_dim=hidden_dim,
                dropout=dropout
            )
        elif encoder_type == 'sage':  # [新增分支]
            # print(f"   [Model] Initializing Private Encoder: GraphSAGE")
            self.encoder = SAGEEncoder(
                num_entities=num_entities,
                feature_dim=feature_dim,
                hidden_dim=hidden_dim,
                dropout=dropout
            )
        else:
            # Default to GCN
            self.encoder = GCNEncoder(
                num_entities=num_entities,
                feature_dim=feature_dim,
                hidden_dim=hidden_dim,
                dropout=dropout
            )
        # 2. 初始化共享投影器 (Shared)
        self.projector = MLPProjector(
            input_dim=hidden_dim,
            output_dim=output_dim,
            dropout=dropout
        )

    def forward(self, adj):
        """
        完整前向传播
        """
        # 1. 提取结构特征
        struct_features = self.encoder(adj)

        # 2. 映射语义空间
        semantic_embeddings = self.projector(struct_features)

        # [关键修复] 强制归一化，防止特征坍缩
        # 这对于 Cosine Similarity Loss 至关重要
        return F.normalize(semantic_embeddings, p=2, dim=1)

    # ... (后续的 state_dict 相关方法保持不变) ...
    # 只需要保留下面的 get/load 方法即可
    def get_shared_state_dict(self):
        return self.projector.state_dict()

    def load_shared_state_dict(self, shared_state):
        self.projector.load_state_dict(shared_state)

    def get_private_state_dict(self):
        return self.encoder.state_dict()
