# src/models/decoupled.py
import torch
import torch.nn as nn
from .encoders.gcn import GCNEncoder
from .projectors.mlp import MLPProjector


class DecoupledModel(nn.Module):
    """
    FedAnchor 核心模型架构：
    组合 Private Encoder 和 Shared Projector。
    """

    def __init__(self, cfg, num_entities):
        """
        :param cfg: Hydra 配置对象 (cfg.task.model)
        :param num_entities: 实体数量 (用于初始化 Encoder)
        """
        super(DecoupledModel, self).__init__()

        # 参数解包
        feature_dim = cfg.gcn_dim
        hidden_dim = cfg.gcn_hidden
        output_dim = 768  # SBERT 维度，通常固定，也可以放配置里
        dropout = cfg.dropout

        # 1. 初始化私有编码器 (Private)
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
        完整前向传播：Adjacency -> Encoder -> Features -> Projector -> Embeddings
        """
        # 1. 提取结构特征
        struct_features = self.encoder(adj)

        # 2. 映射语义空间
        semantic_embeddings = self.projector(struct_features)

        return semantic_embeddings

    # ======================================================
    #  联邦学习专用接口 (FL Interfaces)
    # ======================================================

    def get_shared_state_dict(self):
        """
        [Client 端调用]
        只获取需要上传给 Server 的参数 (Projector)。
        """
        # 直接返回子模块的 state_dict，不用担心 key 的前缀问题
        return self.projector.state_dict()

    def load_shared_state_dict(self, shared_state):
        """
        [Client 端调用]
        加载来自 Server 的全局参数。
        """
        self.projector.load_state_dict(shared_state)

    def get_private_state_dict(self):
        """
        [Client 本地保存用]
        获取私有参数 (Encoder)。
        """
        return self.encoder.state_dict()
