# src/core/client.py
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import logging
from typing import Dict, Any

from src.utils.config import Config
from src.models.decoupled import DecoupledModel
from src.models.gcn import GCN


class FederatedClient:
    """
    联邦学习客户端
    负责本地训练、私有参数维护和锚点更新。
    """

    def __init__(self, client_id: str, config: Config, data: Dict[str, Any]):
        self.client_id = client_id
        self.cfg = config
        self.device = config.device
        self.logger = logging.getLogger(f"Client-{client_id}")

        # 1. 解包数据
        # data 包含: 'adj', 'features', 'num_ent', 'anchors' (SBERT embeddings)

        # 【核心修复】MPS 不支持 Sparse Tensor，必须强制留在 CPU
        if self.device.type == 'mps':
            self.adj = data['adj']  # Keep on CPU
            # self.logger.info("MPS detected: Keeping sparse adjacency matrix on CPU.")
        else:
            self.adj = data['adj'].to(self.device)

        self.num_entities = data['num_ent']

        # 初始锚点 (SBERT Embeddings)
        # 格式: {entity_id: tensor_embedding}
        self.anchors_map = data.get('anchors', {})
        self.sbert_target = self._build_anchor_tensor(self.anchors_map)

        # 2. 初始化模型
        self.model = self._init_model()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.cfg.fl_lr)
        self.criterion = nn.MarginRankingLoss(margin=self.cfg.fl_margin)

    def _init_model(self):
        """初始化模型 (支持 Decoupled 或 普通 GCN)"""
        if self.cfg.model_arch == 'decoupled':
            # self.logger.info(f"Initializing Decoupled Model")
            return DecoupledModel(
                num_entities=self.num_entities,
                feature_dim=self.cfg.gcn_dim,
                hidden_dim=self.cfg.gcn_hidden,
                output_dim=self.cfg.bert_dim,
                dropout=self.cfg.gcn_dropout
            ).to(self.device)
        else:
            return GCN(
                num_entities=self.num_entities,
                feature_dim=self.cfg.gcn_dim,
                hidden_dim=self.cfg.gcn_hidden,
                output_dim=self.cfg.bert_dim,
                dropout=self.cfg.gcn_dropout
            ).to(self.device)

    def _build_anchor_tensor(self, anchors_map):
        """将锚点字典转换为 Tensor，用于快速计算 Loss"""
        target_tensor = torch.zeros(self.num_entities, self.cfg.bert_dim)
        train_indices = []

        for eid, emb in anchors_map.items():
            if eid < self.num_entities:
                target_tensor[eid] = emb
                train_indices.append(eid)

        self.train_indices = torch.tensor(train_indices, device=self.device)
        self.logger.info(f"Loaded {len(train_indices)} anchors.")
        return target_tensor.to(self.device)

    def update_anchors(self, new_anchors: Dict[int, torch.Tensor]):
        """
        [Stage 2.3] 迭代自训练：更新伪标签锚点
        """
        count = 0
        for eid, emb in new_anchors.items():
            if eid < self.num_entities:
                self.sbert_target[eid] = emb.to(self.device)
                count += 1

        # 更新训练索引 (非零行)
        mask = self.sbert_target.abs().sum(dim=1) > 1e-6
        self.train_indices = torch.nonzero(mask).squeeze().to(self.device)
        self.logger.info(
            f"Anchors Updated. Total: {len(self.train_indices)} (+{count} new)")

    def train_local(self, global_weights: Dict[str, torch.Tensor] = None, epochs: int = None):
        """
        执行本地训练
        """
        # 1. 加载全局共享参数
        if global_weights:
            self._load_shared_weights(global_weights)

        # 2. 训练循环
        self.model.train()
        local_epochs = epochs if epochs else self.cfg.fl_local_epochs
        total_loss = 0.0

        for epoch in range(local_epochs):
            self.optimizer.zero_grad()

            # Forward 全图
            # 注意：这里的 self.adj 可能在 CPU (MPS模式下)，model 在 GPU
            # src/models/gcn.py 里的 GCNLayer 已经处理了这种情况
            output = self.model(self.adj)

            # 只计算有锚点的节点的 Loss
            out_batch = output[self.train_indices]
            target_batch = self.sbert_target[self.train_indices]

            # 正样本相似度
            pos_sim = F.cosine_similarity(out_batch, target_batch)

            # 困难负采样 (Hard Negative Mining)
            with torch.no_grad():
                # [N, N]
                sim_mat = torch.mm(F.normalize(out_batch, dim=1),
                                   F.normalize(target_batch, dim=1).T)
                sim_mat.fill_diagonal_(-2.0)
                hard_neg_indices = sim_mat.argmax(dim=1)

            neg_target = target_batch[hard_neg_indices]
            neg_sim = F.cosine_similarity(out_batch, neg_target)

            # Margin Loss
            y = torch.ones_like(pos_sim)
            loss = self.criterion(pos_sim, neg_sim, y)

            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / local_epochs
        return self.model.state_dict(), avg_loss

    def _load_shared_weights(self, global_weights):
        """
        安全加载全局参数，严格保护私有层 (Private GCN)
        """
        my_state = self.model.state_dict()
        loaded_keys = []

        for k, v in global_weights.items():
            if self.cfg.model_arch == 'decoupled' and "struct_encoder" in k:
                continue
            if "initial_features" in k:
                continue

            if k in my_state:
                my_state[k] = v
                loaded_keys.append(k)

        self.model.load_state_dict(my_state)

    @torch.no_grad()
    def get_embeddings(self):
        """推理，获取最终实体向量"""
        self.model.eval()
        # 这里同样利用了 GCNLayer 里的自动兼容逻辑
        return self.model(self.adj).detach().cpu()
