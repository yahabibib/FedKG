# src/federation/client_structure.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sentence_transformers import SentenceTransformer
import logging
from tqdm import tqdm

from src.models.decoupled import DecoupledModel
from src.utils.graph import build_adjacency_matrix
from src.utils.device_manager import DeviceManager

log = logging.getLogger(__name__)


class ClientStructure:
    def __init__(self, client_id, cfg, dataset, device_manager: DeviceManager):
        self.client_id = client_id
        self.cfg = cfg
        self.dataset = dataset
        self.dm = device_manager
        self.device = self.dm.main_device

        # 1. 邻接矩阵 (CPU)
        self.adj = build_adjacency_matrix(
            dataset.triples,
            dataset.num_entities,
            device='cpu'
        )

        # 2. Frozen SBERT (用于初始化和语义一致性检查)
        sbert_path = cfg.task.sbert_checkpoint
        log.info(f"[{client_id}] Loading Frozen SBERT from: {sbert_path}")
        self.sbert = SentenceTransformer(sbert_path, device='cpu')
        self.sbert.eval()

        # 3. 预计算 SBERT Anchors (作为初始训练目标)
        self.anchor_embeddings = self._precompute_anchors()

        # 4. 初始化模型
        self.model = DecoupledModel(cfg.task.model, dataset.num_entities)

        # 训练索引：初始时所有有 SBERT 的实体都是训练集
        self.train_indices = torch.arange(dataset.num_entities)

    def _precompute_anchors(self):
        log.info(f"[{self.client_id}] Pre-computing semantic anchors...")
        ids = self.dataset.ids
        texts = self.dataset.get_text_list(ids, mode='desc')

        self.sbert.to(self.device)
        with torch.no_grad():
            embs = self.sbert.encode(
                texts,
                batch_size=self.dm.get_safe_batch_size(64),
                convert_to_tensor=True,
                show_progress_bar=True,
                device=self.device
            )
        self.sbert.to('cpu')
        self.dm.clean_memory()
        return embs.cpu()

    def update_anchors(self, indices, new_embeddings):
        """
        [互学习核心] 更新本地锚点
        :param indices: 需要更新的实体 ID 列表
        :param new_embeddings: 新的目标向量 (来自 Peer GCN)
        """
        if new_embeddings.device.type != 'cpu':
            new_embeddings = new_embeddings.cpu()
        if not torch.is_tensor(indices):
            indices = torch.tensor(indices)
        if indices.device.type != 'cpu':
            indices = indices.cpu()

        # 直接覆盖：这意味着我们相信 Peer 的结构推断优于初始的 SBERT
        self.anchor_embeddings[indices] = new_embeddings
        # log.info(f"[{self.client_id}] Updated anchors for {len(indices)} entities.")

    def train(self, custom_epochs=None):
        """标准 GCN 训练 (带早停)"""
        # 支持外部传入动态 Epochs
        epochs = custom_epochs if custom_epochs is not None else self.cfg.task.federated.local_epochs

        self.model.to(self.device)
        self.model.train()

        optimizer = optim.Adam(self.model.parameters(),
                               lr=self.cfg.task.federated.lr)
        criterion = nn.MarginRankingLoss(margin=self.cfg.task.federated.margin)
        batch_size = self.dm.get_safe_batch_size(
            self.cfg.task.federated.batch_size)

        n_samples = len(self.train_indices)

        # 早停参数
        stop_threshold = 0.08
        patience = 3
        min_delta = 0.005
        early_stop_counter = 0
        prev_epoch_loss = float('inf')
        total_loss_record = 0.0

        for epoch in range(epochs):
            perm = torch.randperm(n_samples)
            epoch_loss_sum = 0.0
            steps = 0

            for i in range(0, n_samples, batch_size):
                idx = perm[i: i+batch_size]
                batch_ids = self.train_indices[idx].to(self.device)

                # Forward
                output_emb = self.model(self.adj)
                struct_batch = output_emb[batch_ids]

                # Target: 这里使用的是 self.anchor_embeddings
                # 在互学习模式下，这部分 target 会随着轮次更新为 Peer 的 Structure Output
                target_batch = self.anchor_embeddings[batch_ids.cpu()].to(
                    self.device)

                # Loss
                pos_sim = F.cosine_similarity(struct_batch, target_batch)

                with torch.no_grad():
                    sim_mat = torch.mm(F.normalize(
                        struct_batch), F.normalize(target_batch).T)
                    sim_mat.fill_diagonal_(-2.0)
                    hard_neg_idx = sim_mat.argmax(dim=1)
                neg_target = target_batch[hard_neg_idx]
                neg_sim = F.cosine_similarity(struct_batch, neg_target)

                loss = criterion(pos_sim, neg_sim, torch.ones_like(pos_sim))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss_sum += loss.item()
                steps += 1

            avg_loss = epoch_loss_sum / max(1, steps)

            # 早停检查
            if avg_loss < stop_threshold:
                if (prev_epoch_loss - avg_loss) < min_delta:
                    early_stop_counter += 1
                    if early_stop_counter >= patience:
                        log.info(
                            f"   🛑 [{self.client_id}] Early stop @ Ep {epoch+1} (Loss={avg_loss:.4f})")
                        total_loss_record = avg_loss
                        break
                else:
                    early_stop_counter = 0

            prev_epoch_loss = avg_loss
            total_loss_record = avg_loss

        if self.dm.is_offload_enabled():
            self.model.to('cpu')
            self.dm.clean_memory()

        return self.model.get_shared_state_dict(), total_loss_record

    def get_embeddings(self):
        self.model.to(self.device)
        self.model.eval()
        with torch.no_grad():
            embs = self.model(self.adj)
        self.model.to('cpu')
        return embs.cpu()
