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
        """
        if new_embeddings.device.type != 'cpu':
            new_embeddings = new_embeddings.cpu()
        if not torch.is_tensor(indices):
            indices = torch.tensor(indices)
        if indices.device.type != 'cpu':
            indices = indices.cpu()

        self.anchor_embeddings[indices] = new_embeddings

    def train(self, custom_epochs=None):
        """标准结构训练 (带进度条和早停)"""
        epochs = custom_epochs if custom_epochs is not None else self.cfg.task.federated.local_epochs

        self.model.to(self.device)
        self.model.train()

        optimizer = optim.Adam(self.model.parameters(),
                               lr=self.cfg.task.federated.lr)
        criterion = nn.MarginRankingLoss(margin=self.cfg.task.federated.margin)
        batch_size = self.dm.get_safe_batch_size(
            self.cfg.task.federated.batch_size)

        n_samples = len(self.train_indices)

        # [新增] 从配置中读取是否使用 Hard Mining，默认为 True
        use_hard_mining = self.cfg.task.federated.get('hard_mining', True)
        # 第一次打印日志提示
        if epochs > 0:
            log.info(
                f"   [{self.client_id}] Mining Strategy: {'🔥 Hard Negative' if use_hard_mining else '🎲 Random Negative'}")

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

            pbar = tqdm(range(0, n_samples, batch_size),
                        desc=f"[{self.client_id}] Ep {epoch+1}/{epochs}",
                        leave=False)

            for i in pbar:
                idx = perm[i: i+batch_size]
                batch_ids = self.train_indices[idx].to(self.device)

                # Forward
                output_emb = self.model(self.adj)
                struct_batch = output_emb[batch_ids]

                # Target
                target_batch = self.anchor_embeddings[batch_ids.cpu()].to(
                    self.device)

                # Loss
                pos_sim = F.cosine_similarity(struct_batch, target_batch)

                # --- 负采样策略分支 ---
                if use_hard_mining:
                    # [策略 A] Hard Negative Mining
                    with torch.no_grad():
                        sim_mat = torch.mm(F.normalize(
                            struct_batch), F.normalize(target_batch).T)
                        sim_mat.fill_diagonal_(-2.0)
                        neg_idx = sim_mat.argmax(dim=1)
                else:
                    # [策略 B] Random Negative Mining (In-batch)
                    # 随机生成一个偏移量，使得每个样本选取 batch 内的其他样本作为负例
                    curr_bs = struct_batch.size(0)
                    if curr_bs > 1:
                        shift = torch.randint(
                            1, curr_bs, (curr_bs,), device=self.device)
                        neg_idx = (torch.arange(
                            curr_bs, device=self.device) + shift) % curr_bs
                    else:
                        neg_idx = torch.zeros(
                            curr_bs, dtype=torch.long, device=self.device)

                neg_target = target_batch[neg_idx]
                neg_sim = F.cosine_similarity(struct_batch, neg_target)

                loss = criterion(pos_sim, neg_sim, torch.ones_like(pos_sim))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss_sum += loss.item()
                steps += 1

                pbar.set_postfix({'loss': f"{loss.item():.4f}"})

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
