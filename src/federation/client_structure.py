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


def info_nce_loss(features, targets, temperature=0.07):
    """
    [InfoNCE Loss]
    features: Structure Prediction [B, D]
    targets: SBERT Ground Truth [B, D]
    temperature: 越小越尖锐，推荐 0.05-0.1
    """
    logits = torch.mm(features, targets.T)
    logits /= temperature
    labels = torch.arange(features.size(0)).to(features.device)
    return F.cross_entropy(logits, labels)


class ClientStructure:
    def __init__(self, client_id, cfg, dataset, device_manager):
        self.client_id = client_id
        self.cfg = cfg
        self.dataset = dataset
        self.dm = device_manager
        self.device = self.dm.main_device

        # 1. 构建图
        self.adj, self.edge_types, self.num_rels = build_adjacency_matrix(
            dataset.triples, dataset.num_entities, device='cpu', return_edge_types=True
        )

        # 2. 加载 SBERT (Teacher)
        sbert_path = cfg.task.sbert_checkpoint
        log.info(f"[{client_id}] Loading Frozen SBERT: {sbert_path}")
        self.sbert = SentenceTransformer(sbert_path, device='cpu').eval()

        # 3. 预计算 Anchors
        self.anchor_embeddings = self._precompute_anchors()

        # 4. 初始化模型 (Student)
        self.model = DecoupledModel(
            cfg.task.model, dataset.num_entities, self.num_rels)

        # 训练集索引 (全量)
        self.train_indices = torch.arange(dataset.num_entities)

    def _precompute_anchors(self):
        log.info(f"[{self.client_id}] Pre-computing SBERT anchors...")
        texts = self.dataset.get_text_list(self.dataset.ids, 'desc')
        self.sbert.to(self.device)
        with torch.no_grad():
            embs = self.sbert.encode(
                texts, batch_size=512, convert_to_tensor=True,
                show_progress_bar=False, device=self.device
            )
        self.sbert.to('cpu')
        return embs.cpu()

    def update_anchors(self, indices, new_embeddings):
        """
        [修复版] 接收来自对端的结构化伪标签，更新本地目标
        兼容 List 和 Tensor 类型的 indices
        """
        # 1. 确保 Embedding 在 CPU
        if isinstance(new_embeddings, torch.Tensor):
            new_embeddings = new_embeddings.cpu()

        # 2. 确保 indices 是 Tensor (以便后续索引操作)
        if isinstance(indices, list):
            indices = torch.tensor(indices, dtype=torch.long)

        if isinstance(indices, torch.Tensor):
            indices = indices.cpu()

        # 3. 执行更新
        self.anchor_embeddings[indices] = new_embeddings

    def calc_internal_fidelity(self):
        """
        计算内部对齐度 (Internal Fidelity)
        衡量 Structure Encoder 对本地 SBERT 知识的吸收程度。
        """
        self.model.to(self.device).eval()
        batch_size = 2048
        total_sim = 0.0
        n_batches = 0

        with torch.no_grad():
            struct_full = self.model(self.adj, self.edge_types)  # [N, D]

            num_nodes = struct_full.shape[0]
            for i in range(0, num_nodes, batch_size):
                end = min(i + batch_size, num_nodes)

                s_emb = F.normalize(struct_full[i:end], p=2, dim=1)
                t_emb = F.normalize(
                    self.anchor_embeddings[i:end].to(self.device), p=2, dim=1)

                sim = F.cosine_similarity(s_emb, t_emb).mean()
                total_sim += sim.item()
                n_batches += 1

        self.model.to('cpu')
        return total_sim / max(1, n_batches)

    def train(self, custom_epochs=None):
        """
        纯结构训练 (Pure Structure Training)
        """
        epochs = custom_epochs if custom_epochs is not None else self.cfg.task.federated.local_epochs
        temperature = self.cfg.task.federated.get('temperature', 0.07)
        batch_size = self.dm.get_safe_batch_size(
            self.cfg.task.federated.batch_size)

        self.model.to(self.device).train()
        optimizer = optim.Adam(self.model.parameters(),
                               lr=self.cfg.task.federated.lr)

        if epochs > 0:
            log.info(
                f"   [{self.client_id}] Strategy: InfoNCE (Tau={temperature}) | Target: SBERT/Peer-Struct")

        total_loss = 0.0
        n_samples = len(self.train_indices)

        for epoch in range(epochs):
            perm = torch.randperm(n_samples)
            epoch_loss = 0.0
            steps = 0

            pbar = tqdm(range(0, n_samples, batch_size),
                        desc=f"Ep {epoch+1}/{epochs}", leave=False)
            for i in pbar:
                idx = perm[i:i+batch_size]
                batch_ids = self.train_indices[idx].to(self.device)

                # Forward (Pure Structure)
                output_emb = self.model(self.adj, self.edge_types)
                struct_batch = F.normalize(output_emb[batch_ids], p=2, dim=1)

                # Target (Anchors)
                target_batch = F.normalize(
                    self.anchor_embeddings[batch_ids.cpu()].to(self.device), p=2, dim=1)

                # InfoNCE Loss
                loss = info_nce_loss(struct_batch, target_batch, temperature)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                steps += 1
                pbar.set_postfix({'loss': f"{loss.item():.4f}"})

            total_loss = epoch_loss / max(1, steps)

        # 计算 Fidelity
        fidelity_score = self.calc_internal_fidelity()

        if self.dm.is_offload_enabled():
            self.model.to('cpu')
            self.dm.clean_memory()

        return self.model.get_shared_state_dict(), total_loss, fidelity_score
