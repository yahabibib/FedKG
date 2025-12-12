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

        # 1. 构建邻接矩阵 (CPU)
        self.adj = build_adjacency_matrix(
            dataset.triples,
            dataset.num_entities,
            device='cpu'
        )

        # 2. 加载 Frozen SBERT (CPU)
        sbert_path = cfg.task.sbert_checkpoint
        log.info(f"[{client_id}] Loading Frozen SBERT from: {sbert_path}")
        self.sbert = SentenceTransformer(sbert_path, device='cpu')
        self.sbert.eval()

        # 3. 预计算 SBERT Anchors
        # 建议这里加一个简单的缓存检测，避免每次重启都算一遍 (可选优化)
        self.anchor_embeddings = self._precompute_anchors()

        # 4. 初始化模型
        self.model = DecoupledModel(cfg.task.model, dataset.num_entities)
        self.train_indices = torch.arange(dataset.num_entities)

    def _precompute_anchors(self):
        # ... (保持原有的预计算逻辑不变) ...
        # 为节省篇幅省略，保持你之前的代码即可
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
        [关键修复] 更新本地锚点 (Self-training)
        """
        # 确保 new_embeddings 在 CPU (因为 self.anchor_embeddings 在 CPU)
        if new_embeddings.device.type != 'cpu':
            new_embeddings = new_embeddings.cpu()

        # indices 无论传进来是什么，都转成 tensor 用于索引
        if not torch.is_tensor(indices):
            indices = torch.tensor(indices)
        if indices.device.type != 'cpu':
            indices = indices.cpu()

        # 原地更新 (In-place update)
        self.anchor_embeddings[indices] = new_embeddings
        # log.info(f"[{self.client_id}] Anchors updated for {len(indices)} entities.")

    def train(self, custom_epochs=None):
        """
        训练 GCN
        :param custom_epochs: 如果传入，则覆盖 config 中的 local_epochs
        """
        # 1. 确定 Epochs
        epochs = custom_epochs if custom_epochs is not None else self.cfg.task.federated.local_epochs

        self.model.to(self.device)
        self.model.train()

        optimizer = optim.Adam(self.model.parameters(),
                               lr=self.cfg.task.federated.lr)
        criterion = nn.MarginRankingLoss(margin=self.cfg.task.federated.margin)
        batch_size = self.dm.get_safe_batch_size(
            self.cfg.task.federated.batch_size)

        n_samples = len(self.train_indices)
        total_loss = 0.0

        # --- 早停策略参数 (Early Stopping Config) ---
        stop_threshold = 0.08  # 当 loss 低于这个值时开始监测
        patience = 3           # 容忍几次不下降
        min_delta = 0.005      # 最小下降幅度
        early_stop_counter = 0
        prev_epoch_loss = float('inf')

        # 进度条
        pbar_epoch = range(epochs)

        for epoch in pbar_epoch:
            # Shuffle
            perm = torch.randperm(n_samples)
            epoch_loss_sum = 0.0
            steps = 0

            # Batch Loop (为了日志简洁，这里不给每个batch都打进度条了，只显示Epoch进度)
            for i in range(0, n_samples, batch_size):
                idx = perm[i: i+batch_size]
                batch_ids = self.train_indices[idx].to(self.device)

                # A. GCN Forward
                output_emb = self.model(self.adj)
                struct_batch = output_emb[batch_ids]

                # B. Target (SBERT/Pseudo)
                target_batch = self.anchor_embeddings[batch_ids.cpu()].to(
                    self.device)

                # C. Loss Calculation
                pos_sim = F.cosine_similarity(struct_batch, target_batch)

                # Hard Negative Mining
                with torch.no_grad():
                    sim_mat = torch.mm(F.normalize(
                        struct_batch), F.normalize(target_batch).T)
                    sim_mat.fill_diagonal_(-2.0)
                    hard_neg_idx = sim_mat.argmax(dim=1)

                neg_target = target_batch[hard_neg_idx]
                neg_sim = F.cosine_similarity(struct_batch, neg_target)

                y = torch.ones_like(pos_sim)
                loss = criterion(pos_sim, neg_sim, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss_sum += loss.item()
                steps += 1

            # --- Epoch End Logic ---
            avg_loss = epoch_loss_sum / max(1, steps)

            # 这里的 print 可以根据喜好改为 tqdm.set_postfix
            # log.info(f"[{self.client_id}] Ep {epoch+1}/{epochs} | Loss: {avg_loss:.4f}")

            # --- 自动早停检查 ---
            if avg_loss < stop_threshold:
                # 检查是否还有显著下降
                if (prev_epoch_loss - avg_loss) < min_delta:
                    early_stop_counter += 1
                    if early_stop_counter >= patience:
                        log.info(
                            f"   🛑 [{self.client_id}] Early stopping at Epoch {epoch+1} (Loss={avg_loss:.4f})")
                        total_loss = avg_loss  # 更新为当前loss
                        break
                else:
                    early_stop_counter = 0  # Loss 还在降，重置计数器

            prev_epoch_loss = avg_loss
            total_loss = avg_loss

        # 2. 清理
        if self.dm.is_offload_enabled():
            self.model.to('cpu')
            self.dm.clean_memory()

        return self.model.get_shared_state_dict(), total_loss

    def get_embeddings(self):
        """推理：获取最终的结构 Embedding (用于评估)"""
        self.model.to(self.device)
        self.model.eval()
        with torch.no_grad():
            embs = self.model(self.adj)
        self.model.to('cpu')
        return embs.cpu()
