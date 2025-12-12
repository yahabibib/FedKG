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

        # 1. 构建邻接矩阵 (结构特征源)
        # 注意：为了节省显存和兼容性，adj 建议常驻 CPU，计算时再视情况处理
        self.adj = build_adjacency_matrix(
            dataset.triples,
            dataset.num_entities,
            device='cpu'
        )

        # 2. 加载 SBERT (语义锚点源)
        # 从配置中读取 Phase 1 保存的模型路径
        sbert_path = cfg.task.sbert_checkpoint
        log.info(f"[{client_id}] Loading Frozen SBERT from: {sbert_path}")
        self.sbert = SentenceTransformer(sbert_path, device='cpu')
        self.sbert.eval()  # 永远冻结

        # 3. 预计算所有实体的 SBERT Embedding (作为固定的训练目标)
        # 这样训练 GCN 时就不用反复跑 BERT 了，极大节省显存和时间
        self.anchor_embeddings = self._precompute_anchors()

        # 4. 初始化结构模型 (GCN + MLP)
        self.model = DecoupledModel(cfg.task.model, dataset.num_entities)

        # 训练索引 (所有有文本描述的实体都可以作为训练集)
        # 我们可以过滤掉没有描述的实体，或者对它们使用名字嵌入
        self.train_indices = torch.arange(dataset.num_entities)

    def _precompute_anchors(self):
        """预计算并缓存 SBERT 语义向量"""
        log.info(f"[{self.client_id}] Pre-computing semantic anchors...")

        # 获取所有文本 (优先使用 description)
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

        # 算完赶紧把 SBERT 踢下线，释放显存给 GCN 用
        self.sbert.to('cpu')
        self.dm.clean_memory()

        return embs.cpu()  # 存在 CPU 上，训练时按需取

    def train(self):
        """训练 GCN 结构模型"""
        # 1. 模型上岗
        self.model.to(self.device)
        self.model.train()

        # 优化器 (只优化 GCN 和 MLP)
        optimizer = optim.Adam(self.model.parameters(),
                               lr=self.cfg.task.federated.lr)

        # 损失函数 (Margin Ranking Loss)
        criterion = nn.MarginRankingLoss(margin=self.cfg.task.federated.margin)

        epochs = self.cfg.task.federated.local_epochs
        batch_size = self.dm.get_safe_batch_size(
            self.cfg.task.federated.batch_size)

        # 将数据转为 Tensor
        train_indices = self.train_indices
        n_samples = len(train_indices)

        total_loss = 0.0

        # 简单的 Epoch 循环
        for epoch in range(epochs):
            # Shuffle
            perm = torch.randperm(n_samples)

            pbar = tqdm(range(0, n_samples, batch_size),
                        desc=f"[{self.client_id}] Struct Train Ep{epoch+1}",
                        leave=False)

            for i in pbar:
                idx = perm[i: i+batch_size]
                batch_ids = train_indices[idx].to(self.device)

                # A. 前向传播
                # 这里 adj 在 CPU，GCN 内部会自动处理设备传输
                output_emb = self.model(self.adj)

                # 取出当前 Batch 的结构向量
                struct_batch = output_emb[batch_ids]

                # 取出对应的 SBERT 锚点 (目标)
                target_batch = self.anchor_embeddings[batch_ids.cpu()].to(
                    self.device)

                # B. 计算相似度 (正样本)
                pos_sim = F.cosine_similarity(struct_batch, target_batch)

                # C. 困难负采样 (Hard Negative Mining)
                # 在当前 Batch 内寻找最难区分的负样本
                with torch.no_grad():
                    # Batch 内相似度矩阵
                    sim_mat = torch.mm(F.normalize(
                        struct_batch), F.normalize(target_batch).T)
                    # 屏蔽对角线 (正样本)
                    sim_mat.fill_diagonal_(-2.0)
                    # 找到每行最大的 (最像的负样本)
                    hard_neg_idx = sim_mat.argmax(dim=1)

                neg_target = target_batch[hard_neg_idx]
                neg_sim = F.cosine_similarity(struct_batch, neg_target)

                # D. Loss
                y = torch.ones_like(pos_sim)
                loss = criterion(pos_sim, neg_sim, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        # 2. 训练结束，下岗
        if self.dm.is_offload_enabled():
            self.model.to('cpu')
            self.dm.clean_memory()

        return self.model.get_shared_state_dict(), total_loss / (epochs * len(pbar))

    def get_embeddings(self):
        """推理：获取最终的结构 Embedding (用于评估)"""
        self.model.to(self.device)
        self.model.eval()
        with torch.no_grad():
            embs = self.model(self.adj)
        self.model.to('cpu')
        return embs.cpu()

    def update_anchors(self, indices, new_embeddings):
        """
        更新本地训练的锚点目标 (Self-training update)
        :param indices: 需要更新的实体索引列表 (List or Tensor)
        :param new_embeddings: 新的目标向量 (Tensor)
        """
        # 确保数据在设备上
        if isinstance(indices, list):
            indices = torch.tensor(indices, device=self.device)
        else:
            indices = indices.to(self.device)

        new_embeddings = new_embeddings.to(self.device)

        # 更新 anchor_embeddings (注意: anchor_embeddings 初始是在 CPU 的，这里要看你训练时的策略)
        # 如果 self.anchor_embeddings 在 CPU，这里需要 copy 回去，或者训练时临时覆盖
        # 建议方案：维护一个 mask 或 update 字典

        # 简单移植 main 分支逻辑 (假设 anchor_embeddings 已上载到 GPU 或支持索引更新)
        # 注意：原 main 分支是直接修改 self.sbert_target
        # 这里需要适配:
        self.anchor_embeddings = self.anchor_embeddings.to(
            self.device)  # 确保在修改前在同一设备
        self.anchor_embeddings[indices] = new_embeddings

        # 重新生成训练索引 (所有非零/有效的 anchor 都可以作为训练数据)
        # 或者直接将 indices 加入 train_indices
        log.info(
            f"[{self.client_id}] Updated {len(indices)} anchors via pseudo-labels.")
