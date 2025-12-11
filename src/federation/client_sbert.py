# src/core/client_sbert.py
import torch
import torch.nn as nn
import torch.optim as optim
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader
from tqdm import tqdm
import random
import logging

# 引入你的工具类
from src.utils.device_manager import DeviceManager

log = logging.getLogger(__name__)


class ClientSBERT:
    def __init__(self, client_id, cfg, dataset, device_manager: DeviceManager):
        """
        :param client_id: 客户端标识 (C1/C2)
        :param cfg: 全局配置对象 (Hydra DictConfig)
        :param dataset: KGDataset 对象 (来自 src.data.dataset)
        :param device_manager: 设备管理器实例
        """
        self.client_id = client_id
        self.cfg = cfg
        self.dataset = dataset
        self.dm = device_manager

        # [内存优化] 初始化时模型必须在 CPU
        log.info(f"[{client_id}] Initializing SBERT on CPU...")
        self.model = SentenceTransformer(cfg.task.model.name, device='cpu')

        # 训练数据容器
        self.train_pairs = []

    def encode(self, mode='desc'):
        """
        对本地所有实体进行编码。
        :param mode: 'desc' (描述) 或 'polish' (润色)
        :return: (ids, embeddings)
        """
        # 1. 模型上岗 (Move to GPU/MPS)
        self.model.to(self.dm.main_device)
        self.model.eval()

        # 2. 从 Dataset 获取对齐的 ID 和 文本
        ids = self.dataset.ids
        texts = self.dataset.get_text_list(ids, mode=mode)

        # 3. 批量编码
        batch_size = self.dm.get_safe_batch_size(32)  # 编码通常可以用大一点的 Batch
        with torch.no_grad():
            embs = self.model.encode(
                texts,
                batch_size=batch_size,
                convert_to_tensor=True,
                show_progress_bar=True,
                device=self.dm.main_device
            )

        # 4. 结果回传 CPU
        embs = embs.cpu()

        # 5. 模型下岗 (Move back to CPU & Clean)
        if self.dm.is_offload_enabled():
            self.model.to('cpu')
            self.dm.clean_memory()

        return ids, embs

    def prepare_training_data(self, local_indices, target_emb_desc, target_emb_polish=None):
        """
        根据伪标签准备训练数据。支持 Mixed / Polished / Description 模式。
        :param local_indices: 本地实体的索引列表 (对应 dataset.ids)
        :param target_emb_desc: 对端 Description Embedding
        :param target_emb_polish: 对端 Polished Embedding (可选)
        """
        self.train_pairs = []
        mode = self.cfg.task.strategy.text_mode

        # 确保 target 在 CPU，避免占用显存
        if target_emb_desc.device.type != 'cpu':
            target_emb_desc = target_emb_desc.cpu()
        if target_emb_polish is not None and target_emb_polish.device.type != 'cpu':
            target_emb_polish = target_emb_polish.cpu()

        # 获取对应的本地 ID
        local_ids = [self.dataset.ids[i] for i in local_indices]

        # 1. 获取文本
        texts_desc = self.dataset.get_text_list(local_ids, 'desc')
        texts_polish = self.dataset.get_text_list(local_ids, 'polish')

        for i, _ in enumerate(local_ids):
            # --- 策略 A: 加入 Description 样本 ---
            if mode in ['mixed', 'description']:
                target = target_emb_desc[i].detach().clone()
                self.train_pairs.append((texts_desc[i], target))

            # --- 策略 B: 加入 Polished 样本 ---
            if mode in ['mixed', 'polished']:
                # 如果没有 polish target，回退使用 desc target
                target = target_emb_polish[i].detach().clone(
                ) if target_emb_polish is not None else target_emb_desc[i].detach().clone()
                self.train_pairs.append((texts_polish[i], target))

        # 随机打乱，防止 Mixed 模式下模型偏科
        random.shuffle(self.train_pairs)
        # log.info(f"[{self.client_id}] Prepared {len(self.train_pairs)} training pairs ({mode}).")

    def train(self):
        """执行本地微调"""
        if not self.train_pairs:
            return self.model.state_dict(), 0.0

        # 1. 模型上岗
        self.model.to(self.dm.main_device)
        self.model.train()

        # 配置优化器
        transformer = self.model._first_module().auto_model
        optimizer = optim.AdamW(transformer.parameters(),
                                lr=self.cfg.task.federated.lr)
        criterion = nn.MSELoss()

        batch_size = self.dm.get_safe_batch_size(
            self.cfg.task.federated.batch_size)
        total_loss = 0.0

        # 进度条
        pbar = tqdm(range(0, len(self.train_pairs), batch_size),
                    desc=f"[{self.client_id}] Train",
                    leave=False)

        for i in pbar:
            batch = self.train_pairs[i: i + batch_size]
            if not batch:
                continue

            texts = [b[0] for b in batch]
            # 只有当前 Batch 的 Target 才上 GPU
            targets = torch.stack([b[1] for b in batch]
                                  ).to(self.dm.main_device)

            # SBERT Forward
            features = self.model.tokenize(texts)
            for k in features:
                features[k] = features[k].to(self.dm.main_device)

            out = transformer(**features)
            # Mean Pooling
            token_emb = out.last_hidden_state
            mask = features['attention_mask']
            input_mask_expanded = mask.unsqueeze(
                -1).expand(token_emb.size()).float()
            embeddings = torch.sum(token_emb * input_mask_expanded, 1) / \
                torch.clamp(input_mask_expanded.sum(1), min=1e-9)

            loss = criterion(embeddings, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

            # 及时释放 Tensor
            del targets, features, out, embeddings, loss

        # 2. 训练结束，清理战场
        del optimizer
        if self.dm.is_offload_enabled():
            self.model.to('cpu')
            self.dm.clean_memory()

        avg_loss = total_loss / max(1, len(pbar))
        return self.model.state_dict(), avg_loss
