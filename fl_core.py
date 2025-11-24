# 📄 fl_core.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import collections
import copy
import config
from models import get_model_class
from tqdm import tqdm
import logging


class Client:
    def __init__(self, client_id, device, proxies, **kwargs):
        self.client_id = client_id
        self.device = device
        self.model_type = config.MODEL_ARCH

        # 保存本地的代理副本 (可训练)
        self.proxies = proxies.clone().detach().to(self.device)
        self.proxies.requires_grad = True

        ModelClass = get_model_class(self.model_type)

        # --- 图模型初始化 ---
        if self.model_type in ['gcn', 'decoupled']:
            self.adj = kwargs['adj']
            num_entities = kwargs['num_ent']
            local_bert_embs = kwargs['bert']

            self.model = ModelClass(
                num_entities=num_entities,
                feature_dim=config.GCN_DIM,
                hidden_dim=config.GCN_HIDDEN,
                output_dim=config.BERT_DIM,
                dropout=config.GCN_DROPOUT
            ).to(self.device)

            # 准备 SBERT 数据 (作为 Teacher)
            sbert_tensor = torch.zeros(num_entities, config.BERT_DIM)
            self.train_indices = []
            for ent_id, emb in local_bert_embs.items():
                if ent_id < num_entities:
                    sbert_tensor[ent_id] = emb
                    self.train_indices.append(ent_id)

            self.sbert_target = sbert_tensor.to(self.device)
            self.train_indices = torch.tensor(
                self.train_indices).to(self.device)

            logging.info(
                f"Client {client_id}: {len(self.train_indices)} anchors ready for Proxy Alignment.")

    def update_anchors(self, new_targets_dict):
        # 伪标签逻辑保持不变，只是更新 sbert_target
        count = 0
        for ent_id, new_emb in new_targets_dict.items():
            if ent_id < len(self.sbert_target):
                self.sbert_target[ent_id] = new_emb.to(self.device)
                count += 1
        mask = self.sbert_target.abs().sum(dim=1) > 1e-6
        self.train_indices = torch.nonzero(mask).squeeze().to(self.device)
        logging.info(
            f"    [{self.client_id}] Anchors Updated. Total: {len(self.train_indices)} (+{count})")

    def local_train(self, global_model_state, global_proxies, local_epochs, lr):
        """
        返回: (更新后的模型参数, 更新后的代理参数, 平均Loss)
        """
        # 1. 加载全局模型参数
        if global_model_state is not None:
            my_state = self.model.state_dict()
            for k, v in global_model_state.items():
                if "initial_features" in k:
                    continue
                if self.model_type == 'decoupled' and "struct_encoder" in k:
                    continue
                my_state[k] = v
            self.model.load_state_dict(my_state)

        # 2. 加载全局代理参数
        if global_proxies is not None:
            self.proxies.data = global_proxies.to(self.device)

        # 3. 训练
        if self.model_type in ['gcn', 'decoupled']:
            return self._train_proxy_alignment(local_epochs, lr)
        else:
            raise NotImplementedError("Proxy mode only supports GCN/Decoupled")

    def _train_proxy_alignment(self, epochs, lr):
        self.model.train()

        # 1. 优化器 (保持不变)
        optimizer = optim.Adam([
            {'params': self.model.parameters(), 'lr': lr},
            {'params': [self.proxies], 'lr': config.PROXY_LR}
        ])

        # 2. 定义两个 Loss
        # (A) 主 Loss: MarginRankingLoss (找回丢失的精度)
        criterion_rank = nn.MarginRankingLoss(margin=config.FL_MARGIN)
        # (B) 辅 Loss: KLDivLoss (保留动态代理的调节能力)
        criterion_kl = nn.KLDivLoss(reduction='batchmean')

        temp = config.PROXY_TEMPERATURE
        # 混合权重: 主要是 Ranking，KL 作为辅助
        LAMBDA_KL = 0.05

        total_loss = 0.0

        for epoch in range(epochs):
            optimizer.zero_grad()

            # --- Forward ---
            gcn_out = self.model(self.adj)
            student_emb = gcn_out[self.train_indices]
            # SBERT 仍然是硬锚点
            teacher_emb = self.sbert_target[self.train_indices].detach()

            # --- Loss A: Margin Ranking (复刻原方案的逻辑) ---
            # 正例相似度
            pos_sim = F.cosine_similarity(student_emb, teacher_emb)

            # 硬负采样 (Batch 内)
            with torch.no_grad():
                sim_mat = torch.mm(F.normalize(student_emb, dim=1),
                                   F.normalize(teacher_emb, dim=1).T)
                sim_mat.fill_diagonal_(-2.0)
                hard_neg_indices = sim_mat.argmax(dim=1)

            neg_target = teacher_emb[hard_neg_indices]
            neg_sim = F.cosine_similarity(student_emb, neg_target)

            y_target = torch.ones_like(pos_sim)
            loss_rank = criterion_rank(pos_sim, neg_sim, y_target)

            # --- Loss B: Proxy KL Divergence (动态代理部分) ---
            norm_proxies = F.normalize(self.proxies, dim=1)
            norm_student = F.normalize(student_emb, dim=1)
            norm_teacher = F.normalize(teacher_emb, dim=1)

            # Student 分布
            student_logits = torch.mm(norm_student, norm_proxies.T) / temp
            student_log_probs = F.log_softmax(student_logits, dim=1)

            # Teacher 分布
            with torch.no_grad():
                teacher_logits = torch.mm(norm_teacher, norm_proxies.T) / temp
                teacher_probs = F.softmax(teacher_logits, dim=1)

            loss_kl = criterion_kl(student_log_probs, teacher_probs)

            # --- Total Loss ---
            # 结合两者
            loss = loss_rank + (LAMBDA_KL * loss_kl)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        return self.model.state_dict(), self.proxies.detach().cpu(), total_loss / max(1, epochs)


class Server:
    def __init__(self, initial_proxies):
        self.device = config.DEVICE
        ModelClass = get_model_class(config.MODEL_ARCH)

        # 初始化全局模型
        self.global_model = ModelClass(
            1, config.GCN_DIM, config.GCN_HIDDEN, config.BERT_DIM, 0
        ).to(self.device)

        # 初始化全局代理
        self.global_proxies = initial_proxies.to(self.device)
        logging.info(
            f"Server initialized with {len(self.global_proxies)} dynamic proxies.")

    def get_global_model_state(self):
        return self.global_model.state_dict()

    def get_global_proxies(self):
        return self.global_proxies

    def aggregate(self, client_models, client_proxies):
        """
        同时聚合 模型参数 和 代理参数
        """
        # 1. 聚合模型 (FedAvg)
        avg_weights = collections.OrderedDict()

        # 遍历全局模型的所有参数键
        for key in self.global_model.state_dict().keys():
            # 过滤掉私有层
            if "initial_features" in key:
                continue
            if config.MODEL_ARCH == 'decoupled' and "struct_encoder" in key:
                continue

            # 收集各客户端的该参数
            tensors = [s[key].to(self.device)
                       for s in client_models if key in s]

            if tensors:
                # 【关键修复】检查数据类型
                if torch.is_floating_point(tensors[0]):
                    # 浮点数直接求平均
                    avg_weights[key] = torch.stack(tensors).mean(dim=0)
                else:
                    # 整数 (如 num_batches_tracked) 需要先转 float 再转回 long
                    avg_weights[key] = torch.stack(
                        tensors).float().mean(dim=0).long()

        # 加载聚合后的参数到全局模型
        my_state = self.global_model.state_dict()
        my_state.update(avg_weights)
        self.global_model.load_state_dict(my_state)

        # 2. 聚合代理 (FedAvg)
        # 代理向量本身是 float，直接求平均即可
        stacked_proxies = torch.stack(
            [p.to(self.device) for p in client_proxies])
        new_proxies = stacked_proxies.mean(dim=0)

        # 计算代理移动了多少
        diff = torch.norm(new_proxies - self.global_proxies).item()
        self.global_proxies = new_proxies

        return avg_weights, new_proxies, diff
