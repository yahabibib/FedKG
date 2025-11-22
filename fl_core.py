# 📄 fl_core.py
# 存放联邦学习的核心类 (Client, Server)
# 【修复版】完全适配 Decoupled 架构，修复 Server 初始化参数缺失问题

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import collections
import config
from models import get_model_class
from tqdm import tqdm


class Client:
    def __init__(self, client_id, device, **kwargs):
        self.client_id = client_id
        self.device = device
        self.model_type = config.MODEL_ARCH

        ModelClass = get_model_class(self.model_type)

        # --- 图模型初始化 (GCN 或 Decoupled) ---
        # 【关键修改】将 decoupled 加入判断
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

            sbert_tensor = torch.zeros(num_entities, config.BERT_DIM)
            self.train_indices = []
            for ent_id, emb in local_bert_embs.items():
                if ent_id < num_entities:
                    sbert_tensor[ent_id] = emb
                    self.train_indices.append(ent_id)
            self.sbert_target = sbert_tensor.to(self.device)
            self.train_indices = torch.tensor(
                self.train_indices).to(self.device)
            print(
                f"Client {client_id} ({self.model_type}): {len(self.train_indices)} anchors.")

        # --- 向量投影初始化 (Projection) ---
        elif self.model_type == 'projection':
            local_transe_embs = kwargs['transe']
            local_bert_embs = kwargs['bert']

            self.model = ModelClass(
                input_dim=config.TRANSE_DIM,
                output_dim=config.BERT_DIM
            ).to(self.device)

            self.train_data = []
            for ent_id, transe_emb in local_transe_embs.items():
                if ent_id in local_bert_embs:
                    self.train_data.append(
                        (transe_emb.to(device),
                         local_bert_embs[ent_id].to(device))
                    )
            print(f"Client {client_id} (MLP): {len(self.train_data)} pairs.")

    def update_anchors(self, new_targets_dict):
        count = 0
        for ent_id, new_emb in new_targets_dict.items():
            if ent_id < len(self.sbert_target):
                self.sbert_target[ent_id] = new_emb.to(self.device)
                count += 1
        mask = self.sbert_target.abs().sum(dim=1) > 1e-6
        self.train_indices = torch.nonzero(mask).squeeze().to(self.device)
        print(
            f"    [{self.client_id}] Anchors Updated. Total: {len(self.train_indices)} (+{count})")

    def local_train(self, global_model_state, local_epochs, batch_size, lr):
        # 1. 加载全局参数
        if global_model_state is not None:
            my_state = self.model.state_dict()
            for k, v in global_model_state.items():
                # 【过滤逻辑】保护私有参数
                if "initial_features" in k:
                    continue
                if self.model_type == 'decoupled' and "struct_encoder" in k:
                    continue

                my_state[k] = v
            self.model.load_state_dict(my_state)

        # 2. 分发训练逻辑
        if self.model_type in ['gcn', 'decoupled']:
            return self._train_graph_model(local_epochs, lr)
        elif self.model_type == 'projection':
            return self._train_projection(local_epochs, batch_size, lr)

    def _train_graph_model(self, epochs, lr):
        self.model.train()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.MarginRankingLoss(margin=config.FL_MARGIN)

        pbar = tqdm(range(epochs), desc=f"[{self.client_id}]", leave=False)
        total_loss = 0.0

        for epoch in pbar:
            optimizer.zero_grad()
            output = self.model(self.adj)
            out_batch = output[self.train_indices]
            target_batch = self.sbert_target[self.train_indices]

            pos_sim = F.cosine_similarity(out_batch, target_batch)

            # 困难负采样
            with torch.no_grad():
                sim_mat = torch.mm(F.normalize(out_batch, dim=1),
                                   F.normalize(target_batch, dim=1).T)
                sim_mat.fill_diagonal_(-2.0)
                hard_neg_indices = sim_mat.argmax(dim=1)
            neg_target = target_batch[hard_neg_indices]
            neg_sim = F.cosine_similarity(out_batch, neg_target)

            y = torch.ones_like(pos_sim)
            loss = criterion(pos_sim, neg_sim, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        return self.model.state_dict(), total_loss / max(1, epochs)

    def _train_projection(self, epochs, batch_size, lr):
        self.model.train()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.MarginRankingLoss(margin=config.FL_MARGIN)

        if not self.train_data:
            return self.model.state_dict(), 0.0

        total_loss = 0.0
        batch_count = 0

        for epoch in range(epochs):
            random.shuffle(self.train_data)
            for i in range(0, len(self.train_data), batch_size):
                batch = self.train_data[i: i + batch_size]
                if len(batch) < 2:
                    continue

                transe_batch = torch.stack([b[0] for b in batch])
                bert_batch = torch.stack([b[1] for b in batch])

                proj_transe = self.model(transe_batch)
                pos_sim = F.cosine_similarity(proj_transe, bert_batch)

                bert_neg_batch = torch.roll(bert_batch, shifts=1, dims=0)
                neg_sim = F.cosine_similarity(proj_transe, bert_neg_batch)

                loss = criterion(pos_sim, neg_sim, torch.ones_like(pos_sim))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                batch_count += 1

        return self.model.state_dict(), total_loss / max(1, batch_count)


class Server:
    def __init__(self):
        self.device = config.DEVICE
        ModelClass = get_model_class(config.MODEL_ARCH)

        # 【关键修复】将 decoupled 也纳入图模型初始化逻辑
        if config.MODEL_ARCH in ['gcn', 'decoupled']:
            self.global_model = ModelClass(
                1, config.GCN_DIM, config.GCN_HIDDEN, config.BERT_DIM, 0
            ).to(self.device)
        else:
            self.global_model = ModelClass(
                config.TRANSE_DIM, config.BERT_DIM
            ).to(self.device)

    def get_global_model_state(self):
        return self.global_model.state_dict()

    def aggregate_models(self, client_model_states):
        if not client_model_states:
            return None

        avg_weights = collections.OrderedDict()

        for key in self.global_model.state_dict().keys():
            # 1. 过滤私有节点特征
            if "initial_features" in key:
                continue

            # 2. 【解耦逻辑】过滤私有结构编码器
            if config.MODEL_ARCH == 'decoupled' and "struct_encoder" in key:
                continue

            tensors = [s[key].to(self.device)
                       for s in client_model_states if key in s]
            if not tensors:
                continue

            if torch.is_floating_point(tensors[0]):
                avg_weights[key] = torch.stack(tensors).mean(dim=0)
            else:
                avg_weights[key] = torch.stack(
                    tensors).float().mean(dim=0).long()

        my_state = self.global_model.state_dict()
        my_state.update(avg_weights)
        self.global_model.load_state_dict(my_state)
        return avg_weights
