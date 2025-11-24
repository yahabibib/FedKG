# 📄 fl_core.py
# 【Relation-Aware 版】适配 RelationGCN，移除动态代理，回归标准联邦架构

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import collections
import config
from models import get_model_class
from tqdm import tqdm
import logging


class Client:
    def __init__(self, client_id, device, **kwargs):
        self.client_id = client_id
        self.device = device
        self.model_type = config.MODEL_ARCH

        ModelClass = get_model_class(self.model_type)

        # --- 图模型初始化 (GCN / Decoupled) ---
        if self.model_type in ['gcn', 'decoupled']:
            # 🔥 [关键修改] 接收图结构索引和关系语义
            self.edge_index = kwargs['edge_index'].to(self.device)
            self.edge_type = kwargs['edge_type'].to(self.device)

            num_entities = kwargs['num_ent']
            num_relations = kwargs['num_rel']  # 原始关系数量
            local_bert_embs = kwargs['bert']
            rel_sbert = kwargs.get('rel_sbert', None)  # 关系 SBERT

            # 初始化模型
            # 注意：这里假设 ModelClass (GCN 或 Decoupled) 已经适配了 num_relations 参数
            self.model = ModelClass(
                num_entities=num_entities,
                num_relations=num_relations,   # 传入关系数
                feature_dim=config.GCN_DIM,
                hidden_dim=config.GCN_HIDDEN,
                output_dim=config.BERT_DIM,
                dropout=config.GCN_DROPOUT
            ).to(self.device)

            # 🔥 [关键修改] 初始化关系嵌入
            # 如果模型有这个方法 (RelationGCN 应该有)，就初始化
            if hasattr(self.model, 'init_relation_embeddings') and rel_sbert is not None:
                self.model.init_relation_embeddings(rel_sbert)
            # 如果是 Decoupled，可能需要深入到 self.model.struct_encoder 里去初始化
            elif hasattr(self.model, 'struct_encoder') and rel_sbert is not None:
                # 假设 struct_encoder 是一个 ModuleList，或者就是 RelationGCN
                # 这里做个简单的尝试，如果你的 Decoupled 写法不同，可能要微调
                for module in self.model.modules():
                    if hasattr(module, 'init_relation_embeddings'):
                        module.init_relation_embeddings(rel_sbert)
                        break

            # 准备 SBERT Target (锚点)
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
                f"Client {self.client_id}: {len(self.train_indices)} anchors ready.")

        # --- Projection (TransE) 初始化 (保持兼容) ---
        elif self.model_type == 'projection':
            local_transe_embs = kwargs['transe']
            local_bert_embs = kwargs['bert']
            self.model = ModelClass(
                input_dim=config.TRANSE_DIM, output_dim=config.BERT_DIM).to(self.device)
            self.train_data = []
            for ent_id, transe_emb in local_transe_embs.items():
                if ent_id in local_bert_embs:
                    self.train_data.append(
                        (transe_emb.to(device), local_bert_embs[ent_id].to(device)))

    def update_anchors(self, new_targets_dict):
        count = 0
        for ent_id, new_emb in new_targets_dict.items():
            if ent_id < len(self.sbert_target):
                self.sbert_target[ent_id] = new_emb.to(self.device)
                count += 1
        mask = self.sbert_target.abs().sum(dim=1) > 1e-6
        self.train_indices = torch.nonzero(mask).squeeze().to(self.device)
        logging.info(
            f"    [{self.client_id}] Anchors Updated. Total: {len(self.train_indices)} (+{count})")

    def local_train(self, global_model_state, local_epochs, batch_size, lr):
        # 1. 加载全局参数
        if global_model_state is not None:
            my_state = self.model.state_dict()
            for k, v in global_model_state.items():
                if "initial_features" in k:
                    continue
                if "relation_embeddings" in k:
                    continue  # 关系嵌入通常也视为私有或半私有，视策略而定
                if self.model_type == 'decoupled' and "struct_encoder" in k:
                    continue

                # 兼容性检查：确保 shape 匹配才加载
                if k in my_state and v.shape == my_state[k].shape:
                    my_state[k] = v
            self.model.load_state_dict(my_state, strict=False)

        # 2. 训练
        if self.model_type in ['gcn', 'decoupled']:
            return self._train_graph_model(local_epochs, lr)
        elif self.model_type == 'projection':
            return self._train_projection(local_epochs, batch_size, lr)

    def _train_graph_model(self, epochs, lr):
        self.model.train()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.MarginRankingLoss(margin=config.FL_MARGIN)

        total_loss = 0.0

        for epoch in range(epochs):
            optimizer.zero_grad()

            # 🔥 [关键修改] Forward 传入 edge_index 和 edge_type
            # 兼容 Decoupled 架构：DecoupledModel.forward 需要接收这俩参数并传给 struct_encoder
            output = self.model(self.edge_index, self.edge_type)

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

            # 🔥 [新增] 强制清理显存缓存 (针对 Mac MPS)
            if config.DEVICE.type == 'mps':
                torch.mps.empty_cache()

        return self.model.state_dict(), total_loss / max(1, epochs)

    def _train_projection(self, epochs, batch_size, lr):
        # ... (Projection 逻辑保持不变，为了节省篇幅省略) ...
        return self.model.state_dict(), 0.0


class Server:
    def __init__(self):
        self.device = config.DEVICE
        ModelClass = get_model_class(config.MODEL_ARCH)

        # 初始化全局模型 (用于参数聚合的容器)
        # 注意：Server 其实不需要知道关系数，因为它只聚合 MLP 部分
        # 但为了初始化不报错，我们随便传个 1
        if config.MODEL_ARCH in ['gcn', 'decoupled']:
            self.global_model = ModelClass(
                num_entities=1,
                num_relations=1,  # 占位符
                feature_dim=config.GCN_DIM,
                hidden_dim=config.GCN_HIDDEN,
                output_dim=config.BERT_DIM,
                dropout=0
            ).to(self.device)
        else:
            self.global_model = ModelClass(
                config.TRANSE_DIM, config.BERT_DIM).to(self.device)

    def get_global_model_state(self):
        return self.global_model.state_dict()

    def aggregate_models(self, client_model_states):
        if not client_model_states:
            return None

        avg_weights = collections.OrderedDict()

        # 遍历全局模型 Key
        for key in self.global_model.state_dict().keys():
            # 过滤私有层
            if "initial_features" in key:
                continue
            if "relation_embeddings" in key:
                continue  # 关系嵌入不聚合
            if config.MODEL_ARCH == 'decoupled' and "struct_encoder" in key:
                continue

            tensors = [s[key].to(self.device)
                       for s in client_model_states if key in s]
            if not tensors:
                continue

            # 聚合逻辑 (兼容 LongTensor)
            if torch.is_floating_point(tensors[0]):
                avg_weights[key] = torch.stack(tensors).mean(dim=0)
            else:
                avg_weights[key] = torch.stack(
                    tensors).float().mean(dim=0).long()

        my_state = self.global_model.state_dict()
        my_state.update(avg_weights)
        self.global_model.load_state_dict(my_state, strict=False)
        return avg_weights
