# 📄 fl_core.py
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

        if self.model_type in ['gcn', 'decoupled']:
            self.edge_index = kwargs['edge_index'].to(self.device)
            self.edge_type = kwargs['edge_type'].to(self.device)
            num_relations = kwargs['num_rel']
            local_bert_embs = kwargs['bert']
            rel_sbert = kwargs.get('rel_sbert', None)

            self.model = ModelClass(
                num_entities=kwargs['num_ent'],
                num_relations=num_relations,
                feature_dim=config.GCN_DIM,
                hidden_dim=config.GCN_HIDDEN,
                output_dim=config.BERT_DIM,
                dropout=config.GCN_DROPOUT
            ).to(self.device)

            # 只有当传入了 rel_sbert 时才初始化 (本次我们不传)
            if hasattr(self.model, 'init_relation_embeddings') and rel_sbert is not None:
                self.model.init_relation_embeddings(rel_sbert)
            elif hasattr(self.model, 'struct_encoder') and rel_sbert is not None:
                # 兼容 Decoupled
                self.model.init_relation_embeddings(rel_sbert)

            sbert_tensor = torch.zeros(kwargs['num_ent'], config.BERT_DIM)
            self.train_indices = []
            for ent_id, emb in local_bert_embs.items():
                if ent_id < kwargs['num_ent']:
                    sbert_tensor[ent_id] = emb
                    self.train_indices.append(ent_id)
            self.sbert_target = sbert_tensor.to(self.device)
            self.train_indices = torch.tensor(
                self.train_indices).to(self.device)
            logging.info(
                f"Client {self.client_id}: {len(self.train_indices)} anchors ready.")

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
        if global_model_state is not None:
            my_state = self.model.state_dict()
            for k, v in global_model_state.items():
                if "initial_features" in k:
                    continue
                if "relation_embeddings" in k:
                    continue
                if self.model_type == 'decoupled' and "struct_encoder" in k:
                    continue
                if k in my_state and v.shape == my_state[k].shape:
                    my_state[k] = v
            self.model.load_state_dict(my_state, strict=False)

        return self._train_graph_model(local_epochs, lr)

    def _train_graph_model(self, epochs, lr):
        self.model.train()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.MarginRankingLoss(margin=config.FL_MARGIN)
        total_loss = 0.0

        for epoch in range(epochs):
            optimizer.zero_grad()

            # Forward
            output = self.model(self.edge_index, self.edge_type)

            out_batch = output[self.train_indices]
            target_batch = self.sbert_target[self.train_indices]
            pos_sim = F.cosine_similarity(out_batch, target_batch)

            with torch.no_grad():
                sim_mat = torch.mm(F.normalize(out_batch, dim=1),
                                   F.normalize(target_batch, dim=1).T)
                sim_mat.fill_diagonal_(-2.0)
                hard_neg_indices = sim_mat.argmax(dim=1)

            neg_target = target_batch[hard_neg_indices]
            neg_sim = F.cosine_similarity(out_batch, neg_target)
            loss = criterion(pos_sim, neg_sim, torch.ones_like(pos_sim))

            loss.backward()

            # 🔥 [关键] 梯度裁剪
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=1.0)

            optimizer.step()

            if config.DEVICE.type == 'mps':
                torch.mps.empty_cache()

            total_loss += loss.item()

        return self.model.state_dict(), total_loss / max(1, epochs)


class Server:
    def __init__(self):
        self.device = config.DEVICE
        ModelClass = get_model_class(config.MODEL_ARCH)
        if config.MODEL_ARCH in ['gcn', 'decoupled']:
            self.global_model = ModelClass(
                1, 1, config.GCN_DIM, config.GCN_HIDDEN, config.BERT_DIM, 0).to(self.device)

    def get_global_model_state(self):
        return self.global_model.state_dict()

    def aggregate_models(self, client_model_states):
        if not client_model_states:
            return None
        avg_weights = collections.OrderedDict()
        for key in self.global_model.state_dict().keys():
            if "initial_features" in key:
                continue
            if "relation_embeddings" in key:
                continue
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
        self.global_model.load_state_dict(my_state, strict=False)
        return avg_weights
