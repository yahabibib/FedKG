# src/federation/server.py
import torch
from sentence_transformers import SentenceTransformer
# [修改] 引入 SharedArchitecture
from src.models.decoupled import SharedArchitecture
from collections import OrderedDict
import os
import logging

log = logging.getLogger(__name__)


class Server:
    def __init__(self, cfg):
        self.cfg = cfg
        self.task_type = cfg.task.type
        log.info(f"[Server] Initializing for task: {self.task_type} on CPU...")

        if self.task_type == 'sbert':
            # Phase 1: SBERT Model
            model_name = cfg.task.model.name
            self.global_model = SentenceTransformer(model_name, device='cpu')

        elif self.task_type == 'structure':
            # Phase 2: Shared Structure Model (Projector + Gate)
            input_dim = cfg.task.model.gcn_hidden
            output_dim = 768
            dropout = cfg.task.model.dropout

            # [修改] 初始化 SharedArchitecture 容器
            self.global_model = SharedArchitecture(
                input_dim, output_dim, dropout)
        else:
            raise ValueError(f"Unknown task type: {self.task_type}")

    def aggregate(self, client_weights_list):
        """
        FedAvg 聚合策略
        """
        if not client_weights_list:
            return None

        avg_weights = OrderedDict()
        # 现在 keys 是扁平的 (e.g., 'projector.net.0.weight'), 对应的值是 Tensor
        keys = client_weights_list[0].keys()

        for key in keys:
            # 确保所有 tensor 都在 CPU 上进行处理
            tensors = [w[key].to('cpu') for w in client_weights_list]

            if torch.is_floating_point(tensors[0]):
                avg_weights[key] = torch.stack(tensors).mean(dim=0)
            else:
                avg_weights[key] = torch.stack(
                    tensors).float().mean(dim=0).long()

        self.global_model.load_state_dict(avg_weights)
        return avg_weights

    def get_global_weights(self):
        return self.global_model.state_dict()

    def save_model(self, suffix="best"):
        """保存全局模型"""
        save_dir = os.path.join(
            self.cfg.task.checkpoint.save_dir,
            f"{self.task_type}_{suffix}"
        )

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        log.info(f"💾 Saving global model to: {save_dir}")

        if self.task_type == 'sbert':
            self.global_model.save(save_dir)
        else:
            torch.save(self.global_model.state_dict(),
                       os.path.join(save_dir, "model.pth"))

        log.info("✅ Model saved successfully!")
