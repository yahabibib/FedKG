# src/core/server.py
import torch
from sentence_transformers import SentenceTransformer
from collections import OrderedDict
import os
import logging

log = logging.getLogger(__name__)


class Server:
    def __init__(self, cfg):
        self.cfg = cfg
        # Server 永远驻留 CPU
        log.info("[Server] Initializing on CPU...")
        self.global_model = SentenceTransformer(
            cfg.task.model.name, device='cpu')

    def aggregate(self, client_weights_list):
        """
        FedAvg 聚合策略
        :param client_weights_list: List[OrderedDict] - 客户端 state_dict 列表
        """
        if not client_weights_list:
            return None

        # log.info(f"[Server] Aggregating parameters from {len(client_weights_list)} clients...")
        avg_weights = OrderedDict()

        # 获取第一个客户端的 keys 作为基准
        keys = client_weights_list[0].keys()

        for key in keys:
            # 确保所有 tensor 都在 CPU 上进行平均
            tensors = [w[key].to('cpu') for w in client_weights_list]
            # Stack 后求平均
            avg_weights[key] = torch.stack(tensors).mean(dim=0)

        # 更新全局模型
        self.global_model.load_state_dict(avg_weights)
        return avg_weights

    def get_global_weights(self):
        return self.global_model.state_dict()

    def save_model(self, suffix="best"):
        """保存 SBERT 全局模型 (含 Config 和 Tokenizer)"""
        save_dir = os.path.join(
            self.cfg.task.checkpoint.save_dir,
            f"sbert_{self.cfg.task.strategy.text_mode}_{suffix}"
        )

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        log.info(f"💾 Saving global model to: {save_dir}")
        self.global_model.save(save_dir)
        log.info("✅ Model saved successfully!")
