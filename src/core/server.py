# src/core/server.py
import torch
import logging
from collections import OrderedDict
from typing import List, Dict

from src.utils.config import Config
from src.models.decoupled import DecoupledModel
from src.models.gcn import GCN


class FederatedServer:
    """
    联邦学习服务端
    负责参数聚合 (Aggregation) 和全局模型分发。
    """

    def __init__(self, config: Config):
        self.cfg = config
        self.device = config.device
        self.logger = logging.getLogger("FedServer")

        # 初始化全局模型 (用于持有共享参数结构)
        # 注意：num_entities 设为 1 即可，因为 Server 不存实体特征，只存 MLP 权重
        if self.cfg.model_arch == 'decoupled':
            self.global_model = DecoupledModel(
                num_entities=1,
                feature_dim=self.cfg.gcn_dim,
                hidden_dim=self.cfg.gcn_hidden,
                output_dim=self.cfg.bert_dim
            ).to(self.device)
        else:
            self.global_model = GCN(
                num_entities=1,
                feature_dim=self.cfg.gcn_dim,
                hidden_dim=self.cfg.gcn_hidden,
                output_dim=self.cfg.bert_dim
            ).to(self.device)

    def get_global_weights(self):
        return self.global_model.state_dict()

    def aggregate(self, client_weights_list: List[Dict[str, torch.Tensor]]):
        """
        FedAvg 聚合策略
        """
        if not client_weights_list:
            return None

        self.logger.info(
            f"Aggregating updates from {len(client_weights_list)} clients...")

        # 容器用于累加参数
        avg_weights = OrderedDict()

        # 获取模型所有键
        global_keys = self.global_model.state_dict().keys()

        for key in global_keys:
            # 1. 强制过滤私有参数 (双重保险)
            # 即使客户端传了，Server 也不应该聚合
            if "initial_features" in key:
                continue
            if self.cfg.model_arch == 'decoupled' and "struct_encoder" in key:
                continue

            # 2. 收集所有客户端该层的参数
            tensors = []
            for w in client_weights_list:
                if key in w:
                    tensors.append(w[key].to(self.device))

            if not tensors:
                continue

            # 3. 平均
            # 检查是否为浮点数 (int 类型如 step 计数器不能简单平均)
            if torch.is_floating_point(tensors[0]):
                avg_weights[key] = torch.stack(tensors).mean(dim=0)
            else:
                # 对非浮点参数（较少见）取第一个
                avg_weights[key] = tensors[0]

        # 4. 更新全局模型状态
        current_state = self.global_model.state_dict()
        current_state.update(avg_weights)
        self.global_model.load_state_dict(current_state)

        return avg_weights
