# src/models/projectors/mlp.py
import torch
import torch.nn as nn


class MLPProjector(nn.Module):
    """
    [Shared Component]
    负责将结构特征映射到 SBERT 的语义空间。
    这部分参数在联邦学习中会被上传并聚合。
    """

    def __init__(self, input_dim, output_dim, dropout=0.3):
        """
        :param input_dim: GCN 输出的维度 (hidden_dim)
        :param output_dim: SBERT 的维度 (通常是 768)
        """
        super(MLPProjector, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, input_dim),  # 增加一层非线性变换
            nn.BatchNorm1d(input_dim),        # BN 有助于收敛
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim, output_dim)  # 映射到目标空间
        )

    def forward(self, x):
        return self.net(x)
