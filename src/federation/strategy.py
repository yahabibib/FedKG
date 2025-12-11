# src/core/strategy.py
import torch
import torch.nn.functional as F
import logging

log = logging.getLogger(__name__)


class PseudoLabelGenerator:
    """
    负责生成伪标签的策略类。
    """
    @staticmethod
    def generate(emb1, emb2, threshold=0.85, device='cpu'):
        """
        基于余弦相似度生成互为最近邻 (Mutual Nearest Neighbors)。

        :param emb1: Client 1 的 Embeddings [N1, Dim]
        :param emb2: Client 2 的 Embeddings [N2, Dim]
        :param threshold: 相似度阈值
        :param device: 计算设备 (建议 CPU 以防 OOM)
        :return: List[Tuple(idx1, idx2)] - 索引对列表
        """
        # 显式转到指定设备 (CPU) 计算，避免占用宝贵的 GPU 显存
        e1 = emb1.to(device)
        e2 = emb2.to(device)

        # 1. 计算归一化后的相似度矩阵
        # Sim = (A / |A|) * (B / |B|)^T
        sim = torch.mm(F.normalize(e1, p=2, dim=1),
                       F.normalize(e2, p=2, dim=1).T)

        # 2. 找双向最大值
        # vals1: 每一行的最大值 (C1 -> C2 的最佳匹配)
        vals1, idx1 = sim.max(dim=1)
        # idx2: 每一列的最大值对应的索引 (C2 -> C1 的最佳匹配)
        _, idx2 = sim.max(dim=0)

        pairs = []
        for i in range(len(idx1)):
            j = idx1[i].item()
            # 互为最近邻校验 (Bi-directional check)
            if idx2[j].item() == i:
                if vals1[i] > threshold:
                    pairs.append((i, j))

        return pairs
