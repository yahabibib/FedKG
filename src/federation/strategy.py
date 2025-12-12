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

    @staticmethod
    def generate_fusion(emb_gcn1, emb_sem1, emb_gcn2, emb_sem2, alpha, threshold=0.85, device='cpu'):
        """
        GCN + SBERT 融合生成伪标签 (GCN Training 的关键策略)。

        :param alpha: 融合权重 (Alpha * GCN + (1-Alpha) * SBERT)
        """
        # 1. 显式转到 CPU
        e_gcn1, e_sem1 = emb_gcn1.to(device), emb_sem1.to(device)
        e_gcn2, e_sem2 = emb_gcn2.to(device), emb_sem2.to(device)

        # 2. 归一化
        e_gcn1, e_gcn2 = F.normalize(
            e_gcn1, p=2, dim=1), F.normalize(e_gcn2, p=2, dim=1)
        e_sem1, e_sem2 = F.normalize(
            e_sem1, p=2, dim=1), F.normalize(e_sem2, p=2, dim=1)

        # 3. 计算相似度矩阵 (GCN Sim + SBERT Sim)
        sim_gcn = torch.mm(e_gcn1, e_gcn2.T)
        sim_sem = torch.mm(e_sem1, e_sem2.T)

        # 4. 融合相似度
        sim_fusion = (alpha * sim_gcn) + ((1.0 - alpha) * sim_sem)

        # 5. 找互为最近邻 (MNN Check)
        vals1, idx1 = sim_fusion.max(dim=1)
        _, idx2 = sim_fusion.max(dim=0)

        pairs = []
        for i in range(len(idx1)):
            j = idx1[i].item()
            if idx2[j].item() == i:  # MNN Check
                if vals1[i] > threshold:
                    pairs.append((i, j))

        return pairs
