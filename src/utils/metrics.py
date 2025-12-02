# src/utils/metrics.py
import torch
import torch.nn.functional as F
import logging
from typing import Dict, List, Tuple


class Evaluator:
    """
    对齐效果评估器
    支持: 纯结构(GCN), 纯语义(SBERT), 以及两者融合(Fusion)
    """

    def __init__(self, device):
        self.device = device
        self.logger = logging.getLogger("Evaluator")

    @torch.no_grad()
    def evaluate(self,
                 test_pairs: List[Tuple[int, int]],
                 emb_src: Dict[int, torch.Tensor],
                 emb_tgt: Dict[int, torch.Tensor],
                 sbert_src: Dict[int, torch.Tensor] = None,
                 sbert_tgt: Dict[int, torch.Tensor] = None,
                 alpha: float = 1.0,
                 k_values: List[int] = [1, 10]) -> Tuple[Dict[int, float], float]:
        """
        :param alpha: 1.0 = 仅使用 emb_src/tgt (结构); 0.0 = 仅使用 sbert (语义)
        """
        # 1. 筛选有效对齐对 (确保 ID 在 embedding 中存在)
        valid_pairs = []
        src_ids = []
        tgt_ids = []

        for i, j in test_pairs:
            if i in emb_src and j in emb_tgt:
                valid_pairs.append((i, j))
                src_ids.append(i)
                tgt_ids.append(j)

        if not valid_pairs:
            self.logger.warning("No valid test pairs found!")
            return {k: 0.0 for k in k_values}, 0.0

        # 去重并排序，建立矩阵索引映射
        unique_src = sorted(list(set(src_ids)))
        unique_tgt = sorted(list(set(tgt_ids)))

        src_id2idx = {eid: idx for idx, eid in enumerate(unique_src)}
        tgt_id2idx = {eid: idx for idx, eid in enumerate(unique_tgt)}

        # 2. 准备结构向量矩阵
        t_struct_src = torch.stack([emb_src[eid]
                                   for eid in unique_src]).to(self.device)
        t_struct_tgt = torch.stack([emb_tgt[eid]
                                   for eid in unique_tgt]).to(self.device)

        t_struct_src = F.normalize(t_struct_src, p=2, dim=1)
        t_struct_tgt = F.normalize(t_struct_tgt, p=2, dim=1)

        sim_struct = torch.mm(t_struct_src, t_struct_tgt.T)

        final_sim = sim_struct

        # 3. 语义融合 (如果需要)
        if alpha < 1.0 and sbert_src and sbert_tgt:
            # 确保 SBERT 向量也存在
            try:
                t_sem_src = torch.stack([sbert_src[eid]
                                        for eid in unique_src]).to(self.device)
                t_sem_tgt = torch.stack([sbert_tgt[eid]
                                        for eid in unique_tgt]).to(self.device)

                t_sem_src = F.normalize(t_sem_src, p=2, dim=1)
                t_sem_tgt = F.normalize(t_sem_tgt, p=2, dim=1)

                sim_sem = torch.mm(t_sem_src, t_sem_tgt.T)

                # 融合公式
                final_sim = alpha * sim_struct + (1.0 - alpha) * sim_sem
            except KeyError:
                self.logger.warning(
                    "Missing SBERT embeddings for fusion, falling back to structure only.")

        # 4. 计算 Hits@K 和 MRR
        # 移回 CPU 计算排名
        final_sim = final_sim.cpu()
        hits = {k: 0 for k in k_values}
        mrr = 0.0

        for src_id, tgt_id in valid_pairs:
            row_idx = src_id2idx[src_id]
            col_idx = tgt_id2idx[tgt_id]

            scores = final_sim[row_idx]
            rank = (torch.argsort(scores, descending=True)
                    == col_idx).nonzero().item() + 1

            mrr += 1.0 / rank
            for k in k_values:
                if rank <= k:
                    hits[k] += 1

        count = len(valid_pairs)
        hits_result = {k: (v / count) * 100 for k, v in hits.items()}
        mrr_result = mrr / count

        return hits_result, mrr_result
