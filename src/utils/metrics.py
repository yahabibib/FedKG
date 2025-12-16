# src/utils/metrics.py
import torch
import logging
import torch.nn.functional as F

log = logging.getLogger(__name__)


def eval_alignment(emb1_dict, emb2_dict, pairs, k_values=[1, 5, 10],
                   sbert1_dict=None, sbert2_dict=None, alpha=1.0, device='cpu'):
    """
    对齐评估函数 (支持 Score Fusion)

    :param emb1_dict: Structure Embeddings (Source)
    :param sbert1_dict: Semantic Embeddings (Source) [Optional]
    :param alpha: 融合权重 (1.0 = 纯结构, 0.0 = 纯语义)
    """
    if not emb1_dict or not emb2_dict:
        return {k: 0.0 for k in k_values}, 0.0

    # 1. 筛选有效对
    valid_pairs = []
    src_ids = set()
    trg_ids = set()
    for i1, i2 in pairs:
        if i1 in emb1_dict and i2 in emb2_dict:
            valid_pairs.append((i1, i2))
            src_ids.add(i1)
            trg_ids.add(i2)

    if not valid_pairs:
        return {k: 0.0 for k in k_values}, 0.0

    src_ids_list = sorted(list(src_ids))
    trg_ids_list = sorted(list(trg_ids))
    id2idx_src = {id: i for i, id in enumerate(src_ids_list)}
    id2idx_trg = {id: i for i, id in enumerate(trg_ids_list)}

    # 2. 计算 Structure 相似度矩阵
    t1 = torch.stack([emb1_dict[i].to(device) for i in src_ids_list])
    t2 = torch.stack([emb2_dict[i].to(device) for i in trg_ids_list])
    t1 = F.normalize(t1, p=2, dim=1)
    t2 = F.normalize(t2, p=2, dim=1)
    sim_struct = torch.mm(t1, t2.T)

    # 3. 计算 Fusion 相似度 (Score Fusion)
    if sbert1_dict is not None and sbert2_dict is not None and alpha < 1.0:
        # 确保 SBERT 存在对应的 ID
        # 注意: 这里的 ID 必须和上面 structure 的顺序一致
        sb1 = torch.stack([sbert1_dict[i].to(device) for i in src_ids_list])
        sb2 = torch.stack([sbert2_dict[i].to(device) for i in trg_ids_list])
        sb1 = F.normalize(sb1, p=2, dim=1)
        sb2 = F.normalize(sb2, p=2, dim=1)

        sim_sem = torch.mm(sb1, sb2.T)

        # [核心差异] Score Fusion: 加权相似度分数
        final_sim = (alpha * sim_struct) + ((1.0 - alpha) * sim_sem)
    else:
        final_sim = sim_struct

    # 4. 排序与指标计算
    final_sim = final_sim.cpu()
    hits = {k: 0 for k in k_values}
    mrr = 0.0

    for i1, i2 in valid_pairs:
        row = id2idx_src[i1]
        col = id2idx_trg[i2]

        target_score = final_sim[row, col].item()
        # 简单的 rank 计算: 有多少个分数比目标大
        rank = (final_sim[row] > target_score).sum().item() + 1

        mrr += 1.0 / rank
        for k in k_values:
            if rank <= k:
                hits[k] += 1

    count = len(valid_pairs)
    return {k: (v/count)*100 for k, v in hits.items()}, mrr/count
