# src/utils/metrics.py
import torch
import logging
from tqdm import tqdm

log = logging.getLogger(__name__)


def eval_alignment(emb1_dict, emb2_dict, pairs, k_values=[1, 10, 50], device='cpu'):
    """
    通用的实体对齐评估函数。

    :param emb1_dict: Source KG Embeddings {id: tensor}
    :param emb2_dict: Target KG Embeddings {id: tensor}
    :param pairs: 测试对列表 [(id1, id2), ...]
    :param device: 计算设备 (建议 CPU，避免评估时 OOM)
    :return: (hits_dict, mrr)
    """
    if not emb1_dict or not emb2_dict:
        log.warning("⚠️ Empty embeddings provided for evaluation.")
        return {k: 0.0 for k in k_values}, 0.0

    # 1. 筛选有效测试对 (确保 ID 在 embedding 中存在)
    valid_pairs = []
    src_ids = set()
    trg_ids = set()

    for i1, i2 in pairs:
        if i1 in emb1_dict and i2 in emb2_dict:
            valid_pairs.append((i1, i2))
            src_ids.add(i1)
            trg_ids.add(i2)

    if not valid_pairs:
        log.warning("⚠️ No valid pairs found in embeddings.")
        return {k: 0.0 for k in k_values}, 0.0

    # 2. 构建矩阵
    src_ids_list = sorted(list(src_ids))
    trg_ids_list = sorted(list(trg_ids))

    id2idx_src = {id: i for i, id in enumerate(src_ids_list)}
    id2idx_trg = {id: i for i, id in enumerate(trg_ids_list)}

    # Stack tensors
    # 注意：这里假设输入已经是 Tensor，如果是 numpy 需要转换
    t1 = torch.stack([emb1_dict[i].to(device) for i in src_ids_list])
    t2 = torch.stack([emb2_dict[i].to(device) for i in trg_ids_list])

    # 归一化
    t1 = torch.nn.functional.normalize(t1, p=2, dim=1)
    t2 = torch.nn.functional.normalize(t2, p=2, dim=1)

    # 3. 计算相似度矩阵 (Source x Target)
    # 这一步如果矩阵很大，在 GPU 上可能会 OOM，建议外部传入 device='cpu'
    sim_mat = torch.mm(t1, t2.T)

    # 4. 计算 Rank
    hits = {k: 0 for k in k_values}
    mrr = 0.0

    # 将 sim_mat 转回 CPU 进行排序查找，防止索引操作占用显存
    sim_mat = sim_mat.cpu()

    for i1, i2 in valid_pairs:
        row_idx = id2idx_src[i1]
        col_idx = id2idx_trg[i2]

        # 取出该 Source 实体与所有 Target 实体的相似度
        scores = sim_mat[row_idx]

        # 找到目标 Target 的排名 (降序排列)
        # argsort 比较慢，直接用比较运算加速: count(score > target_score)
        target_score = scores[col_idx].item()
        rank = (scores > target_score).sum().item() + 1

        mrr += 1.0 / rank
        for k in k_values:
            if rank <= k:
                hits[k] += 1

    count = len(valid_pairs)
    hits_result = {k: (v / count) * 100 for k, v in hits.items()}
    mrr_result = mrr / count

    return hits_result, mrr_result
