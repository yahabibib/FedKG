# 📄 evaluate.py
import torch
import torch.nn.functional as F
from tqdm import tqdm
import config


@torch.no_grad()
def evaluate_alignment(test_pairs, emb_dict_1, emb_dict_2, model_1, model_2, k_values,
                       sbert_1=None, sbert_2=None, alpha=None):
    """
    【升级版】支持双模融合评估。
    参数:
    - emb_dict_1/2: 模型输出的结构/映射 Embedding (字典格式 {id: tensor})
    - sbert_1/2:    SBERT 原始语义 Embedding (字典格式 {id: tensor})
    - alpha:        融合权重。None 则使用 config 中的默认值。
    """

    # 如果没有传 alpha，尝试从 config 读取，如果 config 也没有，默认为 1.0 (纯结构)
    if alpha is None:
        alpha = getattr(config, 'EVAL_FUSION_ALPHA', 1.0)

    print(f"\n--- 阶段四：开始评估 (Alpha={alpha}) ---")
    if sbert_1 is not None and sbert_2 is not None and alpha < 1.0:
        print(
            f"   [Mode] Dual-Encoder Fusion ({int(alpha*100)}% GCN + {int((1-alpha)*100)}% SBERT)")
    else:
        print("   [Mode] Single Structure Model (GCN Only)")

    model_1.eval()
    model_2.eval()
    model_1.to(config.DEVICE)
    model_2.to(config.DEVICE)

    # 1. 准备有效测试对
    valid_test_pairs = []
    kg1_ids = set()
    kg2_ids = set()

    for id1, id2 in test_pairs:
        # 确保 ID 在两个模型的 Embedding 中都存在
        if id1 in emb_dict_1 and id2 in emb_dict_2:
            valid_test_pairs.append((id1, id2))
            kg1_ids.add(id1)
            kg2_ids.add(id2)

    kg1_ids = sorted(list(kg1_ids))
    kg2_ids = sorted(list(kg2_ids))

    # 建立 ID 到 矩阵索引 的映射
    id_to_idx_1 = {id: i for i, id in enumerate(kg1_ids)}
    id_to_idx_2 = {id: i for i, id in enumerate(kg2_ids)}

    # 2. 准备 GCN/TransE 结构向量
    # 注意：传入的 emb_dict_1 已经是 tensor 了，这里堆叠起来
    emb_1_struct = torch.stack([emb_dict_1[i]
                               for i in kg1_ids]).to(config.DEVICE)
    emb_2_struct = torch.stack([emb_dict_2[i]
                               for i in kg2_ids]).to(config.DEVICE)

    # 如果还有模型层没跑（针对 ProjectionModel），这里跑一下
    # 对于 GCN 来说，通常在外部已经 inference 好了，这里 model_1 可能是 Identity
    emb_1_struct = model_1(emb_1_struct)
    emb_2_struct = model_2(emb_2_struct)

    # 归一化
    emb_1_struct = F.normalize(emb_1_struct, p=2, dim=1)
    emb_2_struct = F.normalize(emb_2_struct, p=2, dim=1)

    # 计算结构相似度
    sim_struct = torch.mm(emb_1_struct, emb_2_struct.T)

    # 3. 融合 SBERT (如果提供)
    final_sim_matrix = sim_struct  # 默认

    if sbert_1 is not None and sbert_2 is not None and alpha < 1.0:
        # 堆叠 SBERT 向量
        emb_1_sem = torch.stack([sbert_1[i]
                                for i in kg1_ids]).to(config.DEVICE)
        emb_2_sem = torch.stack([sbert_2[i]
                                for i in kg2_ids]).to(config.DEVICE)

        # 归一化
        emb_1_sem = F.normalize(emb_1_sem, p=2, dim=1)
        emb_2_sem = F.normalize(emb_2_sem, p=2, dim=1)

        # 计算语义相似度
        sim_sem = torch.mm(emb_1_sem, emb_2_sem.T)

        # 【核心融合公式】
        final_sim_matrix = (alpha * sim_struct) + ((1.0 - alpha) * sim_sem)

    # 移回 CPU 方便后续排序
    final_sim_matrix = final_sim_matrix.cpu()

    # 4. 计算指标
    hits_at = {k: 0 for k in k_values}
    mrr = 0.0

    for id1, id2 in tqdm(valid_test_pairs, desc="Evaluating"):
        idx1 = id_to_idx_1[id1]
        target_idx2 = id_to_idx_2[id2]

        scores = final_sim_matrix[idx1]
        rank = (torch.argsort(scores, descending=True)
                == target_idx2).nonzero().item() + 1

        mrr += 1.0 / rank
        for k in k_values:
            if rank <= k:
                hits_at[k] += 1

    count = len(valid_test_pairs)
    mrr /= count
    hits_at = {k: (v/count)*100 for k, v in hits_at.items()}

    print(f"Hits@k: {hits_at}")
    print(f"MRR: {mrr:.4f}")
    return hits_at, mrr
