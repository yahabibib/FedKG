import torch
import torch.nn.functional as F
from tqdm import tqdm
import config


@torch.no_grad()
def evaluate_alignment(test_pairs, emb_dict_1, emb_dict_2, model_1, model_2, k_values,
                       sbert_1=None, sbert_2=None, alpha=None):
    """
    【双模评估版】同时评估：
    1. Pure Structure (GCN Only): 验证动态代理是否让 GCN 学到了更好的结构特征。
    2. Fusion (GCN + SBERT): 验证最终上线效果。
    """

    # 默认 Alpha
    if alpha is None:
        alpha = getattr(config, 'EVAL_FUSION_ALPHA', 1.0)

    print(f"\n--- 阶段四：双重评估 (Dual Evaluation) ---")

    model_1.eval()
    model_2.eval()
    model_1.to(config.DEVICE)
    model_2.to(config.DEVICE)

    # 1. 准备数据
    valid_test_pairs = []
    kg1_ids = set()
    kg2_ids = set()

    for id1, id2 in test_pairs:
        if id1 in emb_dict_1 and id2 in emb_dict_2:
            valid_test_pairs.append((id1, id2))
            kg1_ids.add(id1)
            kg2_ids.add(id2)

    kg1_ids = sorted(list(kg1_ids))
    kg2_ids = sorted(list(kg2_ids))

    id_to_idx_1 = {id: i for i, id in enumerate(kg1_ids)}
    id_to_idx_2 = {id: i for i, id in enumerate(kg2_ids)}

    # 2. 计算 GCN 结构相似度 (Pure Structure)
    emb_1_struct = torch.stack([emb_dict_1[i]
                               for i in kg1_ids]).to(config.DEVICE)
    emb_2_struct = torch.stack([emb_dict_2[i]
                               for i in kg2_ids]).to(config.DEVICE)

    emb_1_struct = model_1(emb_1_struct)
    emb_2_struct = model_2(emb_2_struct)

    emb_1_struct = F.normalize(emb_1_struct, p=2, dim=1)
    emb_2_struct = F.normalize(emb_2_struct, p=2, dim=1)

    sim_struct = torch.mm(emb_1_struct, emb_2_struct.T)

    # 3. 计算 SBERT 语义相似度 (如果可用)
    sim_semantic = None
    if sbert_1 is not None and sbert_2 is not None:
        emb_1_sem = torch.stack([sbert_1[i]
                                for i in kg1_ids]).to(config.DEVICE)
        emb_2_sem = torch.stack([sbert_2[i]
                                for i in kg2_ids]).to(config.DEVICE)

        emb_1_sem = F.normalize(emb_1_sem, p=2, dim=1)
        emb_2_sem = F.normalize(emb_2_sem, p=2, dim=1)

        sim_semantic = torch.mm(emb_1_sem, emb_2_sem.T)

    # 4. 定义评估辅助函数
    def calc_metrics(sim_matrix, name):
        sim_matrix = sim_matrix.cpu()
        hits_at = {k: 0 for k in k_values}
        mrr = 0.0

        for id1, id2 in valid_test_pairs:  # 这里量大就不打印 tqdm 了，为了日志整洁
            idx1 = id_to_idx_1[id1]
            target_idx2 = id_to_idx_2[id2]

            scores = sim_matrix[idx1]
            rank = (torch.argsort(scores, descending=True)
                    == target_idx2).nonzero().item() + 1

            mrr += 1.0 / rank
            for k in k_values:
                if rank <= k:
                    hits_at[k] += 1

        count = len(valid_test_pairs)
        mrr /= count
        hits_at = {k: (v/count)*100 for k, v in hits_at.items()}

        print(
            f"   👉 [{name}] Hits@1: {hits_at[1]:.2f} | Hits@10: {hits_at[10]:.2f} | MRR: {mrr:.4f}")
        return hits_at, mrr

    # 5. 执行评估
    print(f"Evaluating {len(valid_test_pairs)} pairs...")

    # (A) 纯 GCN 评估
    h1_gcn, mrr_gcn = calc_metrics(sim_struct, "Pure GCN")

    # (B) 融合评估 (如果 SBERT 存在)
    if sim_semantic is not None:
        sim_fusion = (alpha * sim_struct) + ((1.0 - alpha) * sim_semantic)
        h1_fusion, mrr_fusion = calc_metrics(sim_fusion, f"Fusion α={alpha}")
        return h1_fusion, mrr_fusion
    else:
        return h1_gcn, mrr_gcn
