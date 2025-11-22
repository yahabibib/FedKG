# 📄 evaluate.py
import torch
import torch.nn.functional as F
from tqdm import tqdm
import config


@torch.no_grad()
def evaluate_alignment(test_pairs, transe_embs_1, transe_embs_2, model_1, model_2, k_values):
    """
    【升级版】支持双模型评估。
    - model_1: 用于投影 KG1
    - model_2: 用于投影 KG2
    (如果是联邦模式，传入同一个 global_model 即可)
    """
    print("\n--- 阶段四：开始评估 ---")

    model_1.eval()
    model_2.eval()
    model_1.to(config.DEVICE)
    model_2.to(config.DEVICE)

    # 1. 准备有效测试对
    valid_test_pairs = []
    kg1_ids = set()
    kg2_ids = set()

    for id1, id2 in test_pairs:
        if id1 in transe_embs_1 and id2 in transe_embs_2:
            valid_test_pairs.append((id1, id2))
            kg1_ids.add(id1)
            kg2_ids.add(id2)

    kg1_ids = sorted(list(kg1_ids))
    kg2_ids = sorted(list(kg2_ids))

    # 2. 投影
    emb_1_T = torch.stack([transe_embs_1[i]
                          for i in kg1_ids]).to(config.DEVICE)
    emb_2_T = torch.stack([transe_embs_2[i]
                          for i in kg2_ids]).to(config.DEVICE)

    # 【关键】: 分别使用对应的模型进行投影
    emb_1_proj = model_1(emb_1_T)
    emb_2_proj = model_2(emb_2_T)

    # 3. 计算相似度和排名
    emb_1_norm = F.normalize(emb_1_proj, p=2, dim=1)
    emb_2_norm = F.normalize(emb_2_proj, p=2, dim=1)

    sim_matrix = torch.mm(emb_1_norm, emb_2_norm.T).cpu()

    # 映射索引
    id_to_idx_1 = {id: i for i, id in enumerate(kg1_ids)}
    id_to_idx_2 = {id: i for i, id in enumerate(kg2_ids)}

    hits_at = {k: 0 for k in k_values}
    mrr = 0.0

    for id1, id2 in tqdm(valid_test_pairs, desc="Evaluating"):
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

    print(f"Hits@k: {hits_at}")
    print(f"MRR: {mrr:.4f}")
    return hits_at, mrr


def diagnose_sbert_errors(test_pairs, emb1, emb2, id2uri1, id2uri2):
    """
    【新功能】SBERT 错误案例深度分析
    打印出 Top 10 个误判案例，对比“真目标”和“误判目标”的相似度差异。
    """
    print("\n🕵️‍♂️ SBERT 错题本 (Top 10 误判案例分析):")
    print("=" * 60)

    device = config.DEVICE

    # 1. 准备数据 (确保 ID 顺序一致)
    kg1_ids = sorted(list(emb1.keys()))
    kg2_ids = sorted(list(emb2.keys()))

    # 堆叠为 Tensor
    e1_tensor = torch.stack([emb1[i] for i in kg1_ids]).to(device)
    e2_tensor = torch.stack([emb2[i] for i in kg2_ids]).to(device)

    # 2. 归一化并计算相似度矩阵
    e1_norm = F.normalize(e1_tensor, dim=1)
    e2_norm = F.normalize(e2_tensor, dim=1)
    sim_matrix = torch.mm(e1_norm, e2_norm.T)  # [N1, N2]

    # 3. 建立索引映射 (ID -> Matrix Index)
    id2idx_1 = {eid: i for i, eid in enumerate(kg1_ids)}
    id2idx_2 = {eid: i for i, eid in enumerate(kg2_ids)}

    count = 0

    for src_id, tgt_id in test_pairs:
        if src_id not in id2idx_1 or tgt_id not in id2idx_2:
            continue

        idx1 = id2idx_1[src_id]
        target_idx2 = id2idx_2[tgt_id]

        # 获取预测结果 (Top 1)
        scores = sim_matrix[idx1]
        best_score, best_idx2 = torch.max(scores, dim=0)
        best_idx2 = best_idx2.item()

        # 如果预测错误 (Top 1 不是正确答案)
        if best_idx2 != target_idx2:
            count += 1
            if count > 10:
                break  # 只看前 10 个

            # 获取名称 (从 URI 中提取最后一部分)
            def get_name(uri_map, eid):
                uri = uri_map.get(eid, "Unknown")
                return uri.split('/')[-1].replace('_', ' ')

            src_name = get_name(id2uri1, src_id)
            tgt_name = get_name(id2uri2, tgt_id)

            wrong_id2 = kg2_ids[best_idx2]
            wrong_name = get_name(id2uri2, wrong_id2)

            # 获取正确答案的相似度
            correct_score = scores[target_idx2].item()
            best_score = best_score.item()

            print(f"❌ Case {count}:")
            print(f"   源实体 (KG1): {src_name}")
            print(f"   真目标 (KG2): {tgt_name:<30} (Sim: {correct_score:.4f})")
            print(f"   误判为 (KG2): {wrong_name:<30} (Sim: {best_score:.4f})")
            print(f"   > 差距: {best_score - correct_score:.4f}")
            print("-" * 60)

    print(f"🔍 诊断结束。如果'真目标'和'误判为'语义非常接近，说明 SBERT 遇到了'语义陷阱'。")
