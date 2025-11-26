# 📄 main_centralized.py
# 实验设置 B: Collection (Centralized)
# 模拟将 KG1 和 KG2 数据集中到一台机器，构建大图进行训练
# 注意：这需要合并邻接矩阵，并处理 ID 偏移

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import config
import data_loader
import precompute
import evaluate
from models.gcn import GCN
from tqdm import tqdm
import os
import utils_logger


def run_centralized_experiment():
    print(f"{'='*60}")
    print("🧪 实验 B: Collection (Centralized Training)")
    print(f"   目标: 验证无隐私限制下，合并图结构训练的理论上限")
    print(f"{'='*60}")

    # --- 1. 数据加载与合并 ---
    print("\n[1] 数据合并...")
    ent_1 = data_loader.load_id_map(config.BASE_PATH + "ent_ids_1")
    trip_1 = data_loader.load_triples(config.BASE_PATH + "triples_1")
    num_ent_1 = max(list(ent_1[0].keys())) + 1

    ent_2 = data_loader.load_id_map(config.BASE_PATH + "ent_ids_2")
    trip_2 = data_loader.load_triples(config.BASE_PATH + "triples_2")
    num_ent_2 = max(list(ent_2[0].keys())) + 1

    # 【关键】ID 偏移：KG2 的 ID 需要加上 KG1 的总数，避免冲突
    offset = num_ent_1
    trip_2_shifted = [(h + offset, r, t + offset) for h, r, t in trip_2]

    triples_all = trip_1 + trip_2_shifted
    total_ent = num_ent_1 + num_ent_2
    print(
        f"   Merged Graph: {num_ent_1} (KG1) + {num_ent_2} (KG2) = {total_ent} Entities")
    print(f"   Merged Edges: {len(triples_all)}")

    # --- 2. 预计算 (大图邻接矩阵 & SBERT) ---
    print("\n[2] 构建大图邻接矩阵 & 加载 SBERT...")
    adj_all = precompute.build_adjacency_matrix(triples_all, total_ent)

    # 加载 SBERT (复用缓存)
    sb_1 = precompute.get_bert_embeddings(
        ent_1, {}, "KG1", cache_file="cache/sbert_KG1.pt")
    sb_2 = precompute.get_bert_embeddings(
        ent_2, {}, "KG2", cache_file="cache/sbert_KG2.pt")

    # 合并 SBERT Features (作为训练目标)
    # 构造一个大 Tensor [total_ent, 768]
    sbert_target = torch.zeros(total_ent, config.BERT_DIM)

    train_indices = []
    # 填入 KG1
    for eid, emb in sb_1.items():
        sbert_target[eid] = emb
        train_indices.append(eid)
    # 填入 KG2 (记得加 offset)
    for eid, emb in sb_2.items():
        sbert_target[eid + offset] = emb
        train_indices.append(eid + offset)

    sbert_target = sbert_target.to(config.DEVICE)
    train_indices = torch.tensor(train_indices).to(config.DEVICE)

    # 移动邻接矩阵 (如果是 CUDA)
    if config.DEVICE.type == 'cuda':
        adj_all = adj_all.to(config.DEVICE)

    # --- 3. 初始化集中式模型 ---
    print("\n[3] 初始化集中式 GCN...")
    # 这里直接用一个大 GCN，不需 Decoupled，因为数据都在本地
    model = GCN(
        num_entities=total_ent,
        feature_dim=config.GCN_DIM,
        hidden_dim=config.GCN_HIDDEN,
        output_dim=config.BERT_DIM,
        dropout=config.GCN_DROPOUT
    ).to(config.DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=config.FL_LR)
    criterion = nn.MarginRankingLoss(margin=config.FL_MARGIN)

    # --- 4. 训练循环 ---
    print("\n[4] 开始训练...")
    epochs = 200  # 集中式通常收敛较快，或者设为与联邦总轮次相当

    model.train()
    for epoch in tqdm(range(epochs), desc="Training"):
        optimizer.zero_grad()
        output = model(adj_all)

        # 锚点对齐 Loss
        out_batch = output[train_indices]
        target_batch = sbert_target[train_indices]

        pos_sim = F.cosine_similarity(out_batch, target_batch)

        # 简单负采样
        # 实际代码中可以使用 fl_core 里更复杂的 hard mining，这里为了演示保持简洁
        # 使用随机负采样模拟
        perm = torch.randperm(len(target_batch)).to(config.DEVICE)
        neg_target = target_batch[perm]
        neg_sim = F.cosine_similarity(out_batch, neg_target)

        loss = criterion(pos_sim, neg_sim, torch.ones_like(pos_sim))
        loss.backward()
        optimizer.step()

    # --- 5. 评估 ---
    print("\n[5] 最终评估...")
    model.eval()
    with torch.no_grad():
        embeddings_all = model(adj_all).detach().cpu()

    # 拆分 Embedding 回去
    # KG1: 0 ~ num_ent_1
    emb_1 = {i: embeddings_all[i] for i in range(num_ent_1)}

    # KG2: offset ~ total (注意：Key 要减去 offset 变回原始 ID，以便评估器识别)
    emb_2 = {i: embeddings_all[i + offset] for i in range(num_ent_2)}

    test_pairs = data_loader.load_alignment_pairs(
        config.BASE_PATH + "ref_pairs")

    # 复用 evaluate 模块 (此时模型已包含结构信息，设 Alpha=1.0 纯结构，或者 0.42 融合)
    print("   [Mode] Evaluation with Fusion (Alpha=0.42)")
    hits, mrr = evaluate.evaluate_alignment(
        test_pairs, emb_1, emb_2,
        nn.Identity(), nn.Identity(),  # 模型已推理完毕，传入 Identity
        config.EVAL_K_VALUES,
        sbert_1=sb_1, sbert_2=sb_2,
        alpha=config.EVAL_FUSION_ALPHA
    )

    # ---> 新增记录代码
    utils_logger.log_experiment_result(
        exp_name="Collection (Centralized)",
        dataset=config.CURRENT_DATASET_NAME,
        metrics={"hits1": hits[1], "hits10": hits[10], "mrr": mrr},
        params={"epochs": 200}
    )


if __name__ == "__main__":
    run_centralized_experiment()
