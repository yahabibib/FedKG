# 📄 evaluate_fusion.py
# 【最终适配版】支持 Relation-Aware GCN 的融合评估脚本
# 它可以加载训练好的模型 (Checkpoint)，并测试不同 Alpha 下的融合效果

import torch
import torch.nn.functional as F
import os
import config
import data_loader
import precompute
import fl_core
from tqdm import tqdm

# ==========================================
# 🔧 融合评估的核心函数
# ==========================================


@torch.no_grad()
def run_fusion_eval(test_pairs, gcn_emb_1, gcn_emb_2, sbert_emb_1, sbert_emb_2, alpha=0.6, k_values=[1, 10, 50]):
    """
    alpha: 融合权重。
           alpha=1.0 -> 纯 GCN
           alpha=0.0 -> 纯 SBERT
           alpha=0.5 -> 融合
    """
    print(f"\n⚡️ 开始融合评估 (Alpha = {alpha})")
    print(f"   说明: {int(alpha*100)}% 结构(GCN) + {int((1-alpha)*100)}% 语义(SBERT)")

    device = config.DEVICE

    # 1. 准备有效对
    valid_pairs = []
    kg1_ids, kg2_ids = set(), set()

    for i1, i2 in test_pairs:
        if i1 in gcn_emb_1 and i2 in gcn_emb_2:
            valid_pairs.append((i1, i2))
            kg1_ids.add(i1)
            kg2_ids.add(i2)

    kg1_ids = sorted(list(kg1_ids))
    kg2_ids = sorted(list(kg2_ids))

    id2idx_1 = {id: i for i, id in enumerate(kg1_ids)}
    id2idx_2 = {id: i for i, id in enumerate(kg2_ids)}

    # 2. 堆叠并归一化 - GCN 部分
    t_gcn_1 = torch.stack([gcn_emb_1[i] for i in kg1_ids]).to(device)
    t_gcn_2 = torch.stack([gcn_emb_2[i] for i in kg2_ids]).to(device)
    t_gcn_1 = F.normalize(t_gcn_1, p=2, dim=1)
    t_gcn_2 = F.normalize(t_gcn_2, p=2, dim=1)

    sim_gcn = torch.mm(t_gcn_1, t_gcn_2.T)

    # 3. 堆叠并归一化 - SBERT 部分
    t_sb_1 = torch.stack([sbert_emb_1[i] for i in kg1_ids]).to(device)
    t_sb_2 = torch.stack([sbert_emb_2[i] for i in kg2_ids]).to(device)
    t_sb_1 = F.normalize(t_sb_1, p=2, dim=1)
    t_sb_2 = F.normalize(t_sb_2, p=2, dim=1)

    sim_sb = torch.mm(t_sb_1, t_sb_2.T)

    # 4. 加权融合
    sim_final = (alpha * sim_gcn) + ((1.0 - alpha) * sim_sb)
    sim_final = sim_final.cpu()

    # 5. 计算指标
    hits_at = {k: 0 for k in k_values}
    mrr = 0.0

    for i1, i2 in tqdm(valid_pairs, desc="   Ranking", leave=False):
        idx1 = id2idx_1[i1]
        target_idx2 = id2idx_2[i2]

        scores = sim_final[idx1]
        rank = (torch.argsort(scores, descending=True)
                == target_idx2).nonzero().item() + 1

        mrr += 1.0 / rank
        for k in k_values:
            if rank <= k:
                hits_at[k] += 1

    count = len(valid_pairs)
    mrr /= count
    hits_at = {k: (v/count)*100 for k, v in hits_at.items()}

    print(
        f"   🏆 结果: Hits@1={hits_at[1]:.2f} | Hits@10={hits_at[10]:.2f} | MRR={mrr:.4f}")
    return hits_at, mrr

# ==========================================
# 🚀 主流程
# ==========================================


def main():
    print(f"🔥 启动双模融合脚本 (Relation-Aware Version)")
    print(f"💻 设备: {config.DEVICE}")

    # --- 1. 加载数据 ---
    print("\n[1/4] 加载基础数据...")
    ent_1 = data_loader.load_id_map(config.BASE_PATH + "ent_ids_1")
    rel_1 = data_loader.load_id_map(config.BASE_PATH + "rel_ids_1")  # 需要关系ID
    trip_1 = data_loader.load_triples(config.BASE_PATH + "triples_1")
    num_ent_1 = max(list(ent_1[0].keys())) + 1

    ent_2 = data_loader.load_id_map(config.BASE_PATH + "ent_ids_2")
    rel_2 = data_loader.load_id_map(config.BASE_PATH + "rel_ids_2")  # 需要关系ID
    trip_2 = data_loader.load_triples(config.BASE_PATH + "triples_2")
    num_ent_2 = max(list(ent_2[0].keys())) + 1

    test_pairs = data_loader.load_alignment_pairs(
        config.BASE_PATH + "ref_pairs")

    # --- 2. 准备 SBERT (语义锚点) ---
    print("\n[2/4] 加载 SBERT 缓存...")
    cache_1 = "cache/sbert_KG1.pt"
    cache_2 = "cache/sbert_KG2.pt"

    if os.path.exists(cache_1) and os.path.exists(cache_2):
        sb_1 = torch.load(cache_1, map_location=config.DEVICE)
        sb_2 = torch.load(cache_2, map_location=config.DEVICE)
        print("   ✅ SBERT 缓存加载成功！")
    else:
        print("   ❌ 未找到 SBERT 缓存，请先运行 main.py 生成缓存。")
        return

    # --- 3. 准备 GCN 模型 (结构特征) ---
    print("\n[3/4] 加载训练好的模型...")

    # ⚠️ 请修改这里为你想要测试的 Iteration (通常是 5)
    TARGET_ITER = 5
    ckpt_c1 = f"checkpoints/c1_iter_{TARGET_ITER}.pth"
    ckpt_c2 = f"checkpoints/c2_iter_{TARGET_ITER}.pth"

    if not (os.path.exists(ckpt_c1) and os.path.exists(ckpt_c2)):
        print(f"   ❌ 找不到 Checkpoint 文件: {ckpt_c1}")
        print("   请修改脚本中的 TARGET_ITER。")
        return

    # 🔥 [修改] 构建带关系的图结构 (Edge Index & Type)
    print("   构建带关系的图结构...")
    edge_index_1, edge_type_1 = precompute.build_graph_data(
        trip_1, num_ent_1, len(rel_1[0]))
    edge_index_2, edge_type_2 = precompute.build_graph_data(
        trip_2, num_ent_2, len(rel_2[0]))

    # 初始化空模型
    print("   初始化模型结构 (Relation-Aware)...")
    config.MODEL_ARCH = 'decoupled'

    # 初始化 Client (传入 edge_index, edge_type, num_rel)
    # 注意：不需要传 rel_sbert，因为我们会加载 checkpoint 覆盖权重
    c1 = fl_core.Client("C1_Eval", config.DEVICE,
                        bert={0: torch.zeros(768)}, num_ent=num_ent_1,
                        num_rel=len(rel_1[0]),  # 必须传
                        edge_index=edge_index_1, edge_type=edge_type_1)

    c1.model.load_state_dict(torch.load(ckpt_c1, map_location=config.DEVICE))
    c1.model.eval()

    c2 = fl_core.Client("C2_Eval", config.DEVICE,
                        bert={0: torch.zeros(768)}, num_ent=num_ent_2,
                        num_rel=len(rel_2[0]),  # 必须传
                        edge_index=edge_index_2, edge_type=edge_type_2)

    c2.model.load_state_dict(torch.load(ckpt_c2, map_location=config.DEVICE))
    c2.model.eval()

    print("   ✅ 模型加载完毕！开始推理 GCN 特征...")
    with torch.no_grad():
        # 🔥 [修改] 推理时传入 Edge Index 和 Type
        out_1 = c1.model(c1.edge_index, c1.edge_type).detach().cpu()
        out_2 = c2.model(c2.edge_index, c2.edge_type).detach().cpu()

        gcn_emb_1 = {i: out_1[i] for i in range(len(out_1))}
        gcn_emb_2 = {i: out_2[i] for i in range(len(out_2))}

    # --- 4. 执行融合评估 ---
    print("\n[4/4] 最终对决：不同 Alpha 的效果对比")
    print("=" * 60)

    # 建议扫描范围更广一点，因为模型变强了
    alphas_to_test = [0.0, 1.0, 0.4, 0.5, 0.6, 0.7, 0.8]

    best_h1 = 0
    best_alpha = 0

    for a in alphas_to_test:
        h1, _ = run_fusion_eval(test_pairs, gcn_emb_1,
                                gcn_emb_2, sb_1, sb_2, alpha=a)
        if h1[1] > best_h1:
            best_h1 = h1[1]
            best_alpha = a

    print("\n" + "="*60)
    print(f"🎉 最佳配置: Alpha = {best_alpha}")
    print(f"📈 最佳 Hits@1: {best_h1:.2f}%")
    print("="*60)


if __name__ == "__main__":
    main()
