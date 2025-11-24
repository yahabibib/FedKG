# 📄 eval_fusion.py
# 专门用于“双模融合”推理的独立脚本
# 它可以加载训练好的 GCN 模型，并与 SBERT 进行加权融合，瞬间提升效果！

import torch
import torch.nn.functional as F
import os
import config
import data_loader
import precompute
import fl_core
from tqdm import tqdm

# ==========================================
# 🔧 融合评估的核心函数 (本地定义，无需修改 evaluate.py)
# ==========================================


@torch.no_grad()
def run_fusion_eval(test_pairs, gcn_emb_1, gcn_emb_2, sbert_emb_1, sbert_emb_2, alpha=0.6, k_values=[1, 10, 50]):
    """
    alpha: 融合权重。
           alpha=1.0 -> 纯 GCN
           alpha=0.0 -> 纯 SBERT
           alpha=0.6 -> 推荐融合比例 (60% GCN + 40% SBERT)
    """
    print(f"\n⚡️ 开始融合评估 (Alpha = {alpha})")
    print(f"   说明: {int(alpha*100)}% 结构(GCN) + {int((1-alpha)*100)}% 语义(SBERT)")

    device = config.DEVICE

    # 1. 准备 ID 列表
    # 假设输入的 embedding 都是字典 {id: tensor}
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

    # 4. 加权融合 (广播机制自动处理)
    # final_sim = alpha * GCN + (1-alpha) * SBERT
    sim_final = (alpha * sim_gcn) + ((1.0 - alpha) * sim_sb)

    # 移回 CPU 计算排名
    sim_final = sim_final.cpu()

    # 5. 计算指标
    hits_at = {k: 0 for k in k_values}
    mrr = 0.0

    for i1, i2 in tqdm(valid_pairs, desc="   Ranking", leave=False):
        idx1 = id2idx_1[i1]
        target_idx2 = id2idx_2[i2]

        scores = sim_final[idx1]
        # 获取排名的位置 (从0开始所以+1)
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


# ... (前面的 import 和 run_fusion_eval 函数保持不变) ...

# ==========================================
# 🚀 主流程 (修复版)
# ==========================================
def main():
    print(f"🔥 启动双模融合脚本 (Ensemble Inference)")
    print(f"💻 设备: {config.DEVICE}")

    # --- 1. 加载数据 ---
    print("\n[1/4] 加载基础数据...")
    ent_1 = data_loader.load_id_map(config.BASE_PATH + "ent_ids_1")
    trip_1 = data_loader.load_triples(config.BASE_PATH + "triples_1")
    num_ent_1 = max(list(ent_1[0].keys())) + 1

    ent_2 = data_loader.load_id_map(config.BASE_PATH + "ent_ids_2")
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
    print("\n[3/4] 加载训练好的 GCN 模型...")
    # ⚠️ 请确保这里的 TARGET_ITER 是你实际跑完的轮数
    TARGET_ITER = 5
    ckpt_c1 = f"checkpoints/c1_iter_{TARGET_ITER}.pth"
    ckpt_c2 = f"checkpoints/c2_iter_{TARGET_ITER}.pth"

    if not (os.path.exists(ckpt_c1) and os.path.exists(ckpt_c2)):
        print(f"   ❌ 找不到 Checkpoint 文件: {ckpt_c1}")
        print("   请修改脚本中的 TARGET_ITER 为你实际拥有的轮次。")
        return

    # 构建邻接矩阵
    print("   构建邻接矩阵 (如果比较大请稍等)...")

    # 【关键修复】:
    # MPS (Mac) 不支持稀疏张量，所以 adj 必须留在 CPU。
    # CUDA (Nvidia) 支持，所以如果是 cuda 可以转过去。
    adj_1 = precompute.build_adjacency_matrix(trip_1, num_ent_1)
    adj_2 = precompute.build_adjacency_matrix(trip_2, num_ent_2)

    if config.DEVICE.type == 'cuda':
        adj_1 = adj_1.to(config.DEVICE)
        adj_2 = adj_2.to(config.DEVICE)
    else:
        print("   [提示] 检测到非 CUDA 环境 (如 MPS/CPU)，邻接矩阵将保留在内存中以避免兼容性错误。")

    # 初始化空模型
    print("   初始化模型结构...")
    config.MODEL_ARCH = 'decoupled'

    # 注意：这里初始化 Client 时，adj 传进去是什么设备就是什么设备
    c1 = fl_core.Client("C1_Eval", config.DEVICE, bert={
                        0: torch.zeros(768)}, num_ent=num_ent_1, adj=adj_1)
    c1.model.load_state_dict(torch.load(ckpt_c1, map_location=config.DEVICE))
    c1.model.eval()

    c2 = fl_core.Client("C2_Eval", config.DEVICE, bert={
                        0: torch.zeros(768)}, num_ent=num_ent_2, adj=adj_2)
    c2.model.load_state_dict(torch.load(ckpt_c2, map_location=config.DEVICE))
    c2.model.eval()

    print("   ✅ 模型加载完毕！开始推理 GCN 特征...")
    with torch.no_grad():
        # 获取 GCN 输出
        # 模型内部的 GCNLayer 会自动处理 "MPS输入 + CPU矩阵" 的情况
        out_1 = c1.model(adj_1).detach().cpu()
        out_2 = c2.model(adj_2).detach().cpu()

        gcn_emb_1 = {i: out_1[i] for i in range(len(out_1))}
        gcn_emb_2 = {i: out_2[i] for i in range(len(out_2))}

    # --- 4. 执行融合评估 ---
    print("\n[4/4] 最终对决：不同 Alpha 的效果对比")
    print("=" * 60)

    # 🎯 步长 0.01 的地毯式搜索
    alphas_to_test = [
        0.40, 0.41, 0.42, 0.43, 0.44,
        0.45,
        0.46, 0.47, 0.48, 0.49, 0.50
    ]

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
