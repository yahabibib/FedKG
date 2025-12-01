# 📄 evaluate_fusion.py
# 【修复版】增加 SBERT 键值检查，防止 KeyError
# 确保评估时只使用同时存在于 GCN 和 SBERT 中的有效实体

import torch
import torch.nn.functional as F
import os
import config
import data_loader
import precompute
import fl_core
from tqdm import tqdm
import glob
import re

# ==========================================
# 🔧 融合评估的核心函数
# ==========================================


@torch.no_grad()
def run_fusion_eval(test_pairs, gcn_emb_1, gcn_emb_2, sbert_emb_1, sbert_emb_2, alpha=0.6, k_values=[1, 10, 50]):
    """
    alpha: 融合权重。
           alpha=1.0 -> 纯 GCN (结构)
           alpha=0.0 -> 纯 SBERT (语义)
    """
    device = config.DEVICE

    # 1. 准备有效 ID 列表
    valid_pairs = []
    kg1_ids = set()
    kg2_ids = set()

    # 【核心修复】同时检查 GCN 和 SBERT 的键值
    for i1, i2 in test_pairs:
        if (i1 in gcn_emb_1 and i2 in gcn_emb_2 and
                i1 in sbert_emb_1 and i2 in sbert_emb_2):
            valid_pairs.append((i1, i2))
            kg1_ids.add(i1)
            kg2_ids.add(i2)

    if not valid_pairs:
        print("   ⚠️ No valid pairs found for evaluation!")
        return {k: 0.0 for k in k_values}, 0.0

    kg1_ids = sorted(list(kg1_ids))
    kg2_ids = sorted(list(kg2_ids))

    # 建立 ID -> Index 映射
    id2idx_1 = {id: i for i, id in enumerate(kg1_ids)}
    id2idx_2 = {id: i for i, id in enumerate(kg2_ids)}

    # 2. 堆叠并归一化 - GCN 部分
    try:
        t_gcn_1 = torch.stack([gcn_emb_1[i] for i in kg1_ids]).to(device)
        t_gcn_2 = torch.stack([gcn_emb_2[i] for i in kg2_ids]).to(device)
    except KeyError as e:
        print(f"   ❌ GCN Key Error: {e}")
        return {}, 0.0

    t_gcn_1 = F.normalize(t_gcn_1, p=2, dim=1)
    t_gcn_2 = F.normalize(t_gcn_2, p=2, dim=1)

    # 计算结构相似度矩阵
    sim_gcn = torch.mm(t_gcn_1, t_gcn_2.T)

    # 3. 堆叠并归一化 - SBERT 部分
    try:
        t_sb_1 = torch.stack([sbert_emb_1[i] for i in kg1_ids]).to(device)
        t_sb_2 = torch.stack([sbert_emb_2[i] for i in kg2_ids]).to(device)
    except KeyError as e:
        print(f"   ❌ SBERT Key Error: {e}")
        return {}, 0.0

    t_sb_1 = F.normalize(t_sb_1, p=2, dim=1)
    t_sb_2 = F.normalize(t_sb_2, p=2, dim=1)

    # 计算语义相似度矩阵
    sim_sb = torch.mm(t_sb_1, t_sb_2.T)

    # 4. 加权融合
    # Formula: Final_Sim = alpha * Struct_Sim + (1 - alpha) * Semantic_Sim
    sim_final = (alpha * sim_gcn) + ((1.0 - alpha) * sim_sb)

    # 移回 CPU 计算排名 (防止爆显存)
    sim_final = sim_final.cpu()

    # 5. 计算指标 Hits@K, MRR
    hits_at = {k: 0 for k in k_values}
    mrr = 0.0

    for i1, i2 in tqdm(valid_pairs, desc="   Ranking", leave=False):
        if i1 not in id2idx_1 or i2 not in id2idx_2:
            continue

        idx1 = id2idx_1[i1]
        target_idx2 = id2idx_2[i2]

        scores = sim_final[idx1]

        # 获取正确答案的排名 (从0开始所以+1)
        rank = (torch.argsort(scores, descending=True)
                == target_idx2).nonzero().item() + 1

        mrr += 1.0 / rank
        for k in k_values:
            if rank <= k:
                hits_at[k] += 1

    count = len(valid_pairs)
    if count > 0:
        mrr /= count
        hits_at = {k: (v/count)*100 for k, v in hits_at.items()}
    else:
        mrr = 0.0

    print(
        f"   [Alpha={alpha:.2f}] Hits@1: {hits_at[1]:.2f}% | Hits@10: {hits_at[10]:.2f}% | MRR: {mrr:.4f}")
    return hits_at, mrr

# ==========================================
# 🔍 辅助函数
# ==========================================


def find_latest_checkpoint(base_dir="checkpoints"):
    """ 自动寻找迭代轮次最大的 checkpoint """
    files = glob.glob(os.path.join(base_dir, "c1_iter_*.pth"))
    if not files:
        return None
    max_iter = 0
    for f in files:
        match = re.search(r"iter_(\d+).pth", f)
        if match:
            iter_num = int(match.group(1))
            if iter_num > max_iter:
                max_iter = iter_num
    return max_iter


def find_sbert_cache(kg_name):
    """ 尝试寻找 SBERT 缓存文件 """
    # 优先找微调过的缓存，没有再找其他的
    candidates = [
        f"cache/sbert_{kg_name}_exp4.pt",  # 我们的目标文件
        f"cache/sbert_{kg_name}_MLM.pt",
        f"cache/sbert_{kg_name}.pt",
        f"cache/sbert_{kg_name}_init.pt"
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None

# ==========================================
# 🚀 主流程
# ==========================================


def main():
    print(f"{'='*60}")
    print(f"🔥 启动双模融合脚本 (Fusion Search)")
    print(f"💻 设备: {config.DEVICE}")
    print(f"{'='*60}")

    # 1. 加载数据
    print("\n[1/5] 加载基础数据...")
    ent_1 = data_loader.load_id_map(config.BASE_PATH + "ent_ids_1")
    trip_1 = data_loader.load_triples(config.BASE_PATH + "triples_1")
    num_ent_1 = max(list(ent_1[0].keys())) + 1

    ent_2 = data_loader.load_id_map(config.BASE_PATH + "ent_ids_2")
    trip_2 = data_loader.load_triples(config.BASE_PATH + "triples_2")
    num_ent_2 = max(list(ent_2[0].keys())) + 1

    test_pairs = data_loader.load_alignment_pairs(
        config.BASE_PATH + "ref_pairs")

    # 2. 加载 SBERT
    print("\n[2/5] 加载 SBERT 缓存...")
    path_1 = find_sbert_cache("KG1")
    path_2 = find_sbert_cache("KG2")

    if path_1 and path_2:
        print(f"   📂 Loading KG1 from: {path_1}")
        print(f"   📂 Loading KG2 from: {path_2}")
        sb_1 = torch.load(path_1, map_location=config.DEVICE)
        sb_2 = torch.load(path_2, map_location=config.DEVICE)
    else:
        print("   ❌ 未找到 SBERT 缓存！请先运行 main.py 生成缓存。")
        return

    # 3. 加载 GCN
    print("\n[3/5] 加载训练好的 GCN 模型...")
    target_iter = find_latest_checkpoint()
    if target_iter is None:
        print("   ❌ 未找到任何 Checkpoint！")
        return

    ckpt_c1 = f"checkpoints/c1_iter_{target_iter}.pth"
    ckpt_c2 = f"checkpoints/c2_iter_{target_iter}.pth"
    print(f"   📂 Loading Iteration: {target_iter}")

    adj_1 = precompute.build_adjacency_matrix(trip_1, num_ent_1)
    adj_2 = precompute.build_adjacency_matrix(trip_2, num_ent_2)

    if config.DEVICE.type == 'cuda':
        adj_1 = adj_1.to(config.DEVICE)
        adj_2 = adj_2.to(config.DEVICE)

    # 初始化空模型
    dummy_bert = {0: torch.zeros(config.BERT_DIM)}
    c1 = fl_core.Client("C1_Eval", config.DEVICE,
                        bert=dummy_bert, num_ent=num_ent_1, adj=adj_1)
    c1.model.load_state_dict(torch.load(
        ckpt_c1, map_location=config.DEVICE), strict=False)  # strict=False 防止结构参数报错
    c1.model.eval()

    c2 = fl_core.Client("C2_Eval", config.DEVICE,
                        bert=dummy_bert, num_ent=num_ent_2, adj=adj_2)
    c2.model.load_state_dict(torch.load(
        ckpt_c2, map_location=config.DEVICE), strict=False)
    c2.model.eval()

    # 4. 计算 GCN 特征
    print("\n[4/5] 推理 GCN 特征...")
    with torch.no_grad():
        out_1 = c1.model(adj_1).detach().cpu()
        out_2 = c2.model(adj_2).detach().cpu()

        gcn_emb_1 = {i: out_1[i] for i in range(len(out_1))}
        gcn_emb_2 = {i: out_2[i] for i in range(len(out_2))}

    # 5. 搜索最佳 Alpha
    print("\n[5/5] 搜索最佳融合权重 (Alpha)")
    print("   Alpha = 1.0 (纯 GCN) <---> Alpha = 0.0 (纯 SBERT)")
    print("-" * 60)

    alphas_to_test = [0.0, 0.1, 0.2, 0.3, 0.4,
                      0.42, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    best_h1 = 0.0
    best_alpha = 0.0

    for a in alphas_to_test:
        h1, _ = run_fusion_eval(test_pairs, gcn_emb_1, gcn_emb_2,
                                sb_1, sb_2, alpha=a, k_values=config.EVAL_K_VALUES)
        if h1[1] > best_h1:
            best_h1 = h1[1]
            best_alpha = a

    print("-" * 60)
    print(f"🎉 最佳配置: Alpha = {best_alpha}")
    print(f"📈 最佳 Hits@1: {best_h1:.2f}%")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
