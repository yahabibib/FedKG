from src.core.client import FederatedClient
from src.data.preprocessor import DataPreprocessor
from src.data.loader import DataLoader
from src.utils.logger import setup_logger
from src.utils.config import Config
import sys
import os
import torch
import torch.nn.functional as F

# 路径修复
current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)


def analyze_mining(name, emb1, emb2, id2uri1, id2uri2, threshold=0.85):
    """通用挖掘分析函数"""
    # 移到 CPU 计算，方便演示
    emb1 = F.normalize(emb1.cpu(), dim=1)
    emb2 = F.normalize(emb2.cpu(), dim=1)
    sim_mat = torch.mm(emb1, emb2.T)  # [N1, N2]

    # 找最大值
    val, idx = torch.max(sim_mat, dim=1)

    found_pairs = []
    for i in range(len(emb1)):
        j = idx[i].item()
        score = val[i].item()

        if score > threshold:
            # 安全获取名字
            uri1 = id2uri1.get(i, f"ID_{i}")
            uri2 = id2uri2.get(j, f"ID_{j}")
            found_pairs.append((uri1, uri2, score))

    print(f"\n[{name}] Threshold={threshold} | Found: {len(found_pairs)}")
    return set([(p[0], p[1]) for p in found_pairs]), found_pairs


def main():
    cfg = Config()
    # 强制使用 CPU 方便演示，或者使用 auto
    device = cfg.device
    print(f"🧪 启动挖掘策略对比实验 (Device: {device})...")

    # 1. 加载数据
    loader = DataLoader(cfg)
    prep = DataPreprocessor(cfg)

    id2uri_1, uri2id_1 = loader.load_id_map("ent_ids_1")
    id2uri_2, uri2id_2 = loader.load_id_map("ent_ids_2")

    # 【核心修复】必须与训练时的 ID 空间一致
    num_ent_1 = max(id2uri_1.keys()) + 1 if id2uri_1 else 0
    num_ent_2 = max(id2uri_2.keys()) + 1 if id2uri_2 else 0

    # 2. 加载 SBERT (语义基准)
    print("Loading SBERT...")
    # 注意：这里需要传入 id2uri 以便兜底逻辑工作
    sbert_1 = prep.compute_sbert_embeddings(
        {}, list(id2uri_1.keys()), id2uri_1, "sbert_kg1_finetuned")
    sbert_2 = prep.compute_sbert_embeddings(
        {}, list(id2uri_2.keys()), id2uri_2, "sbert_kg2_finetuned")

    # 3. 加载训练好的 GCN (结构特征)
    print("Loading Trained GCN Models...")
    # 模拟 Client 初始化以加载模型结构
    dummy_adj = torch.sparse_coo_tensor(
        torch.zeros(2, 1), torch.zeros(1), (1, 1))

    # 必须使用 device 初始化，否则 load_state_dict 会报错设备不一致
    c1 = FederatedClient("C1", cfg, {'adj': dummy_adj, 'num_ent': num_ent_1})
    c2 = FederatedClient("C2", cfg, {'adj': dummy_adj, 'num_ent': num_ent_2})

    # 加载权重
    print("Loading Checkpoints...")
    ckpt_path_1 = os.path.join(project_root, "output/checkpoints/c1_best.pth")
    ckpt_path_2 = os.path.join(project_root, "output/checkpoints/c2_best.pth")

    if not os.path.exists(ckpt_path_1) or not os.path.exists(ckpt_path_2):
        print(
            "❌ Checkpoint not found! Please run 'scripts/run_stage2.py --mode full' first.")
        return

    c1.model.load_state_dict(torch.load(ckpt_path_1, map_location=device))
    c2.model.load_state_dict(torch.load(ckpt_path_2, map_location=device))
    c1.model.eval()
    c2.model.eval()

    # 推理 (需要加载真实的 Adj 才能算出 GCN 特征)
    print("Computing GCN Embeddings...")
    trip1 = loader.load_triples("triples_1")
    trip2 = loader.load_triples("triples_2")

    # 这里的 num_ent 也必须正确
    adj1 = prep.build_adjacency_matrix(trip1, num_ent_1)
    adj2 = prep.build_adjacency_matrix(trip2, num_ent_2)

    # 处理 MPS 兼容性 (Client 内部虽然处理了，但我们这里直接调 model，需手动处理)
    # 如果是 MPS，adj 留 CPU；如果是 CUDA，转 GPU
    if device.type != 'mps':
        adj1 = adj1.to(device)
        adj2 = adj2.to(device)

    with torch.no_grad():
        # model 在 device 上，adj 可能在 CPU (MPS) 或 GPU (CUDA)
        # GCNLayer forward 会自动处理 sparse mm
        gcn_1 = c1.model(adj1).cpu()  # 结果移回 CPU
        gcn_2 = c2.model(adj2).cpu()

    # -----------------------------------------------------------
    # 🔬 对比实验：SBERT vs GCN vs Fusion
    # -----------------------------------------------------------

    # 1. 纯语义 (SBERT)
    # 注意：sbert 字典可能不连续，转为 tensor 矩阵
    def dict_to_tensor(emb_dict, num_ent):
        dim = list(emb_dict.values())[0].shape[0]
        mat = torch.zeros(num_ent, dim)
        for k, v in emb_dict.items():
            if k < num_ent:
                mat[k] = v
        return mat

    sbert_mat_1 = dict_to_tensor(sbert_1, num_ent_1)
    sbert_mat_2 = dict_to_tensor(sbert_2, num_ent_2)

    sbert_pairs, sbert_list = analyze_mining(
        "Pure SBERT", sbert_mat_1, sbert_mat_2, id2uri_1, id2uri_2)

    # 2. 纯结构 (GCN)
    gcn_pairs, gcn_list = analyze_mining(
        "Pure GCN", gcn_1, gcn_2, id2uri_1, id2uri_2)

    # 3. 融合 (Fusion)
    alpha = 0.42
    fusion_1 = alpha * F.normalize(gcn_1, p=2, dim=1) + \
        (1-alpha) * F.normalize(sbert_mat_1, p=2, dim=1)
    fusion_2 = alpha * F.normalize(gcn_2, p=2, dim=1) + \
        (1-alpha) * F.normalize(sbert_mat_2, p=2, dim=1)

    fusion_pairs, fusion_list = analyze_mining(
        "Fusion (Alpha=0.42)", fusion_1, fusion_2, id2uri_1, id2uri_2)

    # -----------------------------------------------------------
    # 🕵️‍♂️ 深度分析
    # -----------------------------------------------------------
    print("\n" + "="*60)
    print("🔍 差异分析 (Structurally Discovered Pairs)")
    print("="*60)

    # 找出：Fusion 找到了，但 SBERT 没找到的 (即由结构立大功的)
    # 注意：这里我们用 set 差集
    structure_wins = fusion_pairs - sbert_pairs

    print(
        f"Found {len(structure_wins)} pairs where Structure helped Semantic break the threshold.\n")

    count = 0
    # 展示前 10 个
    for u1, u2 in list(structure_wins)[:10]:
        print(f"🚀 [New Pair] {u1} <---> {u2}")
        count += 1

    if count == 0:
        print("⚠️ 没找到差异？可能是阈值设置太高，或 SBERT 已经太强了。")


if __name__ == "__main__":
    main()
