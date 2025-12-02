from src.core.server import FederatedServer
from src.core.client import FederatedClient
from src.data.preprocessor import DataPreprocessor
from src.data.loader import DataLoader
from src.utils.metrics import Evaluator
from src.utils.logger import setup_logger, ResultRecorder
from src.utils.config import Config
import sys
import os
import torch
import torch.nn.functional as F
import argparse
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# ==========================================
# 🚀 路径修复
# ==========================================
current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# ==========================================


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fed-LLM-SBERT Stage 2 Runner")
    parser.add_argument("--mode", type=str, default="full",
                        choices=["full", "no_llm", "no_mining"],
                        help="选择实验模式: full(完整), no_llm(无LLM增强), no_mining(无自训练)")
    parser.add_argument("--rounds", type=int, default=None,
                        help="覆盖 Config 中的联邦轮次")
    return parser.parse_args()


def generate_pseudo_pairs(emb1, emb2, valid_ids_1, valid_ids_2, threshold=0.75, device='cpu'):
    """基于双向最近邻挖掘伪对齐"""
    emb1 = F.normalize(emb1.to(device), dim=1)
    emb2 = F.normalize(emb2.to(device), dim=1)
    sim_mat = torch.mm(emb1, emb2.T)

    values_1, indices_1 = torch.max(sim_mat, dim=1)
    values_2, indices_2 = torch.max(sim_mat, dim=0)

    pseudo_pairs = []
    valid_set_1 = set(valid_ids_1)
    valid_set_2 = set(valid_ids_2)

    for i in range(len(emb1)):
        if i not in valid_set_1:
            continue
        j = indices_1[i].item()
        if j not in valid_set_2:
            continue

        if indices_2[j].item() == i:
            if values_1[i].item() > threshold:
                pseudo_pairs.append((i, j))
    return pseudo_pairs


def main():
    args = parse_args()
    cfg = Config()

    # 根据 Argument 动态调整配置
    exp_name = "FedAnchor (Full)"
    cache_suffix = "finetuned"
    total_iterations = 5

    # --- 1. 模式切换逻辑 ---
    if args.mode == "no_llm":
        exp_name = "No LLM (Raw SBERT)"
        # 强制使用原始多语言 BERT，不使用 Stage 1 的产出
        cfg.sbert_model_path = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
        cache_suffix = "raw"  # 使用不同的缓存文件名，避免冲突

    elif args.mode == "no_mining":
        exp_name = "No Mining (Iter=1)"
        total_iterations = 1  # 只跑第一轮，不进行挖掘

    if args.rounds:
        cfg.fl_rounds = args.rounds

    # 初始化 Logger
    logger = setup_logger(f"Stage2_{args.mode}")
    recorder = ResultRecorder()
    writer = SummaryWriter(log_dir=f"logs/tensorboard/{args.mode}")

    logger.info(f"🎬 Starting Experiment: [{exp_name}]")
    logger.info(f"   Mode: {args.mode}")
    logger.info(f"   SBERT: {cfg.sbert_model_path}")
    logger.info(f"   Iterations: {total_iterations}")

    # 2. 数据加载
    loader = DataLoader(cfg)
    preprocessor = DataPreprocessor(cfg)

    id2uri_1, uri2id_1 = loader.load_id_map("ent_ids_1")
    trip1 = loader.load_triples("triples_1")
    id2uri_2, uri2id_2 = loader.load_id_map("ent_ids_2")
    trip2 = loader.load_triples("triples_2")

    num_ent_1 = max(id2uri_1.keys()) + 1 if id2uri_1 else 0
    num_ent_2 = max(id2uri_2.keys()) + 1 if id2uri_2 else 0

    attr1 = loader.load_pickle_descriptions("description1.pkl", uri2id_1)
    attr2 = loader.load_pickle_descriptions("description2.pkl", uri2id_2)
    test_pairs = loader.load_alignment_pairs("ref_pairs")

    # 3. 预处理
    adj1 = preprocessor.build_adjacency_matrix(trip1, num_ent_1)
    adj2 = preprocessor.build_adjacency_matrix(trip2, num_ent_2)

    # 计算 SBERT (注意 cache_suffix 的变化)
    emb1 = preprocessor.compute_sbert_embeddings(
        attr1, list(id2uri_1.keys()), id2uri_1, f"sbert_kg1_{cache_suffix}", model_path=cfg.sbert_model_path
    )
    emb2 = preprocessor.compute_sbert_embeddings(
        attr2, list(id2uri_2.keys()), id2uri_2, f"sbert_kg2_{cache_suffix}", model_path=cfg.sbert_model_path
    )

    # 4. 初始化联邦组件
    server = FederatedServer(cfg)
    client1 = FederatedClient(
        "C1", cfg, {'adj': adj1, 'num_ent': num_ent_1, 'anchors': emb1})
    client2 = FederatedClient(
        "C2", cfg, {'adj': adj2, 'num_ent': num_ent_2, 'anchors': emb2})
    evaluator = Evaluator(cfg.device)

    global_weights = None
    best_hits1 = 0.0
    final_mrr = 0.0

    # 5. 双层循环训练
    for iteration in range(total_iterations):
        logger.info(
            f"\n🔄 Self-Training Iteration {iteration + 1}/{total_iterations}")
        current_fl_rounds = cfg.fl_rounds if iteration == 0 else 30

        # 内层循环: FL Training
        pbar = tqdm(range(current_fl_rounds),
                    desc=f"Iter {iteration+1}", dynamic_ncols=True)
        for r in pbar:
            w1, loss1 = client1.train_local(global_weights)
            w2, loss2 = client2.train_local(global_weights)

            if cfg.use_aggregation:
                global_weights = server.aggregate([w1, w2])

            if (r + 1) % 10 == 0:
                pbar.set_postfix({'loss': f"{loss1:.3f}"})
                writer.add_scalars(
                    'Loss', {'C1': loss1, 'C2': loss2}, iteration * 100 + r)

        # Iteration 结束评估
        logger.info("   Evaluating...")
        final_emb1_tensor = client1.get_embeddings()
        final_emb2_tensor = client2.get_embeddings()

        dict_emb1 = {eid: final_emb1_tensor[eid] for eid in id2uri_1.keys(
        ) if eid < len(final_emb1_tensor)}
        dict_emb2 = {eid: final_emb2_tensor[eid] for eid in id2uri_2.keys(
        ) if eid < len(final_emb2_tensor)}

        hits, mrr = evaluator.evaluate(
            test_pairs, dict_emb1, dict_emb2,
            sbert_src=emb1, sbert_tgt=emb2, alpha=cfg.eval_fusion_alpha
        )

        logger.info(
            f"   📈 Iter {iteration+1} Result: Hits@1={hits[1]:.2f}% | MRR={mrr:.4f}")

        if hits[1] > best_hits1:
            best_hits1 = hits[1]
            final_mrr = mrr
            # 只有 Full 模式才覆盖保存 best model，避免消融实验覆盖掉好模型
            if args.mode == "full":
                if not os.path.exists("output/checkpoints"):
                    os.makedirs("output/checkpoints")
                torch.save(client1.model.state_dict(),
                           "output/checkpoints/c1_best.pth")
                torch.save(client2.model.state_dict(),
                           "output/checkpoints/c2_best.pth")

        # -------------------------------------------------------
        # 核心逻辑升级：双模融合挖掘 (Fusion Mining)
        # -------------------------------------------------------
        if iteration < total_iterations - 1:
            # 阈值策略
            current_threshold = 0.75 + (iteration * 0.05)
            logger.info(
                f"   ⛏️  Mining pseudo-labels (Threshold={current_threshold:.2f})...")

            # 【升级点】不再只用 GCN 特征，而是用 (GCN + SBERT) 的融合特征来挖掘
            # 这样既能利用 GCN 的结构发现能力，又有 SBERT 的语义兜底，防止 GCN 瞎蒙

            # 1. 归一化并移动到同一设备
            gcn_emb1 = F.normalize(final_emb1_tensor.to(cfg.device))
            gcn_emb2 = F.normalize(final_emb2_tensor.to(cfg.device))

            # emb1, emb2 是最开始加载的 SBERT 初始向量 (在 main 函数开头定义的)
            # 确保它们也是 Tensor 并且在 device 上
            # 注意：emb1 是 dict，需要转成 tensor 矩阵 (按 ID 顺序)
            def dict_to_tensor(emb_dict, num_ent, dim, dev):
                mat = torch.zeros(num_ent, dim, device=dev)
                # 只填入存在的 ID，其他默认为 0
                for k, v in emb_dict.items():
                    if k < num_ent:
                        mat[k] = v.to(dev)
                return mat

            sbert_mat1 = dict_to_tensor(
                emb1, num_ent_1, cfg.bert_dim, cfg.device)
            sbert_mat2 = dict_to_tensor(
                emb2, num_ent_2, cfg.bert_dim, cfg.device)
            sbert_mat1 = F.normalize(sbert_mat1)
            sbert_mat2 = F.normalize(sbert_mat2)

            # 2. 融合 (使用配置里的 alpha)
            alpha = cfg.eval_fusion_alpha
            fused_1 = alpha * gcn_emb1 + (1 - alpha) * sbert_mat1
            fused_2 = alpha * gcn_emb2 + (1 - alpha) * sbert_mat2

            # 3. 挖掘
            new_pairs = generate_pseudo_pairs(
                fused_1, fused_2,
                list(id2uri_1.keys()), list(id2uri_2.keys()),
                threshold=current_threshold, device=cfg.device
            )

            logger.info(f"   Found {len(new_pairs)} pseudo-pairs.")

            # 4. 更新
            if len(new_pairs) > 0:
                # 注意：虽然是用融合特征挖掘的，但更新锚点时，
                # 依然是把【对方当前的融合特征】或者【对方当前的 GCN 特征】作为目标？
                # 建议：让模型去拟合对方的 GCN 特征（因为 SBERT 部分它已经学过了），或者拟合融合特征。
                # FedAnchor 原文通常指引去拟合对方的高层特征。这里我们用 GCN 特征比较纯粹。
                update_c1 = {i: final_emb2_tensor[j].to(
                    cfg.device) for i, j in new_pairs}
                update_c2 = {j: final_emb1_tensor[i].to(
                    cfg.device) for i, j in new_pairs}
                client1.update_anchors(update_c1)
                client2.update_anchors(update_c2)
            else:
                logger.warning("   No pseudo-pairs found.")

    logger.info(f"🏁 {exp_name} Finished. Best Hits@1: {best_hits1:.2f}%")
    writer.close()

    # 自动记录结果到 JSON
    recorder.add_record(exp_name, {"hits1": best_hits1, "mrr": final_mrr}, {
                        "mode": args.mode})


if __name__ == "__main__":
    main()
