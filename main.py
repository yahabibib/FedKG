import hydra
from omegaconf import DictConfig, OmegaConf
import logging
import os
import torch
import torch.nn.functional as F

# --- 导入组件 ---
from src.data.dataset import AlignmentTaskData
from src.utils.device_manager import DeviceManager
from src.utils.metrics import eval_alignment
from src.utils.logger import log_experiment_result

# --- 联邦组件 ---
from src.federation.server import Server
from src.federation.client_sbert import ClientSBERT
from src.federation.client_structure import ClientStructure
from src.federation.strategy import PseudoLabelGenerator

log = logging.getLogger(__name__)

# --- 辅助函数：阈值计算 (Curriculum Learning) ---


def get_dynamic_threshold(start_threshold, end_threshold, total_rounds, current_round):
    """
    计算当前轮次的伪标签阈值，实现课程学习策略 (逐步提高阈值)。
    """
    if current_round >= total_rounds:
        return end_threshold

    increment = (end_threshold - start_threshold) / total_rounds
    return start_threshold + current_round * increment


def _fuse_embeddings(struct_dict, sbert_dict, alpha):
    """
    辅助函数：加权融合两个 Embedding 字典
    Res = alpha * Struct + (1-alpha) * SBERT
    """
    fused = {}
    # 确保ID存在且有序
    for eid, s_emb in struct_dict.items():
        if eid in sbert_dict:
            # 归一化 (非常重要)
            v1 = F.normalize(s_emb, dim=0)
            v2 = F.normalize(sbert_dict[eid], dim=0)
            fused[eid] = alpha * v1 + (1.0 - alpha) * v2
    return fused

# --- 辅助函数：字典转换 ---


def to_dict(ids, embs):
    """将有序的ids和embeddings列表转换为字典"""
    return {id: embs[i] for i, id in enumerate(ids)}
# -----------------------------------------------


@hydra.main(config_path="configs", config_name="config", version_base="1.2")
def main(cfg: DictConfig):
    """
    FedAnchor 2.0 主入口 (支持 SBERT微调 和 结构训练 双阶段)
    """
    log.info(f"🚀 Starting Experiment: {cfg.experiment_name}")
    log.info(
        f"⚙️  Task Type: {cfg.task.type} | Mode: {cfg.task.strategy.text_mode}")

    dm = DeviceManager(cfg.system)

    log.info("📚 Loading Datasets...")
    try:
        task_data = AlignmentTaskData(cfg.data)
    except Exception as e:
        log.exception(f"❌ Data loading failed: {e}")
        return

    server = Server(cfg)

    # 4. 任务分发 (Task Dispatch)
    if cfg.task.type == 'sbert':
        log.info("🔹 Entering Phase 1: SBERT Fine-tuning")
        c1 = ClientSBERT("C1", cfg, task_data.source, dm)
        c2 = ClientSBERT("C2", cfg, task_data.target, dm)
        run_sbert_workflow(cfg, server, c1, c2, task_data.test_pairs, dm)

    elif cfg.task.type == 'structure':
        log.info("🔹 Entering Phase 2: Structure Training (GCN)")
        log.info("🏗️ Initializing Structure Clients (Loading Frozen SBERT)...")
        c1 = ClientStructure("C1", cfg, task_data.source, dm)
        c2 = ClientStructure("C2", cfg, task_data.target, dm)
        run_structure_workflow(cfg, server, c1, c2, task_data.test_pairs, dm)

    else:
        log.error(f"❌ Unknown task type: {cfg.task.type}")


def run_sbert_workflow(cfg, server, c1, c2, test_pairs, dm):
    """
    Phase 1: SBERT 混合微调工作流 (含课程学习)
    """
    results = []
    rounds = cfg.task.federated.rounds
    base_threshold = cfg.task.strategy.pseudo_threshold
    text_mode = cfg.task.strategy.text_mode

    # 阈值课程学习参数: 0.75 -> 0.85
    threshold_start = 0.75

    for r in range(rounds + 1):
        # 动态计算阈值 (课程学习)
        current_threshold = get_dynamic_threshold(
            threshold_start, base_threshold, rounds, r)
        log.info(
            f"\n{'='*40}\n🔄 [SBERT] Round {r}/{rounds} [{text_mode}] (Thresh: {current_threshold:.4f})\n{'='*40}")

        # 1. Encode
        ids1_desc, emb1_desc = c1.encode('desc')
        ids2_desc, emb2_desc = c2.encode('desc')
        ids1_poly, emb1_poly = c1.encode('polish')
        ids2_poly, emb2_poly = c2.encode('polish')

        # 2. Evaluate
        d1_desc = to_dict(ids1_desc, emb1_desc)
        d2_desc = to_dict(ids2_desc, emb2_desc)
        h_d, m_d = eval_alignment(d1_desc, d2_desc, test_pairs, device='cpu')
        d1_poly = to_dict(ids1_poly, emb1_poly)
        d2_poly = to_dict(ids2_poly, emb2_poly)
        h_p, m_p = eval_alignment(d1_poly, d2_poly, test_pairs, device='cpu')

        log.info(
            f"   🏆 Result R{r}: Desc H@1={h_d[1]:.2f}% | Polish H@1={h_p[1]:.2f}%")
        results.append(
            {"round": r, "desc_hits1": h_d[1], "desc_mrr": m_d, "poly_hits1": h_p[1], "poly_mrr": m_p})

        if r == rounds:
            break

        # 3. Strategy (Pseudo Labels)
        log.info(
            f"   Generating Pseudo-labels (Threshold={current_threshold:.4f})...")
        pairs_idx = PseudoLabelGenerator.generate(
            emb1_desc, emb2_desc, current_threshold, device='cpu')
        log.info(f"   🌱 Found {len(pairs_idx)} high-confidence pairs.")

        if len(pairs_idx) < 50:
            log.warning("   ⚠️ Too few pairs, skipping training.")
            continue

        # 4. Prepare & Train
        p_idx1 = [p[0] for p in pairs_idx]
        p_idx2 = [p[1] for p in pairs_idx]
        target_desc_c1 = emb2_desc[p_idx2]
        target_desc_c2 = emb1_desc[p_idx1]
        target_poly_c1 = emb2_poly[p_idx2]
        target_poly_c2 = emb1_poly[p_idx1]

        c1.prepare_training_data(p_idx1, target_desc_c1, target_poly_c1)
        c2.prepare_training_data(p_idx2, target_desc_c2, target_poly_for_c2)

        w1, l1 = c1.train()
        log.info(f"   📉 C1 Loss: {l1:.6f}")
        w2, l2 = c2.train()
        log.info(f"   📉 C2 Loss: {l2:.6f}")

        # 5. Aggregate
        server.aggregate([w1, w2])
        c1.model.load_state_dict(server.global_model.state_dict())
        c2.model.load_state_dict(server.global_model.state_dict())
        dm.clean_memory()

    if cfg.task.checkpoint.save_best:
        server.save_model(suffix=f"{text_mode}_round{rounds}")

    log_experiment_result(cfg.experiment_name,
                          cfg.data.name, results[-1], config=cfg)


def run_structure_workflow(cfg, server, c1, c2, test_pairs, dm):
    """
    Phase 2: 结构训练工作流 (GCN Training) - 复刻老版本互学习策略
    """
    results = []
    rounds = cfg.task.federated.rounds
    alpha = cfg.task.eval.alpha

    # [策略调整] 阈值从高到低 (0.8 -> 0.5)
    # 模拟老版本的: max(0.50, 0.80 - (it * 0.05))
    thresh_start = 0.80
    thresh_end = 0.50
    thresh_step = (thresh_start - thresh_end) / max(1, rounds - 1)

    # 1. 备份原始 SBERT (仅用于评估和语义过滤器，不作为训练强约束)
    fixed_sbert_1 = c1.anchor_embeddings.clone().cpu()
    fixed_sbert_2 = c2.anchor_embeddings.clone().cpu()

    # 评估用字典
    sb_emb1 = to_dict(c1.dataset.ids, fixed_sbert_1)
    sb_emb2 = to_dict(c2.dataset.ids, fixed_sbert_2)

    for r in range(rounds + 1):
        # 计算当前阈值 (Decaying)
        current_threshold = max(thresh_end, thresh_start - (r * thresh_step))

        # 动态 Epochs: Round 0 铺底 (100)，后续微调 (20)
        current_epochs = cfg.task.federated.local_epochs if r == 0 else 20

        log.info(
            f"\n{'='*40}\n🏗️  [Structure] Round {r}/{rounds} (Thresh: {current_threshold:.4f} | Epochs: {current_epochs})\n{'='*40}")

        # --- Step 1: 训练 (Train) ---
        log.info(f"   🚀 Training GCN on current anchors (Mutual Targets)...")
        w1, l1 = c1.train(custom_epochs=current_epochs)
        log.info(f"   📉 C1 Loss: {l1:.6f}")
        w2, l2 = c2.train(custom_epochs=current_epochs)
        log.info(f"   📉 C2 Loss: {l2:.6f}")

        # --- Step 2: 聚合 (Aggregate) ---
        server.aggregate([w1, w2])
        global_shared = server.get_global_weights()
        c1.model.load_shared_state_dict(global_shared)
        c2.model.load_shared_state_dict(global_shared)
        dm.clean_memory()

        # --- Step 3: 评估 (Score Fusion) ---
        struct_emb1 = c1.get_embeddings()
        struct_emb2 = c2.get_embeddings()

        st_dict1 = to_dict(c1.dataset.ids, struct_emb1)
        st_dict2 = to_dict(c2.dataset.ids, struct_emb2)

        log.info(f"   📊 Eval [Score Fusion Alpha={alpha}]...")
        # 评估始终以 Fixed SBERT 为基准，保持公平性
        h_f, m_f = eval_alignment(
            st_dict1, st_dict2, test_pairs,
            sbert1_dict=sb_emb1, sbert2_dict=sb_emb2,
            alpha=alpha, device='cpu'
        )
        log.info(f"   🏆 Result R{r}: Hits@1={h_f[1]:.2f}% | MRR={m_f:.4f}")
        results.append({"round": r, "hits1": h_f[1], "mrr": m_f})

        if r == rounds:
            break

        # --- Step 4: 挖掘 (Pure GCN) ---
        log.info(
            f"   💎 Generating Pseudo-labels (GCN Mining, Decaying Thresh={current_threshold:.4f})...")

        pairs_idx = PseudoLabelGenerator.generate(
            struct_emb1, struct_emb2,
            threshold=current_threshold, device='cpu'
        )

        # --- Step 5: 语义一致性过滤 (Safety Net) ---
        # 虽然我们想做互学习，但为了防止完全跑偏，加一个宽松的 SBERT 过滤器
        filtered_pairs = []
        semantic_filter_thresh = 0.25  # 比较宽松，允许一定的语义噪音，只要不太离谱

        if len(pairs_idx) > 0:
            p1 = torch.tensor([p[0] for p in pairs_idx])
            p2 = torch.tensor([p[1] for p in pairs_idx])

            s1_vecs = F.normalize(fixed_sbert_1[p1], p=2, dim=1)
            s2_vecs = F.normalize(fixed_sbert_2[p2], p=2, dim=1)
            sem_sims = (s1_vecs * s2_vecs).sum(dim=1)

            mask = sem_sims > semantic_filter_thresh
            valid_indices = torch.nonzero(mask).squeeze()

            if valid_indices.numel() > 0:
                if valid_indices.ndim == 0:
                    filtered_pairs.append(pairs_idx[valid_indices.item()])
                else:
                    for idx in valid_indices.tolist():
                        filtered_pairs.append(pairs_idx[idx])

            log.info(
                f"   🔍 Semantic Filter: {len(pairs_idx)} -> {len(filtered_pairs)} (Removed {len(pairs_idx)-len(filtered_pairs)})")
        else:
            filtered_pairs = []

        # --- Step 6: 互学习更新 (Co-training Update) ---
        if len(filtered_pairs) > 0:
            p_idx1 = [p[0] for p in filtered_pairs]
            p_idx2 = [p[1] for p in filtered_pairs]

            # [老版本逻辑复刻]
            # C1 的新目标 = C2 现在的 Structure Embedding
            # 这允许 C1 学习 C2 挖掘出的结构信息，而不仅仅是死记 SBERT
            new_anchors_for_c1 = struct_emb2[p_idx2]
            new_anchors_for_c2 = struct_emb1[p_idx1]

            c1.update_anchors(p_idx1, new_anchors_for_c1)
            c2.update_anchors(p_idx2, new_anchors_for_c2)

            log.info(
                f"   ✅ Anchors Expanded: +{len(filtered_pairs)} pairs (Targets = Peer Structure).")
        else:
            log.warning("   ⚠️ No new anchors found.")

    if cfg.task.checkpoint.save_best:
        # 获取当前使用的 encoder 名称 (gcn 或 gat)
        encoder_name = cfg.task.model.encoder_name

        # 1. 保存 Server (Global MLP) -> 加上 encoder 后缀
        server.save_model(suffix=f"structure_{encoder_name}_round{rounds}")

        # 2. 保存 Client 模型 (包含私有 GCN/GAT 参数)
        save_dir = cfg.task.checkpoint.save_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # 命名格式: c{id}_structure_round{rounds}.pth
        # [修改] 命名格式: c{id}_structure_{encoder}_round{rounds}.pth
        # 这样 GCN 和 GAT 的权重文件就会分开，不会覆盖
        c1_path = os.path.join(
            save_dir, f"c1_structure_{encoder_name}_round{rounds}.pth")
        c2_path = os.path.join(
            save_dir, f"c2_structure_{encoder_name}_round{rounds}.pth")

        # 获取包含 GCN+MLP 的完整状态字典
        # 注意: get_shared_state_dict 只返回 MLP，我们要用 state_dict() 获取全部
        torch.save(c1.model.state_dict(), c1_path)
        torch.save(c2.model.state_dict(), c2_path)

        log.info(f"💾 Saved full client models to:")
        log.info(f"   - {c1_path}")
        log.info(f"   - {c2_path}")

    log_experiment_result(cfg.experiment_name,
                          cfg.data.name, results[-1], config=cfg)


if __name__ == "__main__":
    os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
    main()
