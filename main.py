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
    Phase 2: 结构训练工作流 (GCN Training) - 终极修正版 (含语义一致性过滤)
    策略：
    1. 训练：基于当前锚点训练 GCN (动态轮次 + 早停)。
    2. 评估：使用 Score Fusion (GCN + SBERT 加权)。
    3. 挖掘：使用 Pure GCN 挖掘结构相似对。
    4. 过滤：使用 Fixed SBERT 进行语义一致性检查，剔除结构假阳性。
    5. 更新：将锚点锁定回 Fixed SBERT 语义空间。
    """
    results = []
    rounds = cfg.task.federated.rounds
    alpha = cfg.task.eval.alpha
    base_threshold = cfg.task.strategy.pseudo_threshold
    threshold_start = 0.70

    # --- [关键] 1. 锁定原始语义坐标 (Source of Truth) ---
    # 在任何更新发生前，克隆一份原始的 SBERT Anchors
    # 这些是我们的"北极星"，永远不应该被修改
    fixed_sbert_1 = c1.anchor_embeddings.clone().cpu()
    fixed_sbert_2 = c2.anchor_embeddings.clone().cpu()

    # 转换为字典，供评估函数使用
    sb_emb1 = to_dict(c1.dataset.ids, fixed_sbert_1)
    sb_emb2 = to_dict(c2.dataset.ids, fixed_sbert_2)

    for r in range(rounds + 1):
        # 动态阈值 & 动态轮次
        current_threshold = get_dynamic_threshold(
            threshold_start, base_threshold, rounds, r)
        # Round 0 给足 100 轮让 GCN 收敛，后续轮次只需 20 轮微调
        current_epochs = cfg.task.federated.local_epochs if r == 0 else 20

        log.info(
            f"\n{'='*40}\n🏗️  [Structure] Round {r}/{rounds} (Thresh: {current_threshold:.4f} | Epochs: {current_epochs})\n{'='*40}")

        # ===============================================
        # Step 1: 训练 (Train)
        # ===============================================
        log.info(f"   🚀 Training GCN on current anchors...")
        w1, l1 = c1.train(custom_epochs=current_epochs)
        log.info(f"   📉 C1 Loss: {l1:.6f}")
        w2, l2 = c2.train(custom_epochs=current_epochs)
        log.info(f"   📉 C2 Loss: {l2:.6f}")

        # ===============================================
        # Step 2: 聚合 (Aggregate)
        # ===============================================
        server.aggregate([w1, w2])
        global_shared = server.get_global_weights()
        c1.model.load_shared_state_dict(global_shared)
        c2.model.load_shared_state_dict(global_shared)
        dm.clean_memory()

        # ===============================================
        # Step 3: 评估 (Eval - Score Fusion)
        # ===============================================
        struct_emb1 = c1.get_embeddings()
        struct_emb2 = c2.get_embeddings()

        st_dict1 = to_dict(c1.dataset.ids, struct_emb1)
        st_dict2 = to_dict(c2.dataset.ids, struct_emb2)

        log.info(f"   📊 Eval [Score Fusion Alpha={alpha}]...")
        # 传入 fixed SBERT 字典，确保评估标准统一
        h_f, m_f = eval_alignment(
            st_dict1, st_dict2, test_pairs,
            sbert1_dict=sb_emb1, sbert2_dict=sb_emb2,
            alpha=alpha, device='cpu'
        )
        log.info(f"   🏆 Result R{r}: Hits@1={h_f[1]:.2f}% | MRR={m_f:.4f}")
        results.append({"round": r, "hits1": h_f[1], "mrr": m_f})

        if r == rounds:
            break

        # ===============================================
        # Step 4: 挖掘 (Mine - Pure GCN)
        # ===============================================
        log.info(
            f"   💎 Generating Pseudo-labels (GCN Mining, Thresh={current_threshold:.4f})...")

        # 使用 GCN 结构向量发现潜在对齐
        # 这一步是为了找出那些 "SBERT 没看出来，但结构上很像" 的实体
        pairs_idx = PseudoLabelGenerator.generate(
            struct_emb1, struct_emb2,
            threshold=current_threshold, device='cpu'
        )

        # ===============================================
        # Step 5: 语义一致性过滤 (Semantic Filter)
        # ===============================================
        filtered_pairs = []
        # 语义底线：如果 SBERT 相似度低于 0.3，说明语义完全不沾边，判定为结构假阳性
        semantic_filter_thresh = 0.3

        if len(pairs_idx) > 0:
            # 批量操作加速
            p1 = torch.tensor([p[0] for p in pairs_idx])
            p2 = torch.tensor([p[1] for p in pairs_idx])

            # 取出对应的 Fixed SBERT 向量
            s1_vecs = F.normalize(fixed_sbert_1[p1], p=2, dim=1)
            s2_vecs = F.normalize(fixed_sbert_2[p2], p=2, dim=1)

            # 计算成对余弦相似度
            sem_sims = (s1_vecs * s2_vecs).sum(dim=1)

            # 过滤：保留语义相似度 > 0.3 的对子
            mask = sem_sims > semantic_filter_thresh
            valid_indices = torch.nonzero(mask).squeeze()

            # 处理 Tensor 维度边缘情况
            if valid_indices.numel() > 0:
                if valid_indices.ndim == 0:  # 只有一个元素时
                    filtered_pairs.append(pairs_idx[valid_indices.item()])
                else:
                    for idx in valid_indices.tolist():
                        filtered_pairs.append(pairs_idx[idx])

            removed_count = len(pairs_idx) - len(filtered_pairs)
            log.info(
                f"   🔍 Semantic Filter: Checked {len(pairs_idx)} pairs -> Kept {len(filtered_pairs)} pairs. (Removed {removed_count} noise)")
        else:
            log.warning("   ⚠️ No structural pairs found to filter.")

        # ===============================================
        # Step 6: 更新锚点 (Update - Lock to SBERT)
        # ===============================================
        if len(filtered_pairs) > 0:
            p_idx1 = [p[0] for p in filtered_pairs]
            p_idx2 = [p[1] for p in filtered_pairs]

            # C1 的新目标：既然 p_idx1 对应 p_idx2，那就让 p_idx1 去学 p_idx2 的 SBERT 向量
            # 这样保证了目标永远在语义空间内，不会发生 GCN 互卷导致的漂移
            new_anchors_for_c1 = fixed_sbert_2[p_idx2]
            new_anchors_for_c2 = fixed_sbert_1[p_idx1]

            c1.update_anchors(p_idx1, new_anchors_for_c1)
            c2.update_anchors(p_idx2, new_anchors_for_c2)

            log.info(
                f"   ✅ Anchors Expanded: +{len(filtered_pairs)} pairs (Targets locked to Fixed SBERT).")
        else:
            log.warning("   ⚠️ No new anchors added this round.")

    # Save
    if cfg.task.checkpoint.save_best:
        server.save_model(suffix=f"structure_round{rounds}")
    log_experiment_result(cfg.experiment_name,
                          cfg.data.name, results[-1], config=cfg)


if __name__ == "__main__":
    os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
    main()
