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
    Phase 2: 结构训练工作流 (GCN Training) - 修复版
    逻辑顺序: Train(基于现有锚点) -> Aggregate -> Eval -> Mine(为下一轮准备)
    """
    results = []
    rounds = cfg.task.federated.rounds
    alpha = cfg.task.eval.alpha
    base_threshold = cfg.task.strategy.pseudo_threshold

    # 降低起始阈值，给第一轮更多机会
    threshold_start = 0.70

    # 获取初始 SBERT 锚点用于融合计算
    sb_emb1 = to_dict(c1.dataset.ids, c1.anchor_embeddings)
    sb_emb2 = to_dict(c2.dataset.ids, c2.anchor_embeddings)

    for r in range(rounds + 1):
        # 动态计算阈值
        current_threshold = get_dynamic_threshold(
            threshold_start, base_threshold, rounds, r)

        log.info(
            f"\n{'='*40}\n🏗️  [Structure] Round {r}/{rounds} (Thresh: {current_threshold:.4f})\n{'='*40}")

        # =================================================
        # 1. 训练 (Train) & 聚合 (Aggregate)
        # =================================================
        # [核心修复] 无论是否挖掘到新标签，都必须基于现有锚点训练 GCN
        # Round 0 时，这里会利用初始的 SBERT 锚点把随机 GCN 训练成有意义的形状
        log.info("   🚀 Training GCN on current anchors...")
        w1, l1 = c1.train()
        log.info(f"   📉 C1 Loss: {l1:.6f}")
        w2, l2 = c2.train()
        log.info(f"   📉 C2 Loss: {l2:.6f}")

        # 聚合更新
        server.aggregate([w1, w2])
        global_shared = server.get_global_weights()
        c1.model.load_shared_state_dict(global_shared)
        c2.model.load_shared_state_dict(global_shared)

        # 清理显存
        dm.clean_memory()

        # =================================================
        # 2. 评估 (Eval)
        # =================================================
        # 获取训练后的 Embeddings
        struct_emb1 = c1.get_embeddings()
        struct_emb2 = c2.get_embeddings()

        st_dict1 = to_dict(c1.dataset.ids, struct_emb1)
        st_dict2 = to_dict(c2.dataset.ids, struct_emb2)

        log.info(f"   📊 Eval [Fusion Alpha={alpha}]...")
        fused_1 = _fuse_embeddings(st_dict1, sb_emb1, alpha)
        fused_2 = _fuse_embeddings(st_dict2, sb_emb2, alpha)

        h_f, m_f = eval_alignment(fused_1, fused_2, test_pairs, device='cpu')

        # 记录结果
        log.info(f"   🏆 Result R{r}: Hits@1={h_f[1]:.2f}% | MRR={m_f:.4f}")
        results.append({"round": r, "hits1": h_f[1], "mrr": m_f})

        if r == rounds:
            break

        # =================================================
        # 3. 策略挖掘 (Mine & Update) - 为下一轮做准备
        # =================================================
        log.info(
            f"   융 Generating Fusion-driven Pseudo-labels (Alpha={alpha}, Thresh={current_threshold:.4f})...")

        # 使用训练后的 Struct Embedding 进行挖掘
        fusion_pairs_idx = PseudoLabelGenerator.generate_fusion(
            struct_emb1, c1.anchor_embeddings,
            struct_emb2, c2.anchor_embeddings,
            alpha=alpha,
            threshold=current_threshold,
            device='cpu'
        )

        log.info(
            f"   Found {len(fusion_pairs_idx)} new pairs for anchor expansion.")

        if len(fusion_pairs_idx) > 0:
            p_idx1 = [p[0] for p in fusion_pairs_idx]
            p_idx2 = [p[1] for p in fusion_pairs_idx]

            # 提取对端训练好的 Structure Embedding 作为新的 Teacher 信号
            # 注意：这里使用 struct_emb (GCN输出) 还是 fused (融合) 取决于策略
            # 原论文通常是用 Structure Embedding 做互监督
            new_anchors_for_c1 = struct_emb2[p_idx2]
            new_anchors_for_c2 = struct_emb1[p_idx1]

            # 更新 Client 的本地锚点集 (这会影响下一轮的 Train)
            c1.update_anchors(p_idx1, new_anchors_for_c1)
            c2.update_anchors(p_idx2, new_anchors_for_c2)
        else:
            log.info(
                "   ⚠️ No new anchors found, keeping previous anchors for next round.")

    # Save and Log (Final)
    if cfg.task.checkpoint.save_best:
        server.save_model(suffix=f"structure_round{rounds}")
    log_experiment_result(cfg.experiment_name,
                          cfg.data.name, results[-1], config=cfg)


if __name__ == "__main__":
    os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
    main()
