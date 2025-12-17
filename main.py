# main.py
import hydra
from omegaconf import DictConfig, OmegaConf
import logging
import os
import torch
import torch.nn.functional as F
import json
import numpy as np

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
        log.info("🔹 Entering Phase 2: Structure Training (GAT Optimized)")
        log.info("🏗️ Initializing Structure Clients (Loading Frozen SBERT)...")
        c1 = ClientStructure("C1", cfg, task_data.source, dm)
        c2 = ClientStructure("C2", cfg, task_data.target, dm)
        run_structure_workflow(cfg, server, c1, c2, task_data.test_pairs, dm)

    else:
        log.error(f"❌ Unknown task type: {cfg.task.type}")


def run_sbert_workflow(cfg, server, c1, c2, test_pairs, dm):
    """
    Phase 1: SBERT 混合微调工作流 (保持原样)
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
        # Fixed minor variable name typo from previous version if any
        c2.prepare_training_data(p_idx2, target_desc_c2, target_poly_c2)

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
    Phase 2: Structure Alignment Workflow
    (Performance-Aware Curriculum Federation)
    """
    rounds = cfg.task.federated.rounds
    # 读取课程学习阈值，默认 0.70
    curriculum_thresh = cfg.task.federated.get('curriculum_thresh', 0.70)

    # 挖掘阈值衰减策略
    thresh_start = 0.85
    thresh_end = 0.60
    thresh_step = (thresh_start - thresh_end) / max(1, rounds - 1)

    best_hits1 = 0.0
    results_history = []

    # ---------------------------------------------------------
    # 0. 初始基准评估 (SBERT Baseline)
    # ---------------------------------------------------------
    log.info("\n" + "="*60)
    log.info("📊 Baseline Evaluation: SBERT (Before Structure Training)")
    log.info("="*60)

    # 获取纯 SBERT 特征
    s1_base = F.normalize(c1.anchor_embeddings, p=2, dim=1)
    s2_base = F.normalize(c2.anchor_embeddings, p=2, dim=1)

    d1_base = {id: s1_base[i] for i, id in enumerate(c1.dataset.ids)}
    d2_base = {id: s2_base[i] for i, id in enumerate(c2.dataset.ids)}

    # alpha=0.0 代表纯 SBERT 评估
    h_base, mrr_base = eval_alignment(
        d1_base, d2_base, test_pairs, k_values=[1], alpha=0.0)
    log.info(f"🏆 SBERT Baseline: Hits@1={h_base[1]:.2f}% | MRR={mrr_base:.4f}")
    log.info("   (Target: Structure model should try to match this fidelity first!)\n")

    # ---------------------------------------------------------
    # 联邦训练循环
    # ---------------------------------------------------------
    for r in range(rounds + 1):
        # 计算当前挖掘阈值
        curr_mining_thresh = max(thresh_end, thresh_start - (r * thresh_step))
        # 动态 Epochs: Round 0 需要多跑一会来热身
        curr_epochs = 100 if r == 0 else cfg.task.federated.local_epochs

        log.info(f"\n{'='*40}")
        log.info(f"🏗️  [Structure] Round {r}/{rounds}")
        log.info(f"{'='*40}")

        # --- Step 1: Local Training ---
        log.info(
            f"🚀 Training {cfg.task.model.encoder_name.upper()} (Target=SBERT/Peer)...")

        # 接收: 权重, Loss, 以及 [Internal Fidelity]
        w1, l1, fid1 = c1.train(custom_epochs=curr_epochs)
        w2, l2, fid2 = c2.train(custom_epochs=curr_epochs)

        avg_fidelity = (fid1 + fid2) / 2
        log.info(f"   📉 Loss: C1={l1:.4f} | C2={l2:.4f}")
        log.info(
            f"   🎓 Internal Fidelity: C1={fid1:.3f} | C2={fid2:.3f} | Avg={avg_fidelity:.3f}")
        log.info(f"      (Curriculum Threshold: {curriculum_thresh})")

        # --- Step 2: Aggregation ---
        # log.info("🔗 Aggregating Shared Weights...")
        server.aggregate([w1, w2])
        global_weights = server.get_global_weights()

        # 分发更新
        c1.model.load_shared_state_dict(global_weights)
        c2.model.load_shared_state_dict(global_weights)

        # --- Step 3: Dual Evaluation (双重评估) ---
        log.info(f"📊 Round {r} Evaluation...")

        c1.model.to(c1.device).eval()
        c2.model.to(c2.device).eval()

        with torch.no_grad():
            # A. 获取纯结构特征 (Pure Structure)
            emb1_struct = F.normalize(
                c1.model(c1.adj, c1.edge_types), p=2, dim=1)
            emb2_struct = F.normalize(
                c2.model(c2.adj, c2.edge_types), p=2, dim=1)

            # B. 获取 SBERT 特征 (Anchors)
            emb1_sbert = F.normalize(
                c1.anchor_embeddings.to(c1.device), p=2, dim=1)
            emb2_sbert = F.normalize(
                c2.anchor_embeddings.to(c2.device), p=2, dim=1)

            # C. 融合特征 (Gate 辅助推理)
            # 只有在推理时才进行融合！
            emb1_fused, alpha1 = c1.model.fuse(emb1_struct, emb1_sbert)
            emb2_fused, alpha2 = c2.model.fuse(emb2_struct, emb2_sbert)

            # 打印 Gate 的倾向性
            log.info(
                f"      [Gate Stats] C1_Struct_Weight: {alpha1.mean():.3f} | C2_Struct_Weight: {alpha2.mean():.3f}")

            # 准备字典用于 eval_alignment
            d1_s = {id: emb1_struct[i].cpu()
                    for i, id in enumerate(c1.dataset.ids)}
            d2_s = {id: emb2_struct[i].cpu()
                    for i, id in enumerate(c2.dataset.ids)}

            d1_f = {id: emb1_fused[i].cpu()
                    for i, id in enumerate(c1.dataset.ids)}
            d2_f = {id: emb2_fused[i].cpu()
                    for i, id in enumerate(c2.dataset.ids)}

        # 清理显存
        c1.model.to('cpu')
        c2.model.to('cpu')
        if dm.is_offload_enabled():
            dm.clean_memory()

        # 3.1 评估纯结构 (Student Grade) -> 看 GAT 学得怎么样
        h_s, mrr_s = eval_alignment(
            d1_s, d2_s, test_pairs, k_values=[1], alpha=1.0)

        # 3.2 评估融合效果 (Final Grade) -> 实际部署效果
        h_f, mrr_f = eval_alignment(
            d1_f, d2_f, test_pairs, k_values=[1, 10], alpha=1.0)

        log.info(
            f"   🔹 [Pure Structure] Hits@1={h_s[1]:.2f}% | MRR={mrr_s:.4f}")
        log.info(
            f"   🏆 [Fused Model   ] Hits@1={h_f[1]:.2f}% | Hits@10={h_f[10]:.2f}%")

        # 记录结果
        results_history.append({
            "round": r,
            "fidelity": avg_fidelity,
            "pure_hits1": h_s[1],
            "fused_hits1": h_f[1],
            "fused_hits10": h_f[10]
        })

        if h_f[1] > best_hits1:
            best_hits1 = h_f[1]
            server.save_model("best")

        if r == rounds:
            break

        # --- Step 4: Curriculum-Controlled Mining (课程学习控制) ---
        # 核心逻辑：只有当 Fidelity > Thresh 时，才认为 Structure 模型“懂了”，允许它去挖掘
        if avg_fidelity < curriculum_thresh:
            log.warning(
                f"   ⚠️ Fidelity ({avg_fidelity:.3f}) < Thresh ({curriculum_thresh}). Skipping Mining.")
            log.info("      (Student is not ready yet. Continuing Imitation...)")
            continue

        log.info(
            f"   💎 Fidelity Passed! Generating Pseudo-labels (Thresh={curr_mining_thresh:.4f})...")

        # 使用【融合特征】进行挖掘，因为它是最强的
        new_pairs = PseudoLabelGenerator.generate(
            emb1_fused.cpu(), emb2_fused.cpu(),
            threshold=curr_mining_thresh,
            device='cpu'
        )

        if len(new_pairs) > 0:
            # 互学习：如果 Structure 认为 A-B 对齐，则更新 SBERT Anchor
            # 这样下一轮 GAT 就会被强迫去拟合这个新的、带有结构信息的目标
            src_idx = [p[0] for p in new_pairs]
            tgt_idx = [p[1] for p in new_pairs]

            # C1 的新目标是 C2 的 Embedding
            c1.update_anchors(src_idx, emb2_fused[tgt_idx].cpu())
            # C2 的新目标是 C1 的 Embedding
            c2.update_anchors(tgt_idx, emb1_fused[src_idx].cpu())

            log.info(f"   ✅ Anchors Updated: {len(new_pairs)} pairs injected.")
        else:
            log.info("   ⚠️ No reliable pairs found.")

    log.info(f"🏁 Final Best Hits@1: {best_hits1:.2f}%")

    # 保存训练历史
    history_path = os.path.join(os.getcwd(), "training_history.json")
    with open(history_path, 'w') as f:
        json.dump(results_history, f, indent=4)

    res = {
        "dataset": cfg.data.name,
        "mode": "structure",
        "best_hits1": best_hits1
    }
    log_experiment_result("structure_phase2", cfg.data.name, res)


if __name__ == "__main__":
    os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
    main()
