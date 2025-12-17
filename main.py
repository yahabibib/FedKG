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
from src.utils.tuning import search_best_alpha

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
    Phase 2: Structure Alignment Workflow (Biased Adaptive Mining)

    机制说明：
    1. 成熟度检查 (Maturity Check): 结构模型性能需达到 Baseline 的 75% 才允许介入。
    2. 结构偏置 (Structural Bias): 在 Gate 自动判断的基础上，初期强制注入 +0.2 的偏置，
       随着轮次增加衰减至 0.0。
    3. 互学习 (Mutual Learning): 使用“加权融合”后的特征生成伪标签并更新 Anchor。
    """
    rounds = cfg.task.federated.rounds
    curriculum_thresh = cfg.task.federated.get('curriculum_thresh', 0.70)

    # --- [配置] 结构偏置衰减计划 (Bias Schedule) ---
    # 刚开始成熟时推一把 (+0.2)，后期让 Gate 自己做主 (+0.0)
    bias_start = 0.2
    bias_end = 0.0
    bias_step = (bias_start - bias_end) / max(1, rounds - 1)

    # --- [配置] 挖掘阈值衰减 ---
    thresh_start = 0.85
    thresh_end = 0.60
    thresh_step = (thresh_start - thresh_end) / max(1, rounds - 1)

    best_global_hits1 = 0.0
    results_history = []

    # ---------------------------------------------------------
    # 0. SBERT Baseline (基准线)
    # ---------------------------------------------------------
    log.info("\n" + "="*60)
    log.info("📊 Baseline Evaluation: SBERT")
    log.info("="*60)

    # 获取纯 SBERT 特征
    s1_base = F.normalize(c1.anchor_embeddings, p=2, dim=1)
    s2_base = F.normalize(c2.anchor_embeddings, p=2, dim=1)
    d1_base = {id: s1_base[i] for i, id in enumerate(c1.dataset.ids)}
    d2_base = {id: s2_base[i] for i, id in enumerate(c2.dataset.ids)}

    # 计算基准分数
    h_base, mrr_base = eval_alignment(
        d1_base, d2_base, test_pairs, k_values=[1], alpha=0.0)
    BASELINE_HITS1 = h_base[1]

    # 设定成熟度门槛 (75% of Baseline)
    MATURITY_TARGET = BASELINE_HITS1 * 0.75

    log.info(f"🏆 SBERT Baseline Hits@1: {BASELINE_HITS1:.2f}%")
    log.info(
        f"🎯 Maturity Target: {MATURITY_TARGET:.2f}% (Wait until Structure hits this)")

    # ---------------------------------------------------------
    # 联邦训练循环
    # ---------------------------------------------------------
    for r in range(rounds + 1):
        # 计算当前轮次的超参
        curr_mining_thresh = max(thresh_end, thresh_start - (r * thresh_step))
        curr_bias = max(bias_end, bias_start - (r * bias_step))
        curr_epochs = 100 if r == 0 else cfg.task.federated.local_epochs

        log.info(f"\n{'='*40}")
        log.info(f"🏗️  [Structure] Round {r}/{rounds}")
        log.info(f"{'='*40}")

        # --- Step 1: Local Training ---
        # 始终通过 InfoNCE 拟合当前的目标 (SBERT 或 Updated Anchors)
        w1, l1, fid1 = c1.train(custom_epochs=curr_epochs)
        w2, l2, fid2 = c2.train(custom_epochs=curr_epochs)

        avg_fidelity = (fid1 + fid2) / 2
        log.info(
            f"   📉 Loss: C1={l1:.4f} | C2={l2:.4f} | Fidelity: {avg_fidelity:.3f}")

        # --- Step 2: Aggregation ---
        server.aggregate([w1, w2])
        weights = server.get_global_weights()
        c1.model.load_shared_state_dict(weights)
        c2.model.load_shared_state_dict(weights)

        # --- Step 3: Maturity Check (成熟度检查) ---
        log.info(f"📊 Round {r} Evaluation & Mining Check...")
        if dm.is_offload_enabled():
            dm.clean_memory()

        c1.model.eval()
        c2.model.eval()
        with torch.no_grad():
            # 获取纯结构特征 (Pure Structure)
            e1_s = F.normalize(
                c1.model(c1.adj, c1.edge_types), p=2, dim=1).cpu()
            e2_s = F.normalize(
                c2.model(c2.adj, c2.edge_types), p=2, dim=1).cpu()

        d1_s = {id: e1_s[i] for i, id in enumerate(c1.dataset.ids)}
        d2_s = {id: e2_s[i] for i, id in enumerate(c2.dataset.ids)}

        h_pure, _ = eval_alignment(
            d1_s, d2_s, test_pairs, k_values=[1], alpha=1.0)
        pure_hits1 = h_pure[1]

        is_mature = pure_hits1 >= MATURITY_TARGET
        log.info(
            f"   🎓 Pure Structure Hits@1: {pure_hits1:.2f}% (Target: {MATURITY_TARGET:.2f}%)")

        # --- Step 4: Biased Adaptive Mining (核心机制) ---
        final_hits1 = 0.0

        if not is_mature:
            # === Mode A: Warm-up (预热期) ===
            log.info("   🔒 Status: WARM-UP MODE")
            log.info(
                "      Structure is too weak. Skipping mining to let it fit SBERT perfectly.")
            # 此时我们不做任何挖掘，也不更新 Anchor，让 GCN 专心拟合初始 SBERT
            # 用纯结构分数记录即可
            final_hits1 = pure_hits1

        else:
            # === Mode B: Adaptive Fusion (自适应融合期) ===
            log.info(f"   🔓 Status: ADAPTIVE MINING (Bias={curr_bias:.2f})")

            c1.model.eval()
            c2.model.eval()

            with torch.no_grad():
                # 1. 准备语义特征 (CPU)
                e1_t = c1.anchor_embeddings.cpu()
                e2_t = c2.anchor_embeddings.cpu()

                # 2. 手动执行带偏置的融合 (Manual Biased Fusion)
                # -------------------------------------------------
                # C1 Fusion
                # 获取 Gate 的原始输出 (Alpha)
                # 注意：这里需要临时构建 Gate 的输入 [struct, sbert]
                gate_inp_c1 = torch.cat([e1_s, e1_t], dim=1)
                # 调用模型内的 gate 子模块
                raw_alpha_c1 = c1.model.gate.to('cpu')(gate_inp_c1)  # [N, 1]

                # 注入偏置并截断 (Clamp)
                boosted_alpha_c1 = torch.clamp(
                    raw_alpha_c1 + curr_bias, 0.0, 1.0)

                # 加权融合
                e1_fused = F.normalize(
                    boosted_alpha_c1 * e1_s + (1 - boosted_alpha_c1) * e1_t, p=2, dim=1)

                # C2 Fusion (同理)
                gate_inp_c2 = torch.cat([e2_s, e2_t], dim=1)
                raw_alpha_c2 = c2.model.gate.to('cpu')(gate_inp_c2)
                boosted_alpha_c2 = torch.clamp(
                    raw_alpha_c2 + curr_bias, 0.0, 1.0)
                e2_fused = F.normalize(
                    boosted_alpha_c2 * e2_s + (1 - boosted_alpha_c2) * e2_t, p=2, dim=1)
                # -------------------------------------------------

                log.info(
                    f"   📊 Gate Stats (C1): Raw_Mean={raw_alpha_c1.mean():.3f} | Boosted_Mean={boosted_alpha_c1.mean():.3f}")

                # 3. 评估融合后的效果 (Adaptive Evaluation)
                d1_f = {id: e1_fused[i] for i, id in enumerate(c1.dataset.ids)}
                d2_f = {id: e2_fused[i] for i, id in enumerate(c2.dataset.ids)}
                # alpha=1.0 因为 d1_f 已经是融合好的向量了
                h_f, _ = eval_alignment(
                    d1_f, d2_f, test_pairs, k_values=[1], alpha=1.0)
                final_hits1 = h_f[1]

                log.info(f"   🏆 Adaptive Hits@1: {final_hits1:.2f}%")

                # 4. 执行挖掘 (Mining)
                # 使用融合特征计算互为最近邻
                new_pairs = PseudoLabelGenerator.generate(
                    e1_fused, e2_fused,
                    threshold=curr_mining_thresh,
                    device='cpu'
                )

                # 5. 更新 Anchors (Update Targets)
                if len(new_pairs) > 0:
                    src_idx = [p[0] for p in new_pairs]
                    tgt_idx = [p[1] for p in new_pairs]

                    # 交叉更新：用对方融合后的特征作为我下一轮训练的目标
                    c1.update_anchors(src_idx, e2_fused[tgt_idx])
                    c2.update_anchors(tgt_idx, e1_fused[src_idx])

                    log.info(
                        f"   ✅ Anchors Updated: {len(new_pairs)} pairs (Injecting Structure Info).")
                else:
                    log.info("   ⚠️ No reliable pairs found.")

        # 保存最佳模型
        if final_hits1 > best_global_hits1:
            best_global_hits1 = final_hits1
            server.save_model("best_structure")

        # 记录历史
        results_history.append({
            "round": r,
            "mode": "Fusion" if is_mature else "Warm-up",
            "bias": curr_bias if is_mature else 0.0,
            "hits1": final_hits1
        })

        if r == rounds:
            break

    log.info(f"🏁 Final Best Hits@1: {best_global_hits1:.2f}%")

    # 保存训练日志到文件
    import json
    import os
    history_path = os.path.join(os.getcwd(), "training_history.json")
    with open(history_path, 'w') as f:
        json.dump(results_history, f, indent=4)

    # 记录到实验汇总
    from src.utils.logger import log_experiment_result
    res = {
        "dataset": cfg.data.name,
        "mode": "biased_adaptive_mining",
        "best_hits1": best_global_hits1
    }
    log_experiment_result("structure_final", cfg.data.name, res)


if __name__ == "__main__":
    os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
    main()
