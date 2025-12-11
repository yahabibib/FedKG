import hydra
from omegaconf import DictConfig, OmegaConf
import logging
import os
import torch

# --- 导入我们重构后的核心组件 ---
# 1. 数据层
from src.data.dataset import AlignmentTaskData
# 2. 工具层
from src.utils.device_manager import DeviceManager
from src.utils.metrics import eval_alignment
from src.utils.logger import log_experiment_result
# 3. 联邦层
from src.federation.server import Server
from src.federation.client_sbert import ClientSBERT
from src.federation.strategy import PseudoLabelGenerator

# 获取 Hydra 的 logger，它会自动将日志输出到 outputs/日期/时间/main.log
log = logging.getLogger(__name__)


@hydra.main(config_path="configs", config_name="config", version_base="1.2")
def main(cfg: DictConfig):
    """
    FedAnchor 2.0 主入口
    """
    # 1. 打印实验元信息
    log.info(f"🚀 Starting Experiment: {cfg.experiment_name}")
    log.info(
        f"⚙️  Task Type: {cfg.task.type} | Mode: {cfg.task.strategy.text_mode}")
    log.info(
        f"💻 Device Strategy: {cfg.system.device} (Offload: {cfg.system.memory.offload_to_cpu})")

    # 打印完整配置方便调试 (可选)
    # log.info(f"\n{OmegaConf.to_yaml(cfg)}")

    # 2. 初始化设备管理器 (处理 MPS/CUDA 和显存策略)
    dm = DeviceManager(cfg.system)

    # 3. 加载数据 (AlignmentTaskData 会自动解析 source/target 配置)
    log.info("📚 Loading Datasets...")
    try:
        # 这里会自动加载 ent_ids, pairs, desc, polish 等所有文件
        task_data = AlignmentTaskData(cfg.data)
    except FileNotFoundError as e:
        log.error(f"❌ Data loading failed: {e}")
        return
    except Exception as e:
        log.exception(f"❌ Unexpected error during data loading: {e}")
        return

    # 4. 初始化联邦组件
    # Server: 负责聚合，常驻 CPU
    server = Server(cfg)

    # Client: 负责训练，根据 DeviceManager 策略使用 GPU/MPS
    # 注意：我们将 task_data.source (KGDataset) 传给 C1，task_data.target 传给 C2
    c1 = ClientSBERT("C1", cfg, task_data.source, dm)
    c2 = ClientSBERT("C2", cfg, task_data.target, dm)

    # 5. 任务分发 (Task Dispatch)
    # 根据 config.yaml 中的 task.type 决定运行哪个流程
    if cfg.task.type == 'sbert':
        run_sbert_workflow(cfg, server, c1, c2, task_data.test_pairs, dm)
    elif cfg.task.type == 'structure':
        log.info("🚧 Structure workflow (GCN) is under construction...")
        # run_structure_workflow(cfg, server, c1, c2, task_data)
    else:
        log.error(f"❌ Unknown task type: {cfg.task.type}")


def run_sbert_workflow(cfg, server, c1, c2, test_pairs, dm):
    """
    SBERT 混合微调主流程 (SBERT Mixed Fine-tuning Workflow)
    """
    results = []
    rounds = cfg.task.federated.rounds
    threshold = cfg.task.strategy.pseudo_threshold
    text_mode = cfg.task.strategy.text_mode

    for r in range(rounds + 1):
        log.info(f"\n{'='*40}\n🔄 Round {r}/{rounds} [{text_mode}]\n{'='*40}")

        # --- Step 1: 编码 (Encode) ---
        log.info("   Encoding Entities...")

        # 1.1 编码 Description (作为主要的语义锚点)
        ids1_desc, emb1_desc = c1.encode('desc')
        ids2_desc, emb2_desc = c2.encode('desc')

        # 1.2 编码 Polished (作为结构化文本的对照组，或混合训练的素材)
        ids1_poly, emb1_poly = c1.encode('polish')
        ids2_poly, emb2_poly = c2.encode('polish')

        # --- Step 2: 评估 (Evaluate) ---
        # 我们进行双重评估，既看模型对自然语言(Desc)的理解，也看对结构化文本(Polish)的理解

        # 2.1 评估 Description
        log.info("   📊 Eval [Description Input]...")
        # 构建 {id: tensor} 字典供 eval_alignment 使用
        d1_desc = {id: emb1_desc[i] for i, id in enumerate(ids1_desc)}
        d2_desc = {id: emb2_desc[i] for i, id in enumerate(ids2_desc)}

        h_d, m_d = eval_alignment(d1_desc, d2_desc, test_pairs, device='cpu')

        # 2.2 评估 Polished
        log.info("   📊 Eval [Polished Input]...")
        d1_poly = {id: emb1_poly[i] for i, id in enumerate(ids1_poly)}
        d2_poly = {id: emb2_poly[i] for i, id in enumerate(ids2_poly)}

        h_p, m_p = eval_alignment(d1_poly, d2_poly, test_pairs, device='cpu')

        # 打印并收集结果
        log.info(
            f"   🏆 Result R{r}: Desc H@1={h_d[1]:.2f}% | Polish H@1={h_p[1]:.2f}%")

        current_metrics = {
            "round": r,
            "desc_hits1": h_d[1], "desc_mrr": m_d,
            "poly_hits1": h_p[1], "poly_mrr": m_p
        }
        results.append(current_metrics)

        # 如果是最后一轮，评估完就结束，不进行训练
        if r == rounds:
            break

        # --- Step 3: 策略 - 生成伪标签 (Strategy) ---
        log.info(f"   Generating Pseudo-labels (Threshold={threshold})...")

        # 核心逻辑：我们始终信任 Description 生成的伪标签，因为它的语义质量最高 (Zero-shot 58% vs 41%)
        # 使用 src.federation.strategy.PseudoLabelGenerator
        pairs_idx = PseudoLabelGenerator.generate(
            emb1_desc, emb2_desc, threshold, device='cpu')

        log.info(f"   🌱 Found {len(pairs_idx)} high-confidence pairs.")

        # 安全检查：如果伪标签太少，强行训练会导致过拟合或崩塌
        if len(pairs_idx) < 50:
            log.warning(
                "   ⚠️ Too few pairs (<50), skipping training this round.")
            continue

        # --- Step 4: 准备训练数据 (Data Preparation) ---
        # pairs_idx 是 emb1_desc 和 emb2_desc 的索引对 (index)
        p_idx1 = [p[0] for p in pairs_idx]
        p_idx2 = [p[1] for p in pairs_idx]

        # 提取交叉目标 (Cross-target): C1 学习 C2 的特征，C2 学习 C1 的特征

        # 目标 A: Description Embedding (强语义)
        target_desc_for_c1 = emb2_desc[p_idx2]
        target_desc_for_c2 = emb1_desc[p_idx1]

        # 目标 B: Polished Embedding (强结构) - 用于混合训练
        target_poly_for_c1 = emb2_poly[p_idx2]
        target_poly_for_c2 = emb1_poly[p_idx1]

        # 通知 Client 准备 DataLoader
        # Client 内部会根据 cfg.task.strategy.text_mode 决定如何混合这些数据
        c1.prepare_training_data(
            p_idx1, target_desc_for_c1, target_poly_for_c1)
        c2.prepare_training_data(
            p_idx2, target_desc_for_c2, target_poly_for_c2)

        # --- Step 5: 本地训练 (Local Training) ---
        # 串行训练：C1 上 GPU -> 练完 -> 下 GPU -> C2 上 GPU
        # DeviceManager 会在 Client 内部自动管理显存

        w1, l1 = c1.train()
        log.info(f"   📉 C1 Loss: {l1:.6f}")

        w2, l2 = c2.train()
        log.info(f"   📉 C2 Loss: {l2:.6f}")

        # --- Step 6: 聚合 (Aggregation) ---
        # Server 执行 FedAvg
        server.aggregate([w1, w2])

        # 分发全局参数
        global_weights = server.get_global_weights()
        c1.model.load_state_dict(global_weights)
        c2.model.load_state_dict(global_weights)

        # 强制显存清理 (Double Check)
        dm.clean_memory()

    # --- 流程结束 ---

    # 1. 保存最佳模型
    if cfg.task.checkpoint.save_best:
        server.save_model(suffix=f"{text_mode}_round{rounds}")

    # 2. 记录最终结果到 JSON
    log_experiment_result(
        cfg.experiment_name,
        cfg.data.name,
        results[-1],
        config=cfg
    )
    log.info("✅ SBERT Workflow Completed Successfully.")


if __name__ == "__main__":
    main()
