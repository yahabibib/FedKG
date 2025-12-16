# run_isolation.py
import hydra
from omegaconf import DictConfig
import logging
import torch
import os
import json
from src.data.dataset import AlignmentTaskData
from src.utils.device_manager import DeviceManager
from src.utils.metrics import eval_alignment
from src.federation.client_structure import ClientStructure
from src.utils.graph import build_adjacency_matrix
from src.utils.logger import log_experiment_result

log = logging.getLogger(__name__)

@hydra.main(config_path="configs", config_name="config", version_base="1.2")
def main(cfg: DictConfig):
    """
    [Baseline] Isolation / Local Training
    Client 之间互不通信，仅利用本地结构和 SBERT 锚点进行单机训练。
    """
    log.info(f"🏝️ Starting ISOLATION (Local) Experiment: {cfg.task.model.encoder_name}")
    
    dm = DeviceManager(cfg.system)
    task_data = AlignmentTaskData(cfg.data)
    
    # 初始化 Clients
    c1 = ClientStructure("C1", cfg, task_data.source, dm)
    c2 = ClientStructure("C2", cfg, task_data.target, dm)
    
    # 强制设置参数：单机训练不需要多轮 Round，只需要一次充分的 Local Epochs
    # 我们用 100 个 Epoch 来模拟充分收敛
    epochs = 100 
    
    log.info(f"   🚀 Training C1 Locally for {epochs} epochs...")
    c1.train(custom_epochs=epochs)
    
    log.info(f"   🚀 Training C2 Locally for {epochs} epochs...")
    c2.train(custom_epochs=epochs)
    
    # 评估
    log.info("   📊 Evaluating Local Models...")
    
    # 获取 Embeddings
    emb1 = c1.get_embeddings() # 包含结构特征
    emb2 = c2.get_embeddings()
    
    # 转换为字典
    d1 = {id: emb1[i] for i, id in enumerate(c1.dataset.ids)}
    d2 = {id: emb2[i] for i, id in enumerate(c2.dataset.ids)}
    
    # 评估 (使用 Score Fusion)
    # 也要加载 SBERT 以保持评估公平性
    sbert1 = c1.anchor_embeddings.cpu()
    sbert2 = c2.anchor_embeddings.cpu()
    sd1 = {id: sbert1[i] for i, id in enumerate(c1.dataset.ids)}
    sd2 = {id: sbert2[i] for i, id in enumerate(c2.dataset.ids)}
    
    hits, mrr = eval_alignment(
        d1, d2, task_data.test_pairs, 
        k_values=[1, 5, 10],
        sbert1_dict=sd1, sbert2_dict=sd2,
        alpha=cfg.task.eval.alpha,
        device='cpu'
    )
    
    log.info(f"   🏆 Isolation Result: Hits@1={hits[1]:.2f}% | Hits@10={hits[10]:.2f}% | MRR={mrr:.4f}")
    
    # 保存结果
    res = {
        "setting": "isolation",
        "encoder": cfg.task.model.encoder_name,
        "hits1": hits[1],
        "hits10": hits[10],
        "mrr": mrr
    }
    log_experiment_result("isolation_baseline", cfg.data.name, res)

if __name__ == "__main__":
    os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
    main()