# 📄 test_sbert.py
# 实验设置 A: Isolation (SBERT Baseline)
# 验证仅使用预训练语言模型语义（不训练 GCN）的效果

import torch
import torch.nn as nn
import config
import data_loader
import precompute
import evaluate
import os
import utils_logger


def run_sbert_baseline():
    print(f"{'='*60}")
    print("🧪 实验 A: Isolation (SBERT-Only Baseline)")
    print(f"   目标: 评估未经训练的 SBERT 语义对齐能力 (Zero-shot)")
    print(f"{'='*60}")

    # 1. 加载数据
    ent_1 = data_loader.load_id_map(config.BASE_PATH + "ent_ids_1")
    ent_2 = data_loader.load_id_map(config.BASE_PATH + "ent_ids_2")
    test_pairs = data_loader.load_alignment_pairs(
        config.BASE_PATH + "ref_pairs")

    # 2. 加载或计算 SBERT Embedding
    # 确保 cache 目录存在
    if not os.path.exists("cache"):
        os.makedirs("cache")

    cache_1 = "cache/sbert_KG1.pt"
    cache_2 = "cache/sbert_KG2.pt"

    # 注意：这里会复用 precompute 的逻辑，如果有缓存直接读，没有就算
    sb_1 = precompute.get_bert_embeddings(ent_1, {}, "KG1", cache_file=cache_1)
    sb_2 = precompute.get_bert_embeddings(ent_2, {}, "KG2", cache_file=cache_2)

    # 3. 评估
    # 技巧：我们将 GCN 模型设为 Identity (不做任何处理)，并将 Alpha 设为 0.0 (全语义)
    # 这样 evaluate_alignment 内部就会只计算 SBERT 的相似度
    dummy_model = nn.Identity()

    # 构造伪造的结构 Embedding (全0)，因为 Alpha=0 时它们不会被使用
    # 但为了通过函数接口检查，我们需要传进去
    dummy_emb_1 = {i: torch.zeros(1) for i in sb_1.keys()}
    dummy_emb_2 = {i: torch.zeros(1) for i in sb_2.keys()}

    print("\n[开始评估]...")
    hits, mrr = evaluate.evaluate_alignment(
        test_pairs,
        dummy_emb_1, dummy_emb_2,
        dummy_model, dummy_model,
        config.EVAL_K_VALUES,
        sbert_1=sb_1,
        sbert_2=sb_2,
        alpha=0.0  # <--- 关键：0.0 代表 100% SBERT, 0% GCN
    )

    # ---> 新增记录代码
    utils_logger.log_experiment_result(
        exp_name="Isolation (SBERT)",
        dataset=config.CURRENT_DATASET_NAME,
        metrics={"hits1": hits[1], "hits10": hits[10], "mrr": mrr},
        params={"mode": "zero-shot"}
    )


if __name__ == "__main__":
    run_sbert_baseline()
