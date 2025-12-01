# 🚀 main.py (Final Optimized)
# 1. SBERT 缓存修复
# 2. GCN 训练轮次增加 (首轮 100 epoch)
# 3. 最终模型保存

import torch
import torch.nn.functional as F
import os
import config
import data_loader
import precompute
import fl_core
import evaluate
import logging
import datetime
import warnings
from torch.utils.tensorboard import SummaryWriter

warnings.filterwarnings("ignore")

try:
    import result_logger
    HAS_LOGGER = True
except ImportError:
    HAS_LOGGER = False


def setup_infrastructure():
    for folder in ["checkpoints", "logs", "runs", "cache"]:
        if not os.path.exists(folder):
            os.makedirs(folder)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"logs/exp_{config.CURRENT_DATASET_NAME}_{timestamp}.log"
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[logging.FileHandler(log_filename, encoding='utf-8'), logging.StreamHandler()])
    writer = SummaryWriter(
        log_dir=f"runs/{config.CURRENT_DATASET_NAME}_{timestamp}")
    return writer


def generate_pseudo_pairs(emb1, emb2, valid_ids_1, valid_ids_2, threshold=0.75):
    emb1 = F.normalize(emb1, dim=1)
    emb2 = F.normalize(emb2, dim=1)
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

    logging.info(
        f"     [Mining] Clean pairs found: {len(pseudo_pairs)} (Threshold={threshold})")
    return pseudo_pairs


def run_pipeline():
    writer = setup_infrastructure()
    logging.info(f"{'='*60}")
    logging.info(f"🚀 启动 FedKG (Production Optimized)")
    logging.info(f"   SBERT: {config.BERT_MODEL_NAME}")
    logging.info(f"{'='*60}")

    # 1. 加载数据
    logging.info("📚 Loading Datasets...")
    ent_1 = data_loader.load_id_map(config.BASE_PATH + "ent_ids_1")
    trip_1 = data_loader.load_triples(config.BASE_PATH + "triples_1")
    rel_1, _ = data_loader.load_id_map(config.BASE_PATH + "rel_ids_1")

    ent_2 = data_loader.load_id_map(config.BASE_PATH + "ent_ids_2")
    trip_2 = data_loader.load_triples(config.BASE_PATH + "triples_2")
    rel_2, _ = data_loader.load_id_map(config.BASE_PATH + "rel_ids_2")

    pkl_1 = config.BASE_PATH + "description1.pkl"
    attr_1 = data_loader.load_pickle_descriptions(
        pkl_1, ent_1) if os.path.exists(pkl_1) else {}
    pkl_2 = config.BASE_PATH + "description2.pkl"
    attr_2 = data_loader.load_pickle_descriptions(
        pkl_2, ent_2) if os.path.exists(pkl_2) else {}

    test_pairs = data_loader.load_alignment_pairs(
        config.BASE_PATH + "ref_pairs")
    num_ent_1 = max(list(ent_1[0].keys())) + 1
    num_ent_2 = max(list(ent_2[0].keys())) + 1
    num_rel_1 = max([t[1] for t in trip_1]) + 1
    num_rel_2 = max([t[1] for t in trip_2]) + 1

    valid_ids_1 = list(ent_1[0].keys())
    valid_ids_2 = list(ent_2[0].keys())

    # 2. 预计算 (Embeddings)
    # 使用微调后的模型路径
    current_bert_path = config.BERT_MODEL_NAME
    logging.info(f"\n🔄 Computing Embeddings using: {current_bert_path}")

    # 缓存文件 (确保文件名唯一，避免读取旧的)
    cache_1 = "cache/sbert_KG1_exp4_final.pt"
    cache_2 = "cache/sbert_KG2_exp4_final.pt"

    # 计算实体 Embedding
    # 这里不需要再拼结构了，因为模型已经微调过了，直接用描述即可
    sb_1 = precompute.get_bert_embeddings(
        ent_1, attr_1, "KG1", cache_file=cache_1, custom_model_path=current_bert_path)
    sb_2 = precompute.get_bert_embeddings(
        ent_2, attr_2, "KG2", cache_file=cache_2, custom_model_path=current_bert_path)

    # 关系初始化 (可选)
    rel_emb_1 = precompute.get_relation_embeddings(
        rel_1, "KG1", cache_file="cache/rel_KG1_base.pt")
    rel_emb_2 = precompute.get_relation_embeddings(
        rel_2, "KG2", cache_file="cache/rel_KG2_base.pt")

    adj_1 = precompute.build_adjacency_matrix(trip_1, num_ent_1)
    adj_2 = precompute.build_adjacency_matrix(trip_2, num_ent_2)
    if config.DEVICE.type == 'cuda':
        adj_1 = adj_1.to(config.DEVICE)
        adj_2 = adj_2.to(config.DEVICE)

    # 评估 Baseline
    logging.info("\n[Init] Evaluating SBERT Baseline...")
    hits, _ = evaluate.evaluate_alignment(test_pairs, sb_1, sb_2, torch.nn.Identity(
    ), torch.nn.Identity(), [1], sbert_1=sb_1, sbert_2=sb_2, alpha=0.0)
    logging.info(f"   🏆 SBERT Baseline Hits@1: {hits[1]:.2f}%")

    # 3. 联邦训练
    logging.info("\n🔥 Starting GCN Training...")
    server = fl_core.Server()
    c1 = fl_core.Client("C1", config.DEVICE, bert=sb_1, num_ent=num_ent_1, adj=adj_1,
                        triples=trip_1, num_relations=num_rel_1, rel_init_emb=rel_emb_1)
    c2 = fl_core.Client("C2", config.DEVICE, bert=sb_2, num_ent=num_ent_2, adj=adj_2,
                        triples=trip_2, num_relations=num_rel_2, rel_init_emb=rel_emb_2)

    ITERATIONS = 5
    pseudo_anchors_1 = {}
    pseudo_anchors_2 = {}
    final_hits = {}

    for it in range(ITERATIONS):
        logging.info(f"\n🔄 Iteration {it+1}/{ITERATIONS}")

        if it > 0:
            # 加载上一轮
            c1.model.load_state_dict(torch.load(
                f"checkpoints/c1_iter_{it}.pth", map_location=config.DEVICE), strict=False)
            c2.model.load_state_dict(torch.load(
                f"checkpoints/c2_iter_{it}.pth", map_location=config.DEVICE), strict=False)

            if config.USE_AGGREGATION:
                state = c1.model.state_dict()
                filtered = {k: v for k, v in state.items(
                ) if "initial" not in k and "struct_encoder" not in k and "rel_embedding" not in k}
                server.global_model.load_state_dict(filtered, strict=False)

            if pseudo_anchors_1:
                c1.update_anchors(pseudo_anchors_1)
            if pseudo_anchors_2:
                c2.update_anchors(pseudo_anchors_2)

        global_w = server.get_global_model_state() if (
            it > 0 and config.USE_AGGREGATION) else None

        # 【核心优化】第一轮跑 100 次，让 Loss 充分下降
        rounds = 100 if it == 0 else 30

        for r in range(rounds):
            w1, l1 = c1.local_train(
                global_w, 3, config.FL_BATCH_SIZE, config.FL_LR)
            w2, l2 = c2.local_train(
                global_w, 3, config.FL_BATCH_SIZE, config.FL_LR)
            if config.USE_AGGREGATION:
                global_w = server.aggregate_models([w1, w2])

            if (r + 1) % 10 == 0:
                logging.info(
                    f"   Round {r+1}/{rounds} | Loss: {l1:.4f} / {l2:.4f}")
                writer.add_scalar(f'Loss/Iter{it+1}', l1, r)

        torch.save(c1.model.state_dict(), f"checkpoints/c1_iter_{it+1}.pth")
        torch.save(c2.model.state_dict(), f"checkpoints/c2_iter_{it+1}.pth")

        c1.model.eval()
        c2.model.eval()
        with torch.no_grad():
            emb_1 = c1.model(adj_1).detach().cpu()
            emb_2 = c2.model(adj_2).detach().cpu()
            e1_d = {i: emb_1[i] for i in range(len(emb_1))}
            e2_d = {i: emb_2[i] for i in range(len(emb_2))}
            hits, _ = evaluate.evaluate_alignment(test_pairs, e1_d, e2_d, torch.nn.Identity(), torch.nn.Identity(
            ), config.EVAL_K_VALUES, sbert_1=sb_1, sbert_2=sb_2, alpha=config.EVAL_FUSION_ALPHA)
            final_hits = hits
            writer.add_scalar('Eval/Hits@1', hits[1], it + 1)

        if it < ITERATIONS - 1:
            thresh = 0.75 + (it * 0.05)
            new_pairs = generate_pseudo_pairs(
                emb_1, emb_2, valid_ids_1, valid_ids_2, threshold=thresh)
            for i, j in new_pairs:
                pseudo_anchors_1[i] = emb_2[j]
                pseudo_anchors_2[j] = emb_1[i]
            logging.info(f"   ⚓️ Anchors Updated: {len(new_pairs)}")

    # 【新增】保存最终模型，方便后续分析
    print("💾 Saving Final Models...")
    torch.save(c1.model.state_dict(), "checkpoints/c1_final.pth")
    torch.save(c2.model.state_dict(), "checkpoints/c2_final.pth")

    logging.info(f"\n✨ Final Result Hits@1: {final_hits[1]:.2f}%")
    writer.close()
    if HAS_LOGGER:
        result_logger.log_experiment_result("FedKG (Final)", config.CURRENT_DATASET_NAME, {
                                            "hits1": final_hits.get(1, 0)}, {"strategy": "Pre-Finetuned-SBERT"})


if __name__ == "__main__":
    run_pipeline()
