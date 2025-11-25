# 🚀 main.py
# 【Relation-Aware 版】集成关系感知 GCN，移除动态代理

import torch
import torch.nn.functional as F
import os
import gc
import config
import data_loader
import precompute
import fl_core
import evaluate
import logging
import datetime
from torch.utils.tensorboard import SummaryWriter

# --- 0. 基础设施设置 ---


def setup_infrastructure():
    for folder in ["checkpoints", "logs", "runs", "cache"]:
        if not os.path.exists(folder):
            os.makedirs(folder)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"logs/exp_{config.CURRENT_DATASET_NAME}_{timestamp}.log"

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(
            log_filename, encoding='utf-8'), logging.StreamHandler()]
    )
    writer = SummaryWriter(
        log_dir=f"runs/{config.CURRENT_DATASET_NAME}_{timestamp}")
    return writer


def generate_pseudo_pairs(emb1, emb2, threshold=0.75):
    """ 生成伪标签 """
    emb1 = F.normalize(emb1, dim=1)
    emb2 = F.normalize(emb2, dim=1)
    sim_mat = torch.mm(emb1, emb2.T)
    values_1, indices_1 = torch.max(sim_mat, dim=1)
    values_2, indices_2 = torch.max(sim_mat, dim=0)

    pseudo_pairs = []
    rnn_count = 0
    for i in range(len(emb1)):
        j = indices_1[i].item()
        if indices_2[j].item() == i:
            rnn_count += 1
            if values_1[i].item() > threshold:
                pseudo_pairs.append((i, j))
    logging.info(
        f"     [Diagnosis] RNN pairs: {rnn_count}, Valid Pseudo: {len(pseudo_pairs)}")
    return pseudo_pairs


def run_pipeline():
    writer = setup_infrastructure()

    logging.info(f"{'='*60}")
    logging.info(f"🚀 启动 FedAnchor++ (Relation-Aware GCN)")
    logging.info(f"📚 数据集: {config.CURRENT_DATASET_NAME}")
    logging.info(f"🧠 模型架构: {config.MODEL_INFO}")
    logging.info(f"⚖️  融合权重: Alpha = {config.EVAL_FUSION_ALPHA}")
    logging.info(f"{'='*60}")

    # --- 1. 数据加载 ---
    logging.info("--- 阶段一：数据加载 ---")
    ent_1 = data_loader.load_id_map(config.BASE_PATH + "ent_ids_1")
    rel_1 = data_loader.load_id_map(config.BASE_PATH + "rel_ids_1")
    trip_1 = data_loader.load_triples(config.BASE_PATH + "triples_1")

    pkl_1 = config.BASE_PATH + "description1.pkl"
    attr_1 = data_loader.load_pickle_descriptions(pkl_1, ent_1) if os.path.exists(pkl_1) else \
        data_loader.load_attribute_triples(
            config.BASE_PATH + "zh_att_triples", ent_1)

    ent_2 = data_loader.load_id_map(config.BASE_PATH + "ent_ids_2")
    rel_2 = data_loader.load_id_map(config.BASE_PATH + "rel_ids_2")
    trip_2 = data_loader.load_triples(config.BASE_PATH + "triples_2")

    pkl_2 = config.BASE_PATH + "description2.pkl"
    attr_2 = data_loader.load_pickle_descriptions(pkl_2, ent_2) if os.path.exists(pkl_2) else \
        data_loader.load_attribute_triples(
            config.BASE_PATH + "en_att_triples", ent_2)

    test_pairs = data_loader.load_alignment_pairs(
        config.BASE_PATH + "ref_pairs")
    num_ent_1 = max(list(ent_1[0].keys())) + 1
    num_ent_2 = max(list(ent_2[0].keys())) + 1

    # --- 2. 离线预计算 ---
    logging.info("--- 阶段二：离线预计算 (SBERT & Graph) ---")
    te_1, te_2 = None, None
    if config.MODEL_ARCH == 'projection':
        # ... (TransE 逻辑略) ...
        pass

    # 实体 SBERT
    sb_1 = precompute.get_bert_embeddings(
        ent_1, attr_1, "KG1", cache_file="cache/sbert_KG1.pt")
    sb_2 = precompute.get_bert_embeddings(
        ent_2, attr_2, "KG2", cache_file="cache/sbert_KG2.pt")

    # 🔥 [新增] 关系 SBERT
    rel_sb_1 = precompute.get_relation_embeddings(
        rel_1, "KG1_Rel", cache_file="cache/rel_sbert_KG1.pt")
    rel_sb_2 = precompute.get_relation_embeddings(
        rel_2, "KG2_Rel", cache_file="cache/rel_sbert_KG2.pt")

    # 🔥 [修改] 构建 Relation Graph
    edge_index_1, edge_type_1 = None, None
    edge_index_2, edge_type_2 = None, None

    if config.MODEL_ARCH in ['gcn', 'decoupled']:
        logging.info(f"[Mode: {config.MODEL_ARCH}] 构建带关系的图结构...")
        # 传入原始关系数量，函数内部会处理反向+自环
        edge_index_1, edge_type_1 = precompute.build_graph_data(
            trip_1, num_ent_1, len(rel_1[0]))
        edge_index_2, edge_type_2 = precompute.build_graph_data(
            trip_2, num_ent_2, len(rel_2[0]))

    # --- 3. 联邦迭代训练 ---
    logging.info("--- 阶段三：联邦迭代自训练 ---")
    ITERATIONS = 5
    pseudo_anchors_1 = {}
    pseudo_anchors_2 = {}

    for it in range(ITERATIONS):
        logging.info(f"\n{'#'*40}")
        logging.info(f"🔄 Iteration {it+1}/{ITERATIONS}")
        logging.info(f"{'#'*40}")

        server = fl_core.Server()

        # 🔥 [修改] Client 初始化参数
        # 关键：注释掉 rel_sbert，让模型自己学习关系门控参数
        c1_args = {
            'bert': sb_1, 'num_ent': num_ent_1,
            'num_rel': len(rel_1[0]),
            # 'rel_sbert': rel_sb_1,        <--- 保持注释
            'edge_index': edge_index_1, 'edge_type': edge_type_1
        }

        c2_args = {
            'bert': sb_2, 'num_ent': num_ent_2,
            'num_rel': len(rel_2[0]),
            # 'rel_sbert': rel_sb_2,        <--- 保持注释
            'edge_index': edge_index_2, 'edge_type': edge_type_2
        }

        # 如果不是 GCN，可能需要 TransE 参数
        if config.MODEL_ARCH not in ['gcn', 'decoupled']:
            c1_args['transe'] = te_1
            c2_args['transe'] = te_2

        c1 = fl_core.Client("C1", config.DEVICE, **c1_args)
        c2 = fl_core.Client("C2", config.DEVICE, **c2_args)

        # --- Checkpoint 加载 ---
        if it > 0:
            ckpt_c1 = f"checkpoints/c1_iter_{it}.pth"
            ckpt_c2 = f"checkpoints/c2_iter_{it}.pth"
            try:
                if os.path.exists(ckpt_c1) and os.path.exists(ckpt_c2):
                    c1.model.load_state_dict(torch.load(
                        ckpt_c1, map_location=config.DEVICE), strict=False)
                    c2.model.load_state_dict(torch.load(
                        ckpt_c2, map_location=config.DEVICE), strict=False)
                    logging.info(f"  ✅ Loaded checkpoints from Iter {it}")

                    if config.USE_AGGREGATION:
                        state = c1.model.state_dict()
                        filtered = {k: v for k, v in state.items()
                                    if "initial" not in k and "struct_encoder" not in k and "relation" not in k}
                        server.global_model.load_state_dict(
                            filtered, strict=False)

                    if pseudo_anchors_1:
                        c1.update_anchors(pseudo_anchors_1)
                    if pseudo_anchors_2:
                        c2.update_anchors(pseudo_anchors_2)
            except Exception as e:
                logging.error(f"  ❌ Failed to load checkpoint: {e}")

        # --- 训练 ---
        global_w = server.get_global_model_state() if (
            it > 0 and config.USE_AGGREGATION) else None
        current_rounds = config.FL_ROUNDS if it == 0 else max(
            20, int(config.FL_ROUNDS * 0.5))

        try:
            for r in range(current_rounds):
                w1, l1 = c1.local_train(
                    global_w, config.FL_LOCAL_EPOCHS, config.FL_BATCH_SIZE, config.FL_LR)
                w2, l2 = c2.local_train(
                    global_w, config.FL_LOCAL_EPOCHS, config.FL_BATCH_SIZE, config.FL_LR)

                if config.USE_AGGREGATION:
                    global_w = server.aggregate_models([w1, w2])

                if ((r + 1) % 10 == 0) or (r == 0):
                    logging.info(
                        f"  Round {r+1}/{current_rounds} | Loss: {l1:.4f} / {l2:.4f}")

                writer.add_scalar(f'Loss/C1_Iter{it+1}', l1, r)
                writer.add_scalar(f'Loss/C2_Iter{it+1}', l2, r)

        except RuntimeError as e:
            if "out of memory" in str(e):
                logging.error("  [Error] GPU OOM!")
                break
            else:
                raise e

        # --- 保存 & 评估 ---
        torch.save(c1.model.state_dict(), f"checkpoints/c1_iter_{it+1}.pth")
        torch.save(c2.model.state_dict(), f"checkpoints/c2_iter_{it+1}.pth")

        logging.info(f"\n  🔍 Iteration {it+1} Evaluation:")
        c1.model.eval()
        c2.model.eval()

        with torch.no_grad():
            if config.MODEL_ARCH in ['gcn', 'decoupled']:
                # 🔥 [关键] 推理时也要传入 Edge Index/Type
                emb_1 = c1.model(c1.edge_index, c1.edge_type).detach().cpu()
                emb_2 = c2.model(c2.edge_index, c2.edge_type).detach().cpu()

                e1_d = {i: emb_1[i] for i in range(len(emb_1))}
                e2_d = {i: emb_2[i] for i in range(len(emb_2))}

                hits, mrr = evaluate.evaluate_alignment(
                    test_pairs, e1_d, e2_d,
                    torch.nn.Identity(), torch.nn.Identity(),
                    config.EVAL_K_VALUES,
                    sbert_1=sb_1, sbert_2=sb_2,
                    alpha=config.EVAL_FUSION_ALPHA
                )
            else:
                # TransE 逻辑
                pass

        writer.add_scalar('Eval/MRR', mrr, it + 1)
        writer.add_scalar('Eval/Hits@1', hits.get(1, 0), it + 1)

        # --- 伪标签 ---
        if it < ITERATIONS - 1:
            # 🔥 [关键优化] 大幅降低门槛
            # 第一轮 0.70，后面每轮降 0.05，最低 0.45
            # 原来是 max(0.50, 0.80 - ...) 太高了
            thresh = max(0.45, 0.70 - (it * 0.05))
            logging.info(
                f"  🌱 Generating Pseudo-Labels (Threshold={thresh:.2f})...")
            new_pairs = generate_pseudo_pairs(emb_1, emb_2, threshold=thresh)
            for idx1, idx2 in new_pairs:
                pseudo_anchors_1[idx1] = emb_2[idx2]
                pseudo_anchors_2[idx2] = emb_1[idx1]
            logging.info(f"     Cumulative anchors: {len(pseudo_anchors_1)}")

        del server, c1, c2, global_w, w1, w2, emb_1, emb_2
        gc.collect()

    logging.info("\n--- 实验结束 ---")
    writer.close()


if __name__ == "__main__":
    run_pipeline()
