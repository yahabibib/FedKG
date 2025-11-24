# 🚀 main.py
# 【最终版】集成 TensorBoard、双模融合推理(Config配置)、健壮性检查

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
    for folder in ["checkpoints", "logs", "runs"]:
        if not os.path.exists(folder):
            os.makedirs(folder)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"logs/exp_{config.CURRENT_DATASET_NAME}_{timestamp}.log"

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )

    writer = SummaryWriter(
        log_dir=f"runs/{config.CURRENT_DATASET_NAME}_{timestamp}")
    logging.info(f"📁 日志文件: {log_filename}")
    return writer


def generate_pseudo_pairs(emb1, emb2, threshold=0.75):
    """ 生成伪标签 (RNN 逻辑) """
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

    logging.info(f"     [Diagnosis] Total RNN pairs found: {rnn_count}")
    logging.info(
        f"     [Diagnosis] Pairs passing threshold ({threshold:.2f}): {len(pseudo_pairs)}")
    return pseudo_pairs


def run_pipeline():
    writer = setup_infrastructure()

    logging.info(f"{'='*60}")
    logging.info(f"🚀 启动 FedAnchor++ (Dynamic Proxy Experiment)")
    logging.info(f"📚 数据集: {config.CURRENT_DATASET_NAME}")
    logging.info(f"🧠 模型架构: {config.MODEL_INFO}")
    logging.info(f"⚖️  融合权重: Alpha = {config.EVAL_FUSION_ALPHA}")
    logging.info(
        f"🤖 动态代理: K={config.PROXY_NUM}, Temp={config.PROXY_TEMPERATURE}")
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
    logging.info("--- 阶段二：离线预计算 ---")
    te_1, te_2 = None, None
    if config.MODEL_ARCH == 'projection':
        num_rel = max(len(rel_1[0]), len(rel_2[0]))
        te_1 = precompute.train_transe(
            trip_1, ent_1, num_ent_1, num_rel, "KG1")
        te_2 = precompute.train_transe(
            trip_2, ent_2, num_ent_2, num_rel, "KG2")

    cache_path_1 = os.path.join("cache", "sbert_KG1.pt")
    cache_path_2 = os.path.join("cache", "sbert_KG2.pt")
    if not os.path.exists("cache"):
        os.makedirs("cache")

    sb_1 = precompute.get_bert_embeddings(
        ent_1, attr_1, "KG1", cache_file=cache_path_1)
    sb_2 = precompute.get_bert_embeddings(
        ent_2, attr_2, "KG2", cache_file=cache_path_2)

    # 🔥 [新增] K-Means 初始化代理
    # 合并两个图谱的 SBERT 向量做聚类，保证代理覆盖全局语义空间
    logging.info("--- 阶段 2.5: 初始化动态代理 ---")
    all_sbert = {**sb_1, **sb_2}
    current_proxies = precompute.initialize_proxies(
        all_sbert, config.PROXY_NUM)

    adj_1, adj_2 = None, None
    if config.MODEL_ARCH in ['gcn', 'decoupled']:
        logging.info(f"[Mode: {config.MODEL_ARCH}] 构建邻接矩阵...")
        # 注意：MPS 无法处理稀疏矩阵，所以这里 adj 保持在 CPU
        adj_1 = precompute.build_adjacency_matrix(trip_1, num_ent_1)
        adj_2 = precompute.build_adjacency_matrix(trip_2, num_ent_2)

    # --- 3. 联邦迭代训练 ---
    logging.info("--- 阶段三：联邦迭代自训练 (Dynamic Proxies) ---")
    ITERATIONS = 5
    pseudo_anchors_1 = {}
    pseudo_anchors_2 = {}

    global_step = 0

    for it in range(ITERATIONS):
        logging.info(f"\n{'#'*40}")
        logging.info(f"🔄 Iteration {it+1}/{ITERATIONS}")
        logging.info(f"{'#'*40}")

        # 初始化 Server (传入当前最新的代理)
        server = fl_core.Server(current_proxies)

        c1_args = {'bert': sb_1, 'num_ent': num_ent_1, 'adj': adj_1}
        c2_args = {'bert': sb_2, 'num_ent': num_ent_2, 'adj': adj_2}

        if config.MODEL_ARCH not in ['gcn', 'decoupled']:
            c1_args['transe'] = te_1
            c2_args['transe'] = te_2

        # Client 初始化需要传入 proxies
        c1 = fl_core.Client("C1", config.DEVICE,
                            proxies=current_proxies, **c1_args)
        c2 = fl_core.Client("C2", config.DEVICE,
                            proxies=current_proxies, **c2_args)

        # --- 加载 Checkpoint ---
        if it > 0:
            ckpt_c1 = f"checkpoints/c1_iter_{it}.pth"
            ckpt_c2 = f"checkpoints/c2_iter_{it}.pth"
            try:
                if os.path.exists(ckpt_c1) and os.path.exists(ckpt_c2):
                    state_c1 = torch.load(ckpt_c1, map_location=config.DEVICE)
                    c1.model.load_state_dict(state_c1, strict=False)

                    state_c2 = torch.load(ckpt_c2, map_location=config.DEVICE)
                    c2.model.load_state_dict(state_c2, strict=False)

                    logging.info(f"  ✅ Loaded checkpoints from Iter {it}")

                    if config.USE_AGGREGATION:
                        state = c1.model.state_dict()
                        filtered = {k: v for k, v in state.items()
                                    if "initial" not in k and "struct_encoder" not in k}
                        server.global_model.load_state_dict(
                            filtered, strict=False)

                    # 注意：代理状态通过 current_proxies 变量在内存中传递给下一轮 Server
                    # 这里不需要额外加载代理文件，除非程序崩溃重启

                    if pseudo_anchors_1:
                        c1.update_anchors(pseudo_anchors_1)
                    if pseudo_anchors_2:
                        c2.update_anchors(pseudo_anchors_2)
                else:
                    logging.warning(
                        "  ⚠️ Checkpoints not found. Training from scratch.")
            except Exception as e:
                logging.error(f"  ❌ Failed to load checkpoint: {e}")

        # --- 训练 ---
        global_w = server.get_global_model_state() if (
            it > 0 and config.USE_AGGREGATION) else None
        global_p = server.get_global_proxies()  # 获取当前 Server 端的代理

        current_rounds = config.FL_ROUNDS if it == 0 else max(
            20, int(config.FL_ROUNDS * 0.5))

        try:
            for r in range(current_rounds):
                # 传递 global_p (代理) 给客户端进行训练
                # local_train 返回三个值: 模型权重, 代理权重, Loss
                w1, p1, l1 = c1.local_train(
                    global_w, global_p, config.FL_LOCAL_EPOCHS, config.FL_LR)
                w2, p2, l2 = c2.local_train(
                    global_w, global_p, config.FL_LOCAL_EPOCHS, config.FL_LR)

                # 聚合 (同时聚合模型和代理)
                if config.USE_AGGREGATION:
                    global_w, global_p, p_diff = server.aggregate(
                        [w1, w2], [p1, p2])
                else:
                    p_diff = 0.0

                # 记录日志
                if ((r + 1) % 10 == 0) or (r == 0):
                    mode = "FedAvg" if config.USE_AGGREGATION else "Isolated"
                    logging.info(
                        f"  Round {r+1}/{current_rounds} [{mode}] | Loss: {l1:.4f} / {l2:.4f} | Proxy Shift: {p_diff:.6f}")

                # TensorBoard 记录
                writer.add_scalar(f'Loss/C1_Iter{it+1}', l1, r)
                writer.add_scalar(f'Loss/C2_Iter{it+1}', l2, r)
                writer.add_scalar(
                    f'Proxy/Shift_Iter{it+1}', p_diff, r)  # 监控代理移动幅度
                global_step += 1

            # 本轮 Iteration 结束，更新 current_proxies 为训练后的结果，供下一轮使用
            # 这样代理的进化就能延续到下一个阶段
            current_proxies = global_p.detach().cpu()

        except RuntimeError as e:
            if "out of memory" in str(e):
                logging.error("  [Error] GPU OOM!")
                break
            else:
                raise e

        # --- 保存 ---
        torch.save(c1.model.state_dict(), f"checkpoints/c1_iter_{it+1}.pth")
        torch.save(c2.model.state_dict(), f"checkpoints/c2_iter_{it+1}.pth")

        # --- 评估 (使用 Fusion) ---
        logging.info(f"\n  🔍 Iteration {it+1} Evaluation:")
        c1.model.eval()
        c2.model.eval()

        with torch.no_grad():
            if config.MODEL_ARCH in ['gcn', 'decoupled']:
                # 获取 GCN 特征
                emb_1 = c1.model(c1.adj).detach().cpu()
                emb_2 = c2.model(c2.adj).detach().cpu()

                e1_d = {i: emb_1[i] for i in range(len(emb_1))}
                e2_d = {i: emb_2[i] for i in range(len(emb_2))}

                # 【关键修改】传入 SBERT 和 Alpha 进行融合
                hits, mrr = evaluate.evaluate_alignment(
                    test_pairs, e1_d, e2_d,
                    torch.nn.Identity(), torch.nn.Identity(),
                    config.EVAL_K_VALUES,
                    sbert_1=sb_1, sbert_2=sb_2,   # 传入 SBERT
                    alpha=config.EVAL_FUSION_ALPHA  # 传入 Config 里的 0.42
                )
            else:
                # TransE 旧逻辑
                emb_1 = te_1
                emb_2 = te_2
                hits, mrr = evaluate.evaluate_alignment(
                    test_pairs, {i: emb_1[i] for i in range(len(emb_1))},
                    {i: emb_2[i] for i in range(len(emb_2))},
                    c1.model.cpu(), c1.model.cpu(),
                    config.EVAL_K_VALUES
                )

        if config.DEVICE.type == 'mps':
            torch.mps.empty_cache()
        elif config.DEVICE.type == 'cuda':
            torch.cuda.empty_cache()

        writer.add_scalar('Eval/MRR', mrr, it + 1)
        writer.add_scalar('Eval/Hits@1', hits.get(1, 0), it + 1)

        # --- 伪标签 ---
        if it < ITERATIONS - 1:
            thresh = max(0.50, 0.80 - (it * 0.05))
            logging.info(
                f"  🌱 Generating Pseudo-Labels (Threshold={thresh:.2f})...")

            new_pairs = generate_pseudo_pairs(emb_1, emb_2, threshold=thresh)
            for idx1, idx2 in new_pairs:
                pseudo_anchors_1[idx1] = emb_2[idx2]
                pseudo_anchors_2[idx2] = emb_1[idx1]
            logging.info(f"     Cumulative anchors: {len(pseudo_anchors_1)}")

        logging.info("  [System] Cleaning up memory...")
        del server, c1, c2, global_w, w1, w2, emb_1, emb_2
        # 注意：不要删 current_proxies
        gc.collect()

    logging.info("\n--- 实验结束 ---")
    writer.close()


if __name__ == "__main__":
    run_pipeline()
