# 🚀 main.py
# 【修复版】适配 Decoupled 架构参数传递逻辑

import torch
import torch.nn.functional as F
import os
import gc
import config
import data_loader
import precompute
import fl_core
import evaluate

if not os.path.exists("checkpoints"):
    os.makedirs("checkpoints")


def generate_pseudo_pairs(emb1, emb2, threshold=0.75):
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

    print(f"     [Diagnosis] Total RNN pairs found: {rnn_count}")
    print(
        f"     [Diagnosis] Pairs passing threshold ({threshold:.2f}): {len(pseudo_pairs)}")
    return pseudo_pairs


def run_pipeline():
    print(f"\n{'='*60}")
    print(f"🚀 启动联邦迭代自训练 (Fix: Decoupled Args)")
    print(f"📚 数据集: {config.CURRENT_DATASET_NAME}")
    print(f"🧠 模型架构: {config.MODEL_INFO}")
    print(
        f"🛠️  聚合模式: {'开启 (FedAvg)' if config.USE_AGGREGATION else '关闭 (Isolated)'}")
    print(f"{'='*60}\n")

    # --- 1. 数据加载 ---
    print("--- 阶段一：数据加载 ---")
    ent_1 = data_loader.load_id_map(config.BASE_PATH + "ent_ids_1")
    rel_1 = data_loader.load_id_map(config.BASE_PATH + "rel_ids_1")
    trip_1 = data_loader.load_triples(config.BASE_PATH + "triples_1")

    pkl_1 = config.BASE_PATH + "description1.pkl"
    if os.path.exists(pkl_1):
        attr_1 = data_loader.load_pickle_descriptions(pkl_1, ent_1)
    else:
        attr_1 = data_loader.load_attribute_triples(
            config.BASE_PATH + "zh_att_triples", ent_1)

    ent_2 = data_loader.load_id_map(config.BASE_PATH + "ent_ids_2")
    rel_2 = data_loader.load_id_map(config.BASE_PATH + "rel_ids_2")
    trip_2 = data_loader.load_triples(config.BASE_PATH + "triples_2")

    pkl_2 = config.BASE_PATH + "description2.pkl"
    if os.path.exists(pkl_2):
        attr_2 = data_loader.load_pickle_descriptions(pkl_2, ent_2)
    else:
        attr_2 = data_loader.load_attribute_triples(
            config.BASE_PATH + "en_att_triples", ent_2)

    test_pairs = data_loader.load_alignment_pairs(
        config.BASE_PATH + "ref_pairs")
    num_ent_1 = max(list(ent_1[0].keys())) + 1
    num_ent_2 = max(list(ent_2[0].keys())) + 1
    print(f"KG1: {num_ent_1}, KG2: {num_ent_2}")

    # --- 2. 离线预计算 ---
    print("\n--- 阶段二：离线预计算 ---")
    te_1, te_2 = None, None
    if config.MODEL_ARCH == 'projection':
        num_rel = max(len(rel_1[0]), len(rel_2[0]))
        te_1 = precompute.train_transe(
            trip_1, ent_1, num_ent_1, num_rel, "KG1")
        te_2 = precompute.train_transe(
            trip_2, ent_2, num_ent_2, num_rel, "KG2")

    cache_path_1 = os.path.join("cache", "sbert_KG1.pt")
    cache_path_2 = os.path.join("cache", "sbert_KG2.pt")
    # 确保 cache 目录存在
    if not os.path.exists("cache"):
        os.makedirs("cache")

    sb_1 = precompute.get_bert_embeddings(
        ent_1, attr_1, "KG1", cache_file=cache_path_1)
    sb_2 = precompute.get_bert_embeddings(
        ent_2, attr_2, "KG2", cache_file=cache_path_2)

    adj_1, adj_2 = None, None
    # 【修改】decoupled 模式也需要构建邻接矩阵
    if config.MODEL_ARCH in ['gcn', 'decoupled']:
        print(f"[Mode: {config.MODEL_ARCH}] 构建邻接矩阵...")
        adj_1 = precompute.build_adjacency_matrix(trip_1, num_ent_1)
        adj_2 = precompute.build_adjacency_matrix(trip_2, num_ent_2)

    print("\n--- 阶段 2.5: 锚点质量自检 ---")

    class Identity(torch.nn.Module):
        def forward(self, x): return x
    evaluate.evaluate_alignment(
        test_pairs, sb_1, sb_2, Identity(), Identity(), config.EVAL_K_VALUES)

    # --- 3. 联邦迭代训练 ---
    print("\n--- 阶段三：联邦迭代自训练 ---")
    ITERATIONS = 5
    pseudo_anchors_1 = {}
    pseudo_anchors_2 = {}

    for it in range(ITERATIONS):
        print(f"\n{'#'*40}")
        print(f"🔄 Iteration {it+1}/{ITERATIONS} (Self-Training)")
        print(f"{'#'*40}")

        print("  [System] Initializing clients...")
        server = fl_core.Server()

        c1_args = {'bert': sb_1, 'num_ent': num_ent_1}
        c2_args = {'bert': sb_2, 'num_ent': num_ent_2}

        # 【修改】正确分发参数，decoupled 走 gcn 的路
        if config.MODEL_ARCH in ['gcn', 'decoupled']:
            c1_args['adj'] = adj_1
            c2_args['adj'] = adj_2
        else:
            c1_args['transe'] = te_1
            c2_args['transe'] = te_2

        c1 = fl_core.Client("C1", config.DEVICE, **c1_args)
        c2 = fl_core.Client("C2", config.DEVICE, **c2_args)

        # --- 加载 Checkpoint ---
        if it > 0:
            ckpt_c1 = f"checkpoints/c1_iter_{it}.pth"
            ckpt_c2 = f"checkpoints/c2_iter_{it}.pth"
            print(f"  [System] Loading checkpoints from Iter {it}...")
            try:
                if os.path.exists(ckpt_c1) and os.path.exists(ckpt_c2):
                    # 允许部分加载 (strict=False)
                    c1.model.load_state_dict(torch.load(
                        ckpt_c1, map_location=config.DEVICE), strict=False)
                    c2.model.load_state_dict(torch.load(
                        ckpt_c2, map_location=config.DEVICE), strict=False)

                    if config.USE_AGGREGATION:
                        # 同步 Server
                        state = c1.model.state_dict()
                        # 过滤掉私有层 (initial_features 和 struct_encoder)
                        filtered = {k: v for k, v in state.items()
                                    if "initial" not in k and "struct_encoder" not in k}
                        server.global_model.load_state_dict(
                            filtered, strict=False)

                    if pseudo_anchors_1:
                        c1.update_anchors(pseudo_anchors_1)
                    if pseudo_anchors_2:
                        c2.update_anchors(pseudo_anchors_2)
                else:
                    print(
                        "  [Warning] Checkpoints not found. Training from scratch.")
            except Exception as e:
                print(f"  [Error] Failed to load checkpoint: {e}")

        # --- 训练 ---
        global_w = server.get_global_model_state() if (
            it > 0 and config.USE_AGGREGATION) else None
        current_rounds = config.FL_ROUNDS if it == 0 else max(
            20, int(config.FL_ROUNDS * 0.5))

        try:
            for r in range(current_rounds):
                log_loss = ((r + 1) % 10 == 0) or (r == 0)

                w1, l1 = c1.local_train(
                    global_w, config.FL_LOCAL_EPOCHS, config.FL_BATCH_SIZE, config.FL_LR)
                w2, l2 = c2.local_train(
                    global_w, config.FL_LOCAL_EPOCHS, config.FL_BATCH_SIZE, config.FL_LR)

                if config.USE_AGGREGATION:
                    global_w = server.aggregate_models([w1, w2])

                if log_loss:
                    mode = "FedAvg" if config.USE_AGGREGATION else "Isolated"
                    print(
                        f"  Round {r+1}/{current_rounds} [{mode}] | C1 Loss: {l1:.6f} | C2 Loss: {l2:.6f}")
        except RuntimeError as e:
            if "out of memory" in str(e):
                print("  [Error] GPU OOM! Try reducing Batch Size.")
                break
            else:
                raise e

        # --- 保存 ---
        print(f"  [System] Saving checkpoints Iter {it+1}...")
        torch.save(c1.model.state_dict(), f"checkpoints/c1_iter_{it+1}.pth")
        torch.save(c2.model.state_dict(), f"checkpoints/c2_iter_{it+1}.pth")

        # --- 评估 ---
        print(f"\n  🔍 Iteration {it+1} Evaluation:")
        c1.model.eval()
        c2.model.eval()

        with torch.no_grad():
            # 【修改】decoupled 模式也是通过 forward(adj) 推理
            if config.MODEL_ARCH in ['gcn', 'decoupled']:
                emb_1 = c1.model(c1.adj).detach().cpu()
                emb_2 = c2.model(c2.adj).detach().cpu()
                m_eval = torch.nn.Identity()
            else:
                emb_1 = te_1
                emb_2 = te_2
                m_eval = c1.model.cpu()

        if config.DEVICE.type == 'mps':
            torch.mps.empty_cache()
        elif config.DEVICE.type == 'cuda':
            torch.cuda.empty_cache()

        e1_d = {i: emb_1[i] for i in range(len(emb_1))}
        e2_d = {i: emb_2[i] for i in range(len(emb_2))}
        evaluate.evaluate_alignment(
            test_pairs, e1_d, e2_d, m_eval, m_eval, config.EVAL_K_VALUES)

        # --- 伪标签 ---
        if it < ITERATIONS - 1:
            thresh = max(0.50, 0.80 - (it * 0.05))
            print(
                f"\n  🌱 Generating Pseudo-Labels (Threshold={thresh:.2f})...")

            new_pairs = generate_pseudo_pairs(emb_1, emb_2, threshold=thresh)
            for idx1, idx2 in new_pairs:
                pseudo_anchors_1[idx1] = emb_2[idx2]
                pseudo_anchors_2[idx2] = emb_1[idx1]
            print(f"     [Cache] Cumulative anchors: {len(pseudo_anchors_1)}")

        print("  [System] Cleaning up memory...")
        del server, c1, c2, global_w, w1, w2, emb_1, emb_2
        gc.collect()

    print("\n--- 实验结束 ---")


if __name__ == "__main__":
    run_pipeline()
