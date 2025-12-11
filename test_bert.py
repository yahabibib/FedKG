import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sentence_transformers import SentenceTransformer
from collections import OrderedDict
import os
import pickle
import copy
import config
import data_loader
import evaluate
import utils_logger
from tqdm import tqdm
import gc
import random

# ==========================================
# 🎛️ 实验控制台
# ==========================================
# 推荐使用 'mixed' 以获得最佳效果
# 'description' -> 纯描述
# 'polished'    -> 纯润色
# 'mixed'       -> 混合训练 (Desc + Polish 同时训练，效果最佳)
TEXT_MODE = 'mixed'

# 联邦设置
ROUNDS = 5
LOCAL_EPOCHS = 1
BATCH_SIZE = 4       # 显存安全值
LR = 2e-5
PSEUDO_THRESHOLD = 0.85


def clean_memory():
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

# ==========================================
# 1. 客户端 (支持混合训练 & 动态显存管理)
# ==========================================


class ClientSBERT:
    def __init__(self, client_id, device, text_data_dict):
        self.client_id = client_id
        self.device = device
        self.text_data = text_data_dict

        print(f"    [{client_id}] Initializing model on CPU...")
        self.sbert = SentenceTransformer(config.BERT_MODEL_NAME, device='cpu')

        self.train_pairs = []

    def update_pseudo_labels(self, local_indices, remote_emb_desc, remote_emb_polish):
        """
        接收伪标签，混合两种数据构建训练集
        """
        self.train_pairs = []

        # 确保数据在 CPU
        if remote_emb_desc.device.type != 'cpu':
            remote_emb_desc = remote_emb_desc.cpu()
        if remote_emb_polish is not None and remote_emb_polish.device.type != 'cpu':
            remote_emb_polish = remote_emb_polish.cpu()

        for i, local_id in enumerate(local_indices):
            # 1. 加入 Description 样本 (强语义)
            if 'desc' in self.text_data and local_id in self.text_data['desc']:
                target = remote_emb_desc[i].detach().clone()
                self.train_pairs.append(
                    (self.text_data['desc'][local_id], target))

            # 2. 加入 Polished 样本 (强结构)
            if 'polish' in self.text_data and local_id in self.text_data['polish']:
                # 如果对方没有 polish 向量，就用 desc 向量顶替
                target = remote_emb_polish[i].detach().clone(
                ) if remote_emb_polish is not None else remote_emb_desc[i].detach().clone()
                self.train_pairs.append(
                    (self.text_data['polish'][local_id], target))

        # 关键：打乱数据，防止模型偏科
        random.shuffle(self.train_pairs)

    def train_mixed(self):
        """混合训练循环"""
        if not self.train_pairs:
            return self.sbert.state_dict(), 0.0

        # 模型上 GPU
        self.sbert.to(self.device)
        self.sbert.train()

        transformer = self.sbert._first_module().auto_model
        optimizer = optim.AdamW(transformer.parameters(), lr=LR)
        criterion = nn.MSELoss()

        pbar = tqdm(range(0, len(self.train_pairs), BATCH_SIZE),
                    desc=f"[{self.client_id}] Mixed Train", leave=False)

        total_loss = 0.0
        for i in pbar:
            batch = self.train_pairs[i: i+BATCH_SIZE]
            if not batch:
                continue

            texts = [b[0] for b in batch]
            targets = torch.stack([b[1] for b in batch]).to(self.device)

            features = self.sbert.tokenize(texts)
            for k in features:
                features[k] = features[k].to(self.device)

            out = transformer(**features)
            token_emb = out.last_hidden_state
            mask = features['attention_mask']
            input_mask_expanded = mask.unsqueeze(
                -1).expand(token_emb.size()).float()
            embeddings = torch.sum(token_emb * input_mask_expanded, 1) / \
                torch.clamp(input_mask_expanded.sum(1), min=1e-9)

            loss = criterion(embeddings, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            current_loss = loss.item()
            total_loss += current_loss
            pbar.set_postfix({'loss': f"{current_loss:.6f}"})

            del targets, features, out, embeddings, loss

        # 清理
        del optimizer
        self.sbert.to('cpu')
        clean_memory()
        return self.sbert.state_dict(), total_loss / max(1, len(pbar))

    def encode_all(self, key='desc'):
        """编码指定类型的所有文本"""
        self.sbert.to(self.device)
        self.sbert.eval()

        data_map = self.text_data.get(key, {})
        # Fallback
        if not data_map and key == 'polish':
            data_map = self.text_data.get('desc', {})

        sorted_ids = sorted(list(data_map.keys()))
        texts = [data_map[i] for i in sorted_ids]

        with torch.no_grad():
            embs = self.sbert.encode(
                texts, convert_to_tensor=True, show_progress_bar=True, batch_size=32, device=self.device)
            embs = embs.cpu()

        self.sbert.to('cpu')
        clean_memory()
        return sorted_ids, embs

# ==========================================
# 2. Server
# ==========================================


class ServerSBERT:
    def __init__(self):
        print("    [Server] Initializing on CPU...")
        self.global_model = SentenceTransformer(
            config.BERT_MODEL_NAME, device='cpu')

    def aggregate(self, states):
        if not states:
            return None
        print("   [Server] Aggregating parameters...")
        avg_weights = OrderedDict()
        keys = states[0].keys()
        for key in keys:
            tensors = [s[key].to('cpu') for s in states]
            avg_weights[key] = torch.stack(tensors).mean(dim=0)
        self.global_model.load_state_dict(avg_weights)
        return avg_weights

# ==========================================
# 3. 数据加载 (支持 mixed 模式)
# ==========================================


def load_data_dict(mode):
    print(f"\n📚 Loading Data Dict for Mode: [{mode}]")
    map_tuple_1 = data_loader.load_id_map(config.BASE_PATH + "ent_ids_1")
    map_tuple_2 = data_loader.load_id_map(config.BASE_PATH + "ent_ids_2")

    def _safe_load(suffix, ent_map_tuple):
        path = config.BASE_PATH + suffix
        if not os.path.exists(path):
            return {}
        return data_loader.load_pickle_descriptions(path, ent_map_tuple)

    # 预加载
    desc1 = _safe_load("description1.pkl", map_tuple_1)
    desc2 = _safe_load("description2.pkl", map_tuple_2)
    polish1 = _safe_load("desc_polish_1.pkl", map_tuple_1)
    polish2 = _safe_load("desc_polish_2.pkl", map_tuple_2)

    # Mixed 模式需要同时持有两份数据
    if mode == 'mixed' or mode == 'dual_stage':
        c1_data = {'desc': desc1, 'polish': polish1}
        c2_data = {'desc': desc2, 'polish': polish2}
    elif mode == 'description':
        c1_data = {'desc': desc1}
        c2_data = {'desc': desc2}
    elif mode == 'polished':
        c1_data = {'desc': polish1, 'polish': polish1}
        c2_data = {'desc': polish2, 'polish': polish2}
    else:
        # 默认回退
        c1_data = {'desc': desc1}
        c2_data = {'desc': desc2}

    return c1_data, c2_data

# ==========================================
# 4. 主流程
# ==========================================


def run_pure_sbert():
    print(f"🔥 启动 SBERT 联邦混合微调 | 模式: {TEXT_MODE}")

    c1_data, c2_data = load_data_dict(TEXT_MODE)
    test_pairs = data_loader.load_alignment_pairs(
        config.BASE_PATH + "ref_pairs")

    server = ServerSBERT()
    c1 = ClientSBERT("C1", config.DEVICE, c1_data)
    c2 = ClientSBERT("C2", config.DEVICE, c2_data)

    results = []

    for r in range(ROUNDS + 1):
        print(f"\n{'-'*50}\n🔄 Round {r} / {ROUNDS} [{TEXT_MODE}]\n{'-'*50}")

        # --- Step 1: 编码所有模态 (关键：接住 IDs) ---
        print("   Encoding Entities (Desc & Polish)...")
        ids1_desc, emb1_desc = c1.encode_all('desc')
        ids2_desc, emb2_desc = c2.encode_all('desc')

        ids1_poly, emb1_polish = c1.encode_all('polish')
        ids2_poly, emb2_polish = c2.encode_all('polish')

        # --- Step 2: 双重评估 (Dual Evaluation) ---
        print("   📊 Evaluating on [Description] Input...")
        # 严格对齐 ID
        e1_desc = {id_val: emb1_desc[i] for i, id_val in enumerate(ids1_desc)}
        e2_desc = {id_val: emb2_desc[i] for i, id_val in enumerate(ids2_desc)}
        h_d, m_d = evaluate.evaluate_alignment(test_pairs, {k: torch.zeros(1) for k in e1_desc}, {k: torch.zeros(1) for k in e2_desc},
                                               torch.nn.Identity(), torch.nn.Identity(), config.EVAL_K_VALUES,
                                               sbert_1=e1_desc, sbert_2=e2_desc, alpha=0.0)

        print("   📊 Evaluating on [Polished] Input...")
        # 严格对齐 ID
        e1_pol = {id_val: emb1_polish[i] for i, id_val in enumerate(ids1_poly)}
        e2_pol = {id_val: emb2_polish[i] for i, id_val in enumerate(ids2_poly)}
        h_p, m_p = evaluate.evaluate_alignment(test_pairs, {k: torch.zeros(1) for k in e1_pol}, {k: torch.zeros(1) for k in e2_pol},
                                               torch.nn.Identity(), torch.nn.Identity(), config.EVAL_K_VALUES,
                                               sbert_1=e1_pol, sbert_2=e2_pol, alpha=0.0)

        results.append({
            "round": r,
            "desc_hits1": h_d[1], "desc_mrr": m_d,
            "poly_hits1": h_p[1], "poly_mrr": m_p
        })

        print(
            f"   🏆 Round {r} Result: Desc H@1={h_d[1]:.2f}% | Polish H@1={h_p[1]:.2f}%")

        if r == ROUNDS:
            break

        # --- Step 3: 生成伪标签 (基于最好的 Desc) ---
        print("   Generating Pseudo-labels (based on Description)...")
        emb1_cpu = emb1_desc.cpu()
        emb2_cpu = emb2_desc.cpu()
        sim = torch.mm(F.normalize(emb1_cpu), F.normalize(emb2_cpu).T)
        vals1, idx1 = sim.max(dim=1)
        vals2, idx2 = sim.max(dim=0)

        pseudo_pairs = []
        for i in range(len(idx1)):
            j = idx1[i].item()
            if idx2[j].item() == i and vals1[i] > PSEUDO_THRESHOLD:
                pseudo_pairs.append((i, j))

        print(f"   🌱 Found {len(pseudo_pairs)} pairs.")
        if len(pseudo_pairs) < 50:
            continue

        # --- Step 4: 准备混合训练目标 ---
        # 这里的 i 和 j 是 ids1_desc 和 ids2_desc 的索引
        p_idx1 = [p[0] for p in pseudo_pairs]
        p_idx2 = [p[1] for p in pseudo_pairs]
        real_ids1 = [ids1_desc[i] for i in p_idx1]
        real_ids2 = [ids2_desc[i] for i in p_idx2]

        # 目标1: Peer 的 Description Embedding (直接索引获取)
        target_desc_c1 = emb2_desc[p_idx2]
        target_desc_c2 = emb1_desc[p_idx1]

        # 目标2: Peer 的 Polished Embedding (需要查字典，防止 ID 错位)
        def get_targets_by_id(target_dict, id_list):
            targets = []
            for eid in id_list:
                if eid in target_dict:
                    targets.append(target_dict[eid])
                else:
                    # 如果对面没有 polish，就拿 desc 顶替 (Fallback)
                    targets.append(torch.zeros_like(emb1_desc[0]))
            return torch.stack(targets)

        target_polish_c1 = get_targets_by_id(e2_pol, real_ids2)
        target_polish_c2 = get_targets_by_id(e1_pol, real_ids1)

        # --- Step 5: 混合训练 ---
        # C1 (学习 C2 的 desc 和 polish)
        c1.update_pseudo_labels(real_ids1, target_desc_c1, target_polish_c1)
        _, l1 = c1.train_mixed()
        print(f"   📉 C1 Loss: {l1:.6f}")

        # C2 (学习 C1 的 desc 和 polish)
        c2.update_pseudo_labels(real_ids2, target_desc_c2, target_polish_c2)
        _, l2 = c2.train_mixed()
        print(f"   📉 C2 Loss: {l2:.6f}")

        # --- Step 6: 聚合 ---
        global_w = server.aggregate(
            [c1.sbert.state_dict(), c2.sbert.state_dict()])
        c1.sbert.load_state_dict(global_w)
        c2.sbert.load_state_dict(global_w)

        clean_memory()

    print(f"\n🏆 最终对照结果: {TEXT_MODE}")
    print(f"{'Round':<6} | {'Desc H@1':<10} | {'Polish H@1':<10}")
    print("-" * 35)
    for res in results:
        print(
            f"{res['round']:<6} | {res['desc_hits1']:<10.2f} | {res['poly_hits1']:<10.2f}")

    # ==========================================
    # 💾 新增: 保存最终的全局模型
    # ==========================================
    save_dir = f"checkpoints/sbert_{TEXT_MODE}_round{ROUNDS}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print(f"\n💾 Saving global model to {save_dir} ...")
    server.global_model.save(save_dir)
    print("✅ Model saved successfully!")

    utils_logger.log_experiment_result(
        f"FedSBERT_Pure_{TEXT_MODE}", config.CURRENT_DATASET_NAME, results[-1])


if __name__ == "__main__":
    os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
    run_pure_sbert()
