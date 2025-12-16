# run_centralized.py (The Final Solution: MNN Iterative)
import hydra
from omegaconf import DictConfig
import logging
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import copy

from src.data.dataset import AlignmentTaskData
from src.utils.device_manager import DeviceManager
from src.utils.metrics import eval_alignment
from src.utils.logger import log_experiment_result
from src.models.decoupled import DecoupledModel
from src.utils.graph import build_adjacency_matrix
from sentence_transformers import SentenceTransformer

log = logging.getLogger(__name__)


class CentralizedDataset:
    def __init__(self, d1, d2):
        self.num_ent1 = d1.num_entities
        self.num_ent2 = d2.num_entities
        self.num_entities = self.num_ent1 + self.num_ent2
        self.ids = d1.ids + d2.ids
        offset = self.num_ent1
        t1 = d1.triples
        t2 = []
        for h, r, t in d2.triples:
            t2.append((h + offset, r, t + offset))
        self.triples = t1 + t2
        log.info(f"[Centralized] Merged Graph: {self.num_entities} nodes")


def get_mnn_pairs(emb1, emb2, device='cpu', batch_size=2048):
    """
    计算双向互近邻 (Mutual Nearest Neighbors)
    这是集中式对齐中精度最高的挖掘方式
    """
    num_src = emb1.shape[0]
    num_tgt = emb2.shape[0]

    # 1. Source -> Target 的最近邻
    src_to_tgt = torch.zeros(num_src, dtype=torch.long, device=device)
    src_vals = torch.zeros(num_src, device=device)

    # 分批计算以节省显存
    for i in range(0, num_src, batch_size):
        end = min(i + batch_size, num_src)
        batch_src = emb1[i:end]
        sim = torch.mm(batch_src, emb2.T)  # [B, N_tgt]
        vals, inds = torch.max(sim, dim=1)
        src_to_tgt[i:end] = inds
        src_vals[i:end] = vals

    # 2. Target -> Source 的最近邻
    tgt_to_src = torch.zeros(num_tgt, dtype=torch.long, device=device)

    for i in range(0, num_tgt, batch_size):
        end = min(i + batch_size, num_tgt)
        batch_tgt = emb2[i:end]
        sim = torch.mm(batch_tgt, emb1.T)  # [B, N_src]
        _, inds = torch.max(sim, dim=1)
        tgt_to_src[i:end] = inds

    # 3. 找交集 (互为最近邻)
    pairs = []
    # 遍历所有 source 节点
    for i in range(num_src):
        j = src_to_tgt[i].item()  # i 觉得 j最好
        if tgt_to_src[j].item() == i:  # j 也觉得 i最好
            # 只有互为唯一挚爱，才算配对
            pairs.append((i, j, src_vals[i].item()))

    return pairs


@hydra.main(config_path="configs", config_name="config", version_base="1.2")
def main(cfg: DictConfig):
    log.info(
        f"🏢 Starting CENTRALIZED Experiment (MNN Strategy): {cfg.task.model.encoder_name}")
    dm = DeviceManager(cfg.system)
    raw_data = AlignmentTaskData(cfg.data)

    # 1. 合并数据
    merged_data = CentralizedDataset(raw_data.source, raw_data.target)
    adj, edge_types, num_rels = build_adjacency_matrix(
        merged_data.triples, merged_data.num_entities, device='cpu', return_edge_types=True
    )

    # 2. SBERT Anchors (Fixed)
    sbert = SentenceTransformer(cfg.task.sbert_checkpoint, device='cpu')
    texts1 = raw_data.source.get_text_list(raw_data.source.ids, 'desc')
    texts2 = raw_data.target.get_text_list(raw_data.target.ids, 'desc')

    log.info("   Computing SBERT anchors...")
    sbert.to(dm.main_device)
    with torch.no_grad():
        fixed_anchors = sbert.encode(
            texts1 + texts2, batch_size=64, convert_to_tensor=True, show_progress_bar=True, device=dm.main_device
        ).cpu()
    sbert.to('cpu')
    del sbert

    # 训练目标 (可变)
    train_anchors = fixed_anchors.clone()

    # 3. 模型
    device = dm.main_device
    model = DecoupledModel(
        cfg.task.model, merged_data.num_entities, num_relations=num_rels)
    model.to(device)

    # 4. 训练参数
    rounds = 10
    local_epochs = 30  # 每轮适中
    optimizer = optim.Adam(model.parameters(), lr=5e-4)  # 恢复正常的 LR，因为 MNN 很稳
    criterion = nn.MarginRankingLoss(margin=cfg.task.federated.margin)
    SAMPLE_SIZE = 8192

    # 阈值设置 (MNN 已经很严了，相似度阈值可以适中)
    SIM_THRESH = 0.6
    SEMANTIC_CHECK_THRESH = 0.5  # 双重保险

    best_hits1 = 0.0

    for r in range(rounds + 1):
        log.info(
            f"\n{'='*30}\n🔄 Centralized Round {r}/{rounds} (MNN Mining)\n{'='*30}")

        # --- A. 训练 ---
        model.train()
        curr_epochs = 50 if r == 0 else local_epochs

        for ep in range(curr_epochs):
            out = model(adj, edge_types)  # Full Graph Forward

            perm = torch.randperm(merged_data.num_entities)
            batch_ids = perm[:SAMPLE_SIZE].to(device)

            struct_batch = out[batch_ids]
            target_batch = train_anchors[batch_ids.cpu()].to(device)

            # Hard Mining
            with torch.no_grad():
                sim_mat = torch.mm(F.normalize(struct_batch),
                                   F.normalize(target_batch).T)
                sim_mat.fill_diagonal_(-2.0)
                neg_idx = sim_mat.argmax(dim=1)

            pos_sim = F.cosine_similarity(struct_batch, target_batch)
            neg_sim = F.cosine_similarity(struct_batch, target_batch[neg_idx])

            loss = criterion(pos_sim, neg_sim, torch.ones_like(pos_sim))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (ep+1) % 10 == 0:
                log.info(
                    f"   [Train] Ep {ep+1}/{curr_epochs} Loss={loss.item():.4f}")

        # --- B. 评估 ---
        if dm.is_offload_enabled():
            dm.clean_memory()
        model.eval()
        with torch.no_grad():
            full_emb = model(adj, edge_types).cpu()

        emb1 = full_emb[:raw_data.source.num_entities]
        emb2 = full_emb[raw_data.source.num_entities:]

        # 评估用字典
        d1 = {id: emb1[i] for i, id in enumerate(raw_data.source.ids)}
        d2 = {id: emb2[i] for i, id in enumerate(raw_data.target.ids)}
        sd1 = {id: fixed_anchors[i]
               for i, id in enumerate(raw_data.source.ids)}
        sd2 = {id: fixed_anchors[raw_data.source.num_entities+i]
               for i, id in enumerate(raw_data.target.ids)}

        hits, mrr = eval_alignment(d1, d2, raw_data.test_pairs, [
                                   1], sd1, sd2, alpha=cfg.task.eval.alpha)

        log.info(f"   🏆 Round {r} Result: Hits@1={hits[1]:.2f}%")

        if hits[1] > best_hits1:
            best_hits1 = hits[1]

        if r == rounds:
            break

        # --- C. MNN 挖掘 (关键步骤) ---
        log.info(f"   💎 Running Mutual Nearest Neighbor (MNN) Mining...")

        # 1. 计算 embedding 相似度
        # 归一化以便计算 cosine
        e1_norm = F.normalize(emb1.to(device))
        e2_norm = F.normalize(emb2.to(device))

        # 获取 MNN 对
        raw_pairs = get_mnn_pairs(e1_norm, e2_norm, device=device)

        # 2. 过滤 (相似度阈值 + 语义一致性)
        valid_pairs = []
        for i, j, score in raw_pairs:
            if score > SIM_THRESH:
                # 语义校验
                s1 = fixed_anchors[i]
                s2 = fixed_anchors[j + raw_data.source.num_entities]
                sem_sim = F.cosine_similarity(
                    s1.unsqueeze(0), s2.unsqueeze(0)).item()

                if sem_sim > SEMANTIC_CHECK_THRESH:
                    valid_pairs.append((i, j))

        log.info(
            f"   🔍 MNN Found {len(raw_pairs)} -> Validated {len(valid_pairs)} pairs.")

        # 3. 更新 Anchors
        if len(valid_pairs) > 0:
            src_idx = [p[0] for p in valid_pairs]
            tgt_idx = [p[1] for p in valid_pairs]
            offset = raw_data.source.num_entities
            tgt_idx_offset = [t + offset for t in tgt_idx]

            # 交叉更新，拉近距离
            train_anchors[src_idx] = emb2[tgt_idx].cpu()
            train_anchors[tgt_idx_offset] = emb1[src_idx].cpu()
            log.info(f"   ✅ Anchors Updated.")

    log.info(f"🏁 Centralized Final Best Hits@1: {best_hits1:.2f}%")
    res = {"setting": "centralized",
           "encoder": cfg.task.model.encoder_name, "hits1": best_hits1}
    log_experiment_result("centralized_upperbound", cfg.data.name, res)


if __name__ == "__main__":
    os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
    main()
