import torch
import torch.nn.functional as F
import logging
import numpy as np
from src.utils.metrics import eval_alignment

log = logging.getLogger(__name__)


def search_best_alpha(c1, c2, test_pairs, step=0.05, device='cpu'):
    """
    自动搜索最佳融合权重 Alpha。

    :param c1: Client 1 对象 (包含模型和 Anchors)
    :param c2: Client 2 对象
    :param test_pairs: 验证集/测试集对
    :param step: 搜索步长
    :return: best_alpha, best_metrics (dict)
    """
    # 1. 准备数据：提取两个 Client 的 Structure 和 SBERT 特征
    # 切换到评估模式
    c1.model.eval()
    c2.model.eval()

    with torch.no_grad():
        # 获取 Structure Embeddings (归一化)
        # 注意：这里需要把数据搬运到 CPU 以免显存爆炸，因为搜索过程主要是 CPU 密集型的矩阵运算
        emb1_struct = F.normalize(
            c1.model(c1.adj, c1.edge_types), p=2, dim=1).cpu()
        emb2_struct = F.normalize(
            c2.model(c2.adj, c2.edge_types), p=2, dim=1).cpu()

        # 获取 SBERT Anchors (归一化)
        emb1_sbert = F.normalize(c1.anchor_embeddings, p=2, dim=1).cpu()
        emb2_sbert = F.normalize(c2.anchor_embeddings, p=2, dim=1).cpu()

    # 准备字典格式，供 eval_alignment 使用
    d1_struct = {id: emb1_struct[i] for i, id in enumerate(c1.dataset.ids)}
    d2_struct = {id: emb2_struct[i] for i, id in enumerate(c2.dataset.ids)}

    d1_sbert = {id: emb1_sbert[i] for i, id in enumerate(c1.dataset.ids)}
    d2_sbert = {id: emb2_sbert[i] for i, id in enumerate(c2.dataset.ids)}

    # 2. 暴力搜索最佳 Alpha
    best_alpha = 0.0
    best_hits1 = -1.0
    best_metrics = {}

    # 生成搜索区间 [0.0, 0.05, ..., 1.0]
    search_range = np.arange(0.0, 1.0 + step/2, step)

    # 这里的 log 级别可以设为 debug，避免刷屏
    # log.debug(f"🔎 Tuning Alpha over {len(search_range)} steps...")

    for alpha in search_range:
        # 调用现有的评估函数
        # 注意：eval_alignment 内部实现了 score fusion: alpha * struct + (1-alpha) * sbert
        metrics, mrr = eval_alignment(
            d1_struct, d2_struct, test_pairs,
            k_values=[1, 10],
            sbert1_dict=d1_sbert, sbert2_dict=d2_sbert,
            alpha=alpha,
            device=device
        )

        if metrics[1] > best_hits1:
            best_hits1 = metrics[1]
            best_alpha = alpha
            best_metrics = metrics
            best_metrics['mrr'] = mrr

    log.info(
        f"   🎯 Best Alpha Found: {best_alpha:.2f} | Hits@1: {best_hits1:.2f}%")

    return best_alpha, best_metrics
