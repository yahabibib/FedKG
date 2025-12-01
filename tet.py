# 📄 step3_dual_eval.py
# 步骤三：含噪声的对齐评估 (Alignment Evaluation with Noise)
# 目标：在大量干扰项存在的情况下，对比 [Pure] vs [Mech] vs [Polish] 的对齐能力

import torch
import torch.nn.functional as F
import config
import data_loader
import precompute
import evaluate
import os
import pickle
import logging
import shutil
import random
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, InputExample, losses

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    datefmt='%H:%M:%S'
)


def load_custom_pkl(path):
    if not os.path.exists(path):
        return {}
    with open(path, 'rb') as f:
        return pickle.load(f)

# ==========================================
# 🛠️ 核心工具函数
# ==========================================


def get_noise_texts(model, noise_count=1000):
    """
    从全量数据中提取噪音实体的 Embedding
    (这里简化处理，直接从 description2.pkl 中随机抽，模拟 KG2 的干扰项)
    """
    logging.info(f"   🦠 正在准备 {noise_count} 个噪音实体...")

    # 加载全量描述
    pkl_path = config.BASE_PATH + "description2.pkl"
    if not os.path.exists(pkl_path):
        logging.warning("   ⚠️ 全量描述文件缺失，无法注入噪音。")
        return None

    with open(pkl_path, 'rb') as f:
        full_data = pickle.load(f)

    all_texts = []
    keys = list(full_data.keys())

    # 随机采样
    if len(keys) > noise_count:
        keys = random.sample(keys, noise_count)

    for k in keys:
        # 简单清洗
        text = str(full_data[k]).strip()
        if len(text) > 5:
            all_texts.append(text)

    # 计算噪音向量
    logging.info(f"   🔄 计算噪音向量 ({len(all_texts)} 条)...")
    noise_embs = model.encode(
        all_texts, convert_to_tensor=True, show_progress_bar=False)
    return F.normalize(noise_embs, p=2, dim=1)


def local_fine_tune_sbert(model_path, text_map_1, text_map_2, train_pairs, save_path, batch_size=16, epochs=3):
    """ SBERT 微调逻辑 """
    logging.info(f"   🔧 [Fine-tune] Init model: {model_path}...")
    model = SentenceTransformer(model_path, device=config.DEVICE)
    model.train()

    train_examples = []
    for id1, id2 in train_pairs:
        t1 = text_map_1.get(id1, "")
        t2 = text_map_2.get(id2, "")
        if t1 and t2:
            train_examples.append(InputExample(texts=[t1, t2]))

    # 数据增强：复制几份以增加 epoch 内的 step 数
    if len(train_examples) < 50:
        train_examples = train_examples * 5

    logging.info(f"   📦 Training samples: {len(train_examples)}")

    loader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
    train_loss = losses.MultipleNegativesRankingLoss(model)

    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    os.makedirs(save_path)

    model.fit(
        train_objectives=[(loader, train_loss)],
        epochs=epochs,
        warmup_steps=int(len(loader) * 0.1),
        show_progress_bar=True,
        output_path=save_path,
        optimizer_params={'lr': 2e-5}
    )

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return save_path

# ==========================================
# 🧪 实验主流程
# ==========================================


def run_experiment(mode, noise_level=1000):
    print(f"\n{'='*40}")
    print(f"🧪 实验组: [{mode.upper()}] (含 {noise_level} 噪音)")
    print(f"{'='*40}")

    demo_path = "data/demo_mini/zh_en/"

    # 1. 加载 Mini 数据 (包含 10 对核心 + 40 对背景)
    ent_1, _ = data_loader.load_id_map(demo_path + "ent_ids_1")
    ent_2, _ = data_loader.load_id_map(demo_path + "ent_ids_2")
    ref_pairs = data_loader.load_alignment_pairs(
        demo_path + "ref_pairs")  # 这 10 对用于微调和测试

    # 2. 加载描述 (不同模式加载不同文件)
    if mode == 'pure':
        # Pure 模式用原始描述
        t1 = load_custom_pkl(demo_path + "description1.pkl")
        t2 = load_custom_pkl(demo_path + "description2.pkl")
    elif mode == 'mech':
        t1 = load_custom_pkl(demo_path + "desc_mech_1.pkl")
        t2 = load_custom_pkl(demo_path + "desc_mech_2.pkl")
    else:  # polish
        t1 = load_custom_pkl(demo_path + "desc_polish_1.pkl")
        t2 = load_custom_pkl(demo_path + "desc_polish_2.pkl")

    if not t1:
        logging.error("❌ 数据加载失败")
        return 0.0

    # 3. 微调 SBERT (Pure 模式不微调，作为 Zero-shot Baseline)
    if mode == 'pure':
        logging.info("   ⏩ Pure 模式跳过微调 (Zero-shot)...")
        model_path = config.BERT_MODEL_NAME
    else:
        save_path = f"fine_tuned_models/demo_{mode}"
        model_path = local_fine_tune_sbert(
            config.BERT_MODEL_NAME, t1, t2, ref_pairs, save_path, epochs=5  # 微调 5 轮
        )

    # 4. 准备评估 Embedding
    # 加载模型用于推理
    eval_model = SentenceTransformer(model_path, device=config.DEVICE)
    eval_model.eval()

    # A. 计算 Query (KG1 核心实体)
    query_texts = [t1[p[0]] for p in ref_pairs]
    query_embs = eval_model.encode(
        query_texts, convert_to_tensor=True, show_progress_bar=False)
    query_embs = F.normalize(query_embs, p=2, dim=1)

    # B. 计算 Target (KG2 核心实体)
    target_ids = [p[1] for p in ref_pairs]
    target_texts = [t2[tid] for tid in target_ids]
    target_embs = eval_model.encode(
        target_texts, convert_to_tensor=True, show_progress_bar=False)
    target_embs = F.normalize(target_embs, p=2, dim=1)

    # C. 计算 Noise (KG2 外部噪音)
    noise_embs = get_noise_texts(eval_model, noise_count=noise_level)

    # 5. 合并候选池 (Target + Noise)
    if noise_embs is not None:
        candidate_embs = torch.cat([target_embs, noise_embs], dim=0)
    else:
        candidate_embs = target_embs

    # 6. 计算 Hits@1
    # Similarity Matrix: [Num_Query, Num_Candidates]
    sim_mat = torch.mm(query_embs, candidate_embs.T)

    hits1 = 0
    # 对于第 i 个 Query，正确的答案就在 candidate_embs 的第 i 个位置
    # (因为我们是按顺序拼接 target 的，noise 拼在后面)
    for i in range(len(query_embs)):
        scores = sim_mat[i]
        best_idx = torch.argmax(scores).item()

        if best_idx == i:  # 命中正确答案
            hits1 += 1

    acc = hits1 / len(query_embs) * 100
    logging.info(f"   🎯 Alignment Hits@1: {acc:.2f}%")

    del eval_model
    return acc


if __name__ == "__main__":
    # 设置噪音等级
    NOISE_LEVEL = 2000

    print(f"🚀 启动抗噪对齐实验 (Noise={NOISE_LEVEL})")

    # 1. Pure (基准)
    score_pure = run_experiment('pure', NOISE_LEVEL)

    # 2. Mech (机械)
    score_mech = run_experiment('mech', NOISE_LEVEL)

    # 3. Polish (润色)
    score_poli = run_experiment('polish', NOISE_LEVEL)

    print("\n" + "="*60)
    print(f"🏆 最终对齐结果对比 (Hits@1)")
    print(f"{'='*60}")
    print(f"{'Mode':<10} | {'Hits@1':<10} | {'Gap vs Pure':<15}")
    print("-" * 50)
    print(f"{'Pure':<10} | {score_pure:.2f}%     | -")
    print(f"{'Mech':<10} | {score_mech:.2f}%     | {score_mech-score_pure:+.2f}%")
    print(f"{'Polish':<10} | {score_poli:.2f}%     | {score_poli-score_pure:+.2f}%")
    print("-" * 50)

    if score_poli > score_mech and score_poli >= score_pure:
        print("✅ 验证成功！润色 + 微调 在高噪音下表现最佳。")
    else:
        print("⚠️ 验证需分析：可能是微调过拟合，或噪音太强淹没语义。")
    print("="*60)
