# 📄 step3_four_experiments.py
# 【四组对比实验】验证结构化知识注入的有效性
# 1. Baseline (原始描述)
# 2. Structure Only (仅润色结构)
# 3. Combined (原始 + 润色)
# 4. Finetuned (在 Combined 数据上微调，冻结底层)

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
import gc
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, InputExample, losses

# 配置日志
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(message)s', datefmt='%H:%M:%S')


def clear_memory():
    """ 强制清理显存和内存 """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def clean_name(uri):
    if not isinstance(uri, str):
        return str(uri)
    return uri.split('/')[-1].replace('_', ' ')

# ==========================================
# 🛠️ 数据处理核心函数
# ==========================================


def load_pickle_data(path):
    """ 通用 Pickle 加载 """
    if not os.path.exists(path):
        return {}
    with open(path, 'rb') as f:
        return pickle.load(f)


def prepare_data(ent_map, raw_data, polish_data, mode):
    """
    根据模式构造文本数据
    mode: 'raw' | 'struct' | 'combined'
    """
    final_text = {}
    stats = {"raw_only": 0, "struct_only": 0,
             "combined": 0, "name_fallback": 0}

    # 建立 URI -> Text 索引 (处理 key 不一致问题)
    def build_lookup(data):
        lookup = {}
        for k, v in data.items():
            val = str(v).strip()
            if not val:
                continue
            if isinstance(k, int):
                lookup[k] = val
            elif isinstance(k, str):
                lookup[k] = val
                lookup[k.strip('<>')] = val
        return lookup

    raw_lookup = build_lookup(raw_data)
    polish_lookup = build_lookup(polish_data)

    for eid, uri in ent_map.items():
        # 1. 获取 Raw 描述
        raw_txt = raw_lookup.get(eid, "")
        if not raw_txt:
            clean_uri = uri.strip('<>')
            raw_txt = raw_lookup.get(uri, raw_lookup.get(clean_uri, ""))

        # 2. 获取 Polish 描述 (并尝试提取纯结构部分)
        # 假设 Step 2 生成格式为: "Raw [SEP] Struct"
        full_polish = polish_lookup.get(eid, "")
        if not full_polish:
            clean_uri = uri.strip('<>')
            full_polish = polish_lookup.get(
                uri, polish_lookup.get(clean_uri, ""))

        struct_txt = ""
        if "[SEP]" in full_polish:
            parts = full_polish.split("[SEP]")
            if len(parts) > 1:
                struct_txt = parts[1].strip()
        else:
            # 如果没有 SEP，可能整个就是结构，或者 Step 2 没生成好
            # 这里假设如果没有 SEP，整个视为结构（如果是润色文件的话）
            struct_txt = full_polish

        # 3. 根据模式组合
        text = ""

        if mode == 'raw':
            text = raw_txt

        elif mode == 'struct':
            text = struct_txt

        elif mode == 'combined':
            # 核心拼接逻辑：兜底策略
            if raw_txt and struct_txt:
                text = f"{raw_txt} [SEP] {struct_txt}"
                stats["combined"] += 1
            elif raw_txt:
                text = raw_txt
                stats["raw_only"] += 1
            elif struct_txt:
                text = struct_txt
                stats["struct_only"] += 1

        # 4. 最终兜底：如果为空，用名字
        if len(text) < 2:
            text = clean_name(uri)
            stats["name_fallback"] += 1

        final_text[eid] = text

    logging.info(f"   📊 Data Prep [{mode}]: {stats}")
    return final_text

# ==========================================
# 🔧 SBERT 微调函数 (含冻结层)
# ==========================================


def fine_tune_sbert(model_path, text_map_1, text_map_2, train_pairs, save_path, batch_size=8, epochs=3):
    clear_memory()
    logging.info(f"   🔧 [Fine-tuning] Init: {model_path}")

    model = SentenceTransformer(model_path, device=config.DEVICE)

    # --- 冻结前 10 层 ---
    auto_model = model._first_module().auto_model
    for param in auto_model.embeddings.parameters():
        param.requires_grad = False

    if hasattr(auto_model, 'encoder') and hasattr(auto_model.encoder, 'layer'):
        layers = auto_model.encoder.layer
        freeze_num = max(0, len(layers) - 2)  # 只训最后2层
        for i in range(freeze_num):
            for param in layers[i].parameters():
                param.requires_grad = False
        logging.info(f"   🧊 Frozen {freeze_num}/{len(layers)} layers.")

    model.train()

    # 构造数据
    train_examples = []
    for id1, id2 in train_pairs:
        t1 = text_map_1.get(id1, "")
        t2 = text_map_2.get(id2, "")
        # 简单过滤过短文本
        if len(t1) > 3 and len(t2) > 3:
            train_examples.append(InputExample(texts=[t1, t2]))

    logging.info(f"   📦 Training samples: {len(train_examples)}")

    if len(train_examples) < 10:
        logging.error("   ❌ Samples too few!")
        return None

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
    clear_memory()
    return save_path

# ==========================================
# 🧪 实验执行
# ==========================================


def run_experiment(exp_name, raw_1, raw_2, pol_1, pol_2, ent_1, ent_2, ref_pairs):
    print(f"\n>>> Running Experiment: {exp_name}")

    model_path = config.BERT_MODEL_NAME

    # 1. 准备数据
    if exp_name == "Baseline":
        t1 = prepare_data(ent_1, raw_1, pol_1, 'raw')
        t2 = prepare_data(ent_2, raw_2, pol_2, 'raw')

    elif exp_name == "Structure Only":
        t1 = prepare_data(ent_1, raw_1, pol_1, 'struct')
        t2 = prepare_data(ent_2, raw_2, pol_2, 'struct')

    elif exp_name == "Combined":
        t1 = prepare_data(ent_1, raw_1, pol_1, 'combined')
        t2 = prepare_data(ent_2, raw_2, pol_2, 'combined')

    elif exp_name == "Finetuned":
        t1 = prepare_data(ent_1, raw_1, pol_1, 'combined')
        t2 = prepare_data(ent_2, raw_2, pol_2, 'combined')

        save_path = "fine_tuned_models/exp4_finetuned"
        if not os.path.exists(save_path):
            model_path = fine_tune_sbert(
                model_path, t1, t2, ref_pairs, save_path)
        else:
            logging.info("   ✅ Loading existing finetuned model.")
            model_path = save_path

    # 2. 评估
    logging.info("   📊 Evaluating Alignment...")
    map_1 = (ent_1, None)
    map_2 = (ent_2, None)

    # 强制重算 Embedding
    sb_1 = precompute.get_bert_embeddings(
        map_1, t1, "KG1", cache_file=None, custom_model_path=model_path)
    sb_2 = precompute.get_bert_embeddings(
        map_2, t2, "KG2", cache_file=None, custom_model_path=model_path)

    h, _ = evaluate.evaluate_alignment(ref_pairs, sb_1, sb_2, torch.nn.Identity(
    ), torch.nn.Identity(), [1, 10], sbert_1=sb_1, sbert_2=sb_2, alpha=0.0)

    del sb_1, sb_2
    clear_memory()

    return h[1]


def run():
    print(f"{'='*60}")
    print("🧪 四组对比实验 (Four-Way Comparison)")
    print(f"{'='*60}")

    base_dir = config.BASE_PATH
    # 假设润色文件已在 data/dbp15k/zh_en/ 下
    polish_dir = "data/dbp15k/zh_en/"
    if not os.path.exists(polish_dir + "desc_polish_1.pkl"):
        # 兼容 demo 路径
        polish_dir = "data/demo_mini/zh_en/"

    # 加载基础数据
    ent_1, _ = data_loader.load_id_map(base_dir + "ent_ids_1")
    ent_2, _ = data_loader.load_id_map(base_dir + "ent_ids_2")
    ref_pairs = data_loader.load_alignment_pairs(base_dir + "ref_pairs")

    # 加载原始和润色文件
    logging.info("📂 Loading Pickle Files...")
    raw_1 = load_pickle_data(base_dir + "description1.pkl")
    raw_2 = load_pickle_data(base_dir + "description2.pkl")
    pol_1 = load_pickle_data(polish_dir + "desc_polish_1.pkl")
    pol_2 = load_pickle_data(polish_dir + "desc_polish_2.pkl")

    # 运行四组实验
    r1 = run_experiment("Baseline", raw_1, raw_2, pol_1,
                        pol_2, ent_1, ent_2, ref_pairs)
    r2 = run_experiment("Structure Only", raw_1, raw_2,
                        pol_1, pol_2, ent_1, ent_2, ref_pairs)
    r3 = run_experiment("Combined", raw_1, raw_2, pol_1,
                        pol_2, ent_1, ent_2, ref_pairs)
    r4 = run_experiment("Finetuned", raw_1, raw_2, pol_1,
                        pol_2, ent_1, ent_2, ref_pairs)

    print("\n" + "="*60)
    print(f"🏆 最终结果汇总 (Hits@1)")
    print(f"{'='*60}")
    print(f"{'1. Baseline (Raw)':<25} | {r1:.2f}%")
    print(f"{'2. Structure Only':<25} | {r2:.2f}%")
    print(f"{'3. Combined (Zero-shot)':<25} | {r3:.2f}%")
    print(f"{'4. Finetuned (Frozen)':<25} | {r4:.2f}%")
    print("-" * 60)

    best = max(r1, r2, r3, r4)
    if r4 == best:
        print("✅ 验证成功：结构化微调效果最佳！")
    elif r3 == best:
        print("⚠️ 验证发现：Combined Zero-shot 效果最佳，微调可能过拟合。")
    else:
        print("⚠️ 验证未达预期：Baseline 或 Structure Only 更好。")
    print("="*60)


if __name__ == "__main__":
    run()
