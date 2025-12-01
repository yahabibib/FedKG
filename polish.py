# 📄 step2_local_llm.py
# 【Prompt 逻辑加固版】
# 修复了"张冠李戴"问题 (防止将邻居属性安到头实体上)
# 缩短背景信息长度，防止喧宾夺主

import os
import pickle
import torch
import config
import data_loader
from tqdm import tqdm
import re
import time
from transformers import AutoModelForCausalLM, AutoTokenizer

# ==========================================
# 🔧 模型配置
# ==========================================
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
DEVICE = "cpu"

print(f"{'='*60}")
print(f"🤖 深度结构化润色 (Logic-Safe Prompt)")
print(f"   Model: {MODEL_ID}")
print(f"{'='*60}")

# 加载模型
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float32, device_map=DEVICE, trust_remote_code=True)
except Exception as e:
    print(f"❌ Load failed: {e}")
    exit()


def call_local_llm(prompt):
    messages = [{"role": "system", "content": "You are a helpful knowledge graph assistant."}, {
        "role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        generated_ids = model.generate(
            model_inputs.input_ids,
            attention_mask=model_inputs.attention_mask,
            max_new_tokens=150,
            do_sample=False
        )
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(
        model_inputs.input_ids, generated_ids)]
    return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()


def clean_name(uri):
    if not isinstance(uri, str):
        return str(uri)
    name = uri.split('/')[-1].replace('_',
                                      ' ').replace('<', '').replace('>', '')
    name = re.sub(r'(?<!^)(?=[A-Z])', ' ', name).lower()
    return name.strip()

# --- 智能摘要提取 (缩短长度) ---


def get_smart_summary(text):
    if not text:
        return ""
    text = re.sub(r"\(.*?\)", "", text).replace("（", "").replace("）", "")
    if "。" in text:
        sentences = text.split("。")
    else:
        sentences = text.split(". ")
    summary = sentences[0].strip()
    # 移除太长的背景，只保留前60个字符
    return summary[:60]


def run():
    demo_path = "data/demo_mini/zh_en/"
    if not os.path.exists(demo_path + "ent_ids_1"):
        print("❌ 未找到 Mini 数据集")
        return

    # 1. 加载数据
    print("📚 加载数据...")
    ent_1, _ = data_loader.load_id_map(demo_path + "ent_ids_1")
    ent_2, _ = data_loader.load_id_map(demo_path + "ent_ids_2")
    rel_1, _ = data_loader.load_id_map(demo_path + "rel_ids_1")
    rel_2, _ = data_loader.load_id_map(demo_path + "rel_ids_2")
    trip_1 = data_loader.load_triples(demo_path + "triples_1")
    trip_2 = data_loader.load_triples(demo_path + "triples_2")

    with open(demo_path + "description1.pkl", 'rb') as f:
        attr_1 = pickle.load(f)
    with open(demo_path + "description2.pkl", 'rb') as f:
        attr_2 = pickle.load(f)

    # 2. 处理函数
    def process_dataset(ent_map, rel_map, triples, attr_map, lang_code):
        mech_dict = {}
        polish_dict = {}

        adj = {}
        for h, r, t in triples:
            if h not in adj:
                adj[h] = []
            adj[h].append((r, t))

        lang_name = "Chinese" if lang_code == 'zh' else "English"
        print(f"\n🚀 Processing {len(ent_map)} entities for {lang_name}...")

        stats = {"polished": 0, "skipped": 0}
        pbar = tqdm(ent_map.items())

        for i, (eid, uri) in enumerate(pbar):
            name = clean_name(uri)
            base_desc = attr_map.get(eid, "")

            neighbors = adj.get(eid, [])[:4]

            rich_triples = []
            simple_triples = []

            for r, t in neighbors:
                r_n = clean_name(rel_map.get(r, "rel"))
                t_n = clean_name(ent_map.get(t, "ent"))
                t_desc_raw = attr_map.get(t, "")
                t_ctx = get_smart_summary(t_desc_raw)

                if t_ctx:
                    rich_item = f"- 关系: {r_n} -> 对象: {t_n} (对象背景: {t_ctx})"
                else:
                    rich_item = f"- 关系: {r_n} -> 对象: {t_n}"

                rich_triples.append(rich_item)
                simple_triples.append(f"{r_n}: {t_n}")

            if not simple_triples:
                mech_dict[eid] = base_desc
                polish_dict[eid] = base_desc
                stats["skipped"] += 1
                continue

            # A. 机械版
            mech_dict[eid] = f"{base_desc} [SEP] Structure: {'; '.join(simple_triples)}"

            # B. 润色版 (Prompt 逻辑加固)
            data_str = "\n".join(rich_triples)

            if lang_code == 'zh':
                prompt = (
                    f"请将以下关于主语“{name}”的知识图谱数据，改写成一段通顺的中文介绍。\n"
                    f"【数据说明】\n"
                    f"格式为：'- 关系: X -> 对象: Y (对象背景: Z)'\n"
                    f"注意：Z 是对对象 Y 的描述，**绝对不是**对主语“{name}”的描述！不要张冠李戴。\n\n"
                    f"【要求】\n"
                    f"1. 必须以“{name}”开头。\n"
                    f"2. 包含所有关系和对象。\n"
                    f"3. 可以利用(对象背景)简单解释 Y 是什么，但不要照抄，也不要把 Y 的属性安在“{name}”头上。\n"
                    f"【数据列表】\n{data_str}\n\n"
                    f"直接输出结果："
                )
            else:
                prompt = (
                    f"Summarize the KG data about '{name}' into a paragraph.\n"
                    f"【Format】\n"
                    f"'- Relation: X -> Object: Y (Context: Z)' means Z describes Y, NOT '{name}'.\n\n"
                    f"【Requirements】\n"
                    f"1. Start with '{name}'.\n"
                    f"2. Include all relations.\n"
                    f"3. Use (Context) to briefly explain Y, but DO NOT attribute Z's properties to '{name}'.\n\n"
                    f"【Data】\n{data_str}\n\n"
                    f"Output:"
                )

            polished = ""
            for _ in range(2):
                polished = call_local_llm(prompt)
                polished = polished.replace(
                    "Output:", "").replace("结果:", "").strip()
                if len(polished) > 5:
                    break
                time.sleep(0.1)

            if not polished:
                polished = "; ".join(simple_triples)

            # 监控
            if i % 20 == 0:
                tqdm.write("-" * 40)
                tqdm.write(f"🔎 [Monitor #{i}] Entity: {name}")
                tqdm.write(f"   In:\n{data_str}")
                tqdm.write(f"   Out:\n{polished}")
                tqdm.write("-" * 40)

            polish_dict[eid] = f"{base_desc} [SEP] {polished}"
            stats["polished"] += 1

        print(
            f"   Done. Polished: {stats['polished']}, Skipped: {stats['skipped']}")
        return mech_dict, polish_dict

    m1, p1 = process_dataset(ent_1, rel_1, trip_1, attr_1, 'zh')
    m2, p2 = process_dataset(ent_2, rel_2, trip_2, attr_2, 'en')

    print(f"\n💾 保存结果...")
    with open(demo_path + "desc_mech_1.pkl", 'wb') as f:
        pickle.dump(m1, f)
    with open(demo_path + "desc_mech_2.pkl", 'wb') as f:
        pickle.dump(m2, f)
    with open(demo_path + "desc_polish_1.pkl", 'wb') as f:
        pickle.dump(p1, f)
    with open(demo_path + "desc_polish_2.pkl", 'wb') as f:
        pickle.dump(p2, f)

    print("✅ 深度润色完成！请运行 step3_train_eval.py")


if __name__ == "__main__":
    run()
