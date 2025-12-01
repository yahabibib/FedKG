# 📄 bert_finetuner.py
# 【MLM版】执行 Masked Language Modeling 任务

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForMaskedLM, DataCollatorForLanguageModeling, TrainingArguments, Trainer
import config
import logging
import os
import shutil

# 自定义 Dataset


class TripleDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len=32):
        # 三元组句子很短，32 足够了，省显存快
        self.encodings = tokenizer(
            texts, return_tensors='pt', max_length=max_len, truncation=True, padding='max_length')

    def __getitem__(self, idx):
        return {key: val[idx] for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)


def fine_tune_with_mlm(model_path, sentences, save_path, epochs=3, batch_size=32):
    """
    对 BERT 进行 MLM 预训练 (Domain Adaptive Pre-training)
    """
    logging.info(
        f"   🔧 [MLM Pre-training] Starting with {len(sentences)} sentences...")

    # 1. 加载 HuggingFace 原生模型 (支持 MaskedLM)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForMaskedLM.from_pretrained(model_path)

    # 强制移动到 MPS/CUDA
    if config.DEVICE.type == 'mps':
        model.to("mps")
    elif config.DEVICE.type == 'cuda':
        model.to("cuda")

    # 2. 数据集
    dataset = TripleDataset(sentences, tokenizer)

    # 3. 自动 Mask (15% 概率)
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

    # 4. 训练参数
    training_args = TrainingArguments(
        output_dir=save_path,
        overwrite_output_dir=True,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        save_steps=5000,
        save_total_limit=1,
        logging_steps=50,  # 频繁打印日志
        learning_rate=2e-5,
        weight_decay=0.01,
        report_to="none",
        use_mps_device=True if config.DEVICE.type == 'mps' else False,
        dataloader_pin_memory=False  # 优化 MPS 内存
    )

    # 5. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
    )

    logging.info("   🚀 Starting MLM training...")
    trainer.train()

    # 6. 保存 (存为 HuggingFace 格式，SBERT 也能读)
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    trainer.save_model(save_path)
    tokenizer.save_pretrained(save_path)

    logging.info(f"   ✅ Structure-Aware BERT saved to: {save_path}")

    # 清理
    del model, trainer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    return save_path
