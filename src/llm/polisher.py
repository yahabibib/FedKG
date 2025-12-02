# src/llm/polisher.py
import torch
import logging
from typing import List, Dict, Tuple
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import Dataset, DataLoader

from src.utils.config import Config


class KnowledgePolisher:
    """
    Stage 1: LLM 增强器
    负责将结构化三元组转化为自然语言描述 (Structure-to-Text)。
    """

    def __init__(self, config: Config):
        self.cfg = config
        self.logger = logging.getLogger("KnowledgePolisher")
        self.device = self.cfg.device

        self.logger.info(
            f"正在加载 LLM: {self.cfg.llm_model_id} 到 {self.device}...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.cfg.llm_model_id,
                trust_remote_code=True
            )
            # 批量生成必须设置 padding_side='left'
            self.tokenizer.padding_side = 'left'
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            self.model = AutoModelForCausalLM.from_pretrained(
                self.cfg.llm_model_id,
                torch_dtype=torch.float32,  # 如果显存够大，可用 float16
                device_map=self.device,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            self.model.eval()
            self.logger.info("✅ LLM 加载完成。")
        except Exception as e:
            self.logger.error(f"❌ LLM 加载失败: {e}")
            raise e

    def construct_prompt(self, entity_name: str, relations: List[str], lang: str = 'en') -> str:
        """
        构建强约束 Prompt
        """
        data_str = "\n".join(relations)

        if lang == 'zh':
            prompt = (
                f"请将以下关于主语“{entity_name}”的知识图谱数据，改写成一段通顺的中文介绍。\n"
                f"【数据说明】\n"
                f"格式为：'- 关系: X -> 对象: Y (对象背景: Z)'\n"
                f"注意：Z 是对对象 Y 的描述，**绝对不是**对主语“{entity_name}”的描述！不要张冠李戴。\n\n"
                f"【要求】\n"
                f"1. 必须以“{entity_name}”开头。\n"
                f"2. 包含所有关系和对象。\n"
                f"3. 可以利用(对象背景)简单解释 Y 是什么，但不要照抄，也不要把 Y 的属性安在“{entity_name}”头上。\n"
                f"【数据列表】\n{data_str}\n\n"
                f"直接输出结果："
            )
        else:
            prompt = (
                f"Summarize the KG data about '{entity_name}' into a paragraph.\n"
                f"【Format】\n"
                f"'- Relation: X -> Object: Y (Context: Z)' means Z describes Y, NOT '{entity_name}'.\n\n"
                f"【Requirements】\n"
                f"1. Start with '{entity_name}'.\n"
                f"2. Include all relations.\n"
                f"3. Use (Context) to briefly explain Y, but DO NOT attribute Z's properties to '{entity_name}'.\n\n"
                f"【Data】\n{data_str}\n\n"
                f"Output:"
            )
        return prompt

    def batch_generate(self, prompts: List[str], batch_size: int = 8) -> List[str]:
        """
        执行批量推理
        """
        results = []
        # 使用简单的切片进行 batch 处理
        for i in tqdm(range(0, len(prompts), batch_size), desc="Batch Inference"):
            batch_prompts = prompts[i: i + batch_size]

            # 构造 Chat 模板
            batch_inputs = []
            for p in batch_prompts:
                messages = [
                    {"role": "system",
                        "content": "You are a helpful knowledge graph assistant."},
                    {"role": "user", "content": p}
                ]
                text = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                batch_inputs.append(text)

            # Tokenize
            inputs = self.tokenizer(
                batch_inputs,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=1024
            ).to(self.device)

            # Generate
            with torch.no_grad():
                generated_ids = self.model.generate(
                    inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_new_tokens=200,
                    do_sample=False  # 确定性生成，便于复现
                )

            # Decode (只提取新生成部分)
            input_len = inputs.input_ids.shape[1]
            for gen_ids in generated_ids:
                new_ids = gen_ids[input_len:]
                response = self.tokenizer.decode(
                    new_ids, skip_special_tokens=True).strip()
                # 简单的后处理清洗
                response = response.replace(
                    "Output:", "").replace("结果:", "").strip()
                results.append(response)

        return results

    def clean_memory(self):
        """释放显存"""
        del self.model
        del self.tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
