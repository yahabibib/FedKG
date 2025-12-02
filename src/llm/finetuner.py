# src/llm/finetuner.py
import logging
import os
import shutil
from typing import List, Tuple
import torch
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator

from src.utils.config import Config


class SBERTFinetuner:
    """
    Stage 1.2: SBERT 微调器
    使用对比学习 (InfoNCE Loss) 将 LLM 生成的结构化文本注入 SBERT。
    """

    def __init__(self, config: Config):
        self.cfg = config
        self.logger = logging.getLogger("SBERTFinetuner")
        self.device = self.cfg.device

    def fine_tune(self,
                  train_pairs: List[Tuple[str, str]],
                  output_path: str,
                  epochs: int = 3,
                  batch_size: int = 16,
                  freeze_layers: int = 10):
        """
        执行微调
        :param train_pairs: List of (anchor_text, positive_text)
                            例如 [(原始描述, 润色后的结构文本), ...]
        """
        self.logger.info(f"🚀 开始 SBERT 微调，基座: {self.cfg.sbert_model_path}")
        self.logger.info(
            f"   样本数: {len(train_pairs)} | Epochs: {epochs} | Batch: {batch_size}")

        # 1. 加载模型
        model = SentenceTransformer(
            self.cfg.sbert_model_path, device=str(self.device))

        # 2. 冻结底层参数 (Layer Freezing) - 防止灾难性遗忘
        self._freeze_layers(model, freeze_layers)

        # 3. 准备数据
        train_examples = [
            InputExample(texts=[t1, t2]) for t1, t2 in train_pairs
            if len(t1) > 5 and len(t2) > 5  # 简单过滤短文本
        ]

        train_dataloader = DataLoader(
            train_examples, shuffle=True, batch_size=batch_size)

        # 4. 定义 Loss (Contrastive Loss)
        # MultipleNegativesRankingLoss 自动使用 batch 内的其他样本作为负样本
        train_loss = losses.MultipleNegativesRankingLoss(model)

        # 5. 开始训练
        if os.path.exists(output_path):
            self.logger.warning(f"目录已存在，将被覆盖: {output_path}")
            shutil.rmtree(output_path)

        model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=epochs,
            warmup_steps=int(len(train_dataloader) * 0.1),
            show_progress_bar=True,
            output_path=output_path,
            optimizer_params={'lr': 2e-5}
        )

        self.logger.info(f"✅ SBERT 微调完成，模型已保存至: {output_path}")

        # 清理内存
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    def _freeze_layers(self, model: SentenceTransformer, num_layers_to_freeze: int):
        """
        冻结 Transformer 的前 N 层
        """
        auto_model = model._first_module().auto_model

        # 1. 冻结 Embedding 层
        for param in auto_model.embeddings.parameters():
            param.requires_grad = False

        # 2. 冻结 Encoder 层
        if hasattr(auto_model, 'encoder') and hasattr(auto_model.encoder, 'layer'):
            layers = auto_model.encoder.layer
            total_layers = len(layers)

            # 确保不冻结所有层
            freeze_limit = min(num_layers_to_freeze, total_layers - 1)

            for i in range(freeze_limit):
                for param in layers[i].parameters():
                    param.requires_grad = False

            self.logger.info(
                f"🧊 已冻结前 {freeze_limit}/{total_layers} 层 Transformer 参数。")
        else:
            self.logger.warning("⚠️ 无法识别模型结构，未执行层冻结。")
