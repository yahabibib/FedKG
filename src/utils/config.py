# src/utils/config.py
import os
import torch
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Config:
    """
    Fed-LLM-SBERT 全局配置类
    使用 Dataclass 管理所有参数，支持实例化时覆盖，彻底告别 global variables。
    """

    # ==========================
    # 1. 基础路径配置
    # ==========================
    project_root: str = field(default_factory=lambda: os.getcwd())
    dataset_name: str = "dbp15k"
    # 相对路径，不再写死绝对路径 /Users/...
    relative_data_path: str = "data/dbp15k/zh_en"

    # ==========================
    # 2. 模型架构参数
    # ==========================
    model_arch: str = "decoupled"  # 'gcn', 'decoupled', 'projection'

    # SBERT (LLM 增强后)
    # 建议：这里指向 Stage 1 产出的模型路径
    sbert_model_path: str = "./output/fine_tuned_models/exp4_finetuned"
    bert_dim: int = 768

    # GCN / Graph Model
    gcn_dim: int = 300
    gcn_hidden: int = 600
    gcn_output: int = 768   # 必须与 BERT_DIM 对应
    gcn_layers: int = 2
    gcn_dropout: float = 0.5

    # TransE (如果用的话)
    transe_dim: int = 300
    transe_margin: float = 1.0

    # ==========================
    # 3. 训练与联邦参数
    # ==========================
    use_aggregation: bool = True  # 是否启用联邦聚合
    fl_rounds: int = 100
    fl_local_epochs: int = 5
    fl_batch_size: int = 512
    fl_lr: float = 5e-4
    fl_margin: float = 0.4

    # ==========================
    # 4. LLM 增强 (Stage 1)
    # ==========================
    llm_model_id: str = "Qwen/Qwen2.5-1.5B-Instruct"
    # Stage 1 的中间结果保存路径
    polish_output_dir: str = "data/polished_data"

    # SBERT 推理专用 Batch Size (Mac 建议 32 或 64)
    sbert_batch_size: int = 32

    # ==========================
    # 5. 评估参数
    # ==========================
    eval_k_values: List[int] = field(default_factory=lambda: [1, 10, 50])
    eval_fusion_alpha: float = 0.42  # 融合权重

    # ==========================
    # 6. 动态属性 (Properties)
    # ==========================
    @property
    def data_dir(self) -> str:
        """返回数据的绝对路径"""
        return os.path.join(self.project_root, self.relative_data_path)

    @property
    def device(self) -> torch.device:
        """自动检测最佳设备"""
        if torch.backends.mps.is_available():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")

    def display(self):
        """打印当前配置"""
        print(f"{'='*40}")
        print(f"🔧 Configuration ({self.dataset_name})")
        print(f"   📂 Data Dir: {self.data_dir}")
        print(f"   🖥️ Device: {self.device}")
        print(f"   🕸️ Arch: {self.model_arch}")
        print(f"   🤖 SBERT: {self.sbert_model_path}")
        print(f"{'='*40}")
