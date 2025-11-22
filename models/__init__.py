# 📄 AiStudy/models/__init__.py
from .transe import TransE
from .projection import ProjectionModel
from .gcn import GCN
from .decoupled import DecoupledModel  # 新增

# 模型注册表
MODEL_REGISTRY = {
    'transe': TransE,
    'projection': ProjectionModel,
    'gcn': GCN,
    'decoupled': DecoupledModel  # 新增
}


def get_model_class(model_name):
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Model '{model_name}' not found in registry.")
    return MODEL_REGISTRY[model_name]
