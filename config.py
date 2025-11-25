# 📄 config.py
import torch

# =========================================================
# 🎛️ 【核心控制台】
# =========================================================
CURRENT_DATASET_NAME = 'dbp15k'  # 或 'demo'

# 🧠 模型架构选择
# 'gcn' -> 全聚合 GCN
# 'decoupled' -> 解耦联邦 (Private GCN + Shared MLP)
MODEL_ARCH = 'decoupled'

# 🔬 实验模式: True=联邦聚合, False=孤立训练
USE_AGGREGATION = True

# =========================================================
# ⚙️ 硬件与通用配置
# =========================================================


def get_best_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


DEVICE = get_best_device()

BERT_MODEL_NAME = 'paraphrase-multilingual-mpnet-base-v2'
BERT_DIM = 768
BERT_BATCH_SIZE = 32

# =========================================================
# 📚 数据集预设
# =========================================================
DATASET_CONFIGS = {
    'demo': {
        # 请修改为您实际的 demo 路径
        'base_path': "/Users/yihanbin/Documents/科研/知识图谱/代码/KGE/FedKG/data/demo/",
        'transe_dim': 64, 'transe_epochs': 500, 'transe_batch': 8,
        'gcn_dim': 64, 'gcn_hidden': 128, 'gcn_layers': 2,
        'fl_rounds': 50, 'fl_local_epochs': 10, 'fl_batch': 8, 'fl_lr': 1e-3, 'fl_margin': 0.5,
        'eval_k': [1, 5, 10]
    },
    'dbp15k': {
        # 请修改为您实际的 dbp15k 路径
        'base_path': "/Users/yihanbin/Documents/科研/知识图谱/代码/KGE/FedKG/data/dbp15k/zh_en/",
        'transe_dim': 300, 'transe_epochs': 1000, 'transe_batch': 2048,
        'gcn_dim': 300, 'gcn_hidden': 300, 'gcn_layers': 2,
        'fl_rounds': 100, 'fl_local_epochs': 5, 'fl_batch': 512, 'fl_lr': 1e-4,
        'fl_margin': 0.4,
        'gcn_dropout': 0.2,
        'eval_k': [1, 10, 50]
    }
}

if CURRENT_DATASET_NAME not in DATASET_CONFIGS:
    raise ValueError(f"数据集 '{CURRENT_DATASET_NAME}' 未定义！")

_cfg = DATASET_CONFIGS[CURRENT_DATASET_NAME]
BASE_PATH = _cfg['base_path']

TRANSE_DIM = _cfg['transe_dim']
TRANSE_EPOCHS = _cfg['transe_epochs']
TRANSE_BATCH_SIZE = _cfg['transe_batch']
TRANSE_LR = 0.001
TRANSE_MARGIN = 1.0
TRANSE_P_NORM = 2

GCN_DIM = _cfg['gcn_dim']
GCN_HIDDEN = _cfg['gcn_hidden']
GCN_DROPOUT = _cfg.get('gcn_dropout', 0.3)
GCN_LAYERS = _cfg['gcn_layers']

FL_ROUNDS = _cfg['fl_rounds']
FL_LOCAL_EPOCHS = _cfg['fl_local_epochs']
FL_BATCH_SIZE = _cfg['fl_batch']
FL_LR = _cfg['fl_lr']
FL_MARGIN = _cfg['fl_margin']

# --- 联邦原型对比学习 (Prototype Contrastive Learning) ---
USE_PROTOTYPES = True
PROTO_NUM = 100
PROTO_LAMBDA = 0.1
PROTO_TEMPERATURE = 0.1

EVAL_K_VALUES = _cfg['eval_k']

# --- 🔥 [新增] 融合推理配置 ---
# 0.42 是实验得出的最佳值 (42% GCN + 58% SBERT)
EVAL_FUSION_ALPHA = 0.42

if MODEL_ARCH == 'gcn':
    MODEL_INFO = f"GCN (Dim={GCN_DIM}, Hidden={GCN_HIDDEN}, Drop={GCN_DROPOUT})"
elif MODEL_ARCH == 'projection':
    MODEL_INFO = f"TransE (Dim={TRANSE_DIM}) + MLP Projection"
else:
    MODEL_INFO = "Decoupled (GCN+MLP)"

print(f"⚡️ 配置加载完毕: [{CURRENT_DATASET_NAME}]")
print(f"   🕸️ 架构: {MODEL_INFO}")
print(f"   🎲 模式: {'联邦聚合' if USE_AGGREGATION else '孤立训练'}")
print(f"   ⚖️ 融合 Alpha: {EVAL_FUSION_ALPHA}")
print("-" * 50)
