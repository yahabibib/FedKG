import torch
import torch.nn as nn

class ProjectionModel(nn.Module):
    """ 
    MLP 投影模型 
    用于将 TransE 映射到 SBERT 空间
    """
    def __init__(self, input_dim, output_dim):
        super().__init__()
        
        # 动态隐藏层维度
        hidden_dim = max(512, (input_dim + output_dim) // 2)
        
        print(f"    [Model Init] MLP: {input_dim} -> {hidden_dim} -> {output_dim}")

        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.projection(x)