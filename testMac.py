import torch

if torch.backends.mps.is_available():
    print("太好了！MPS (Apple GPU) 加速可用！")
    device = torch.device("mps")
else:
    print("MPS 不可用，将使用 CPU。")
    device = torch.device("cpu")

print(f"当前使用的设备: {device}")