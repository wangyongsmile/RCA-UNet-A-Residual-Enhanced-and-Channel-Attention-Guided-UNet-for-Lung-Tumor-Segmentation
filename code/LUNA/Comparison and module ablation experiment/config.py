import os
import random
import numpy as np
import torch

# ========== 固定随机种子 ==========
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.backends.cudnn.deterministic = True

# ========== 路径配置 ==========
BASE_PATH = "/home/chenxinyan/LUNA"
os.makedirs(os.path.join(BASE_PATH, "trained_models"), exist_ok=True)
os.makedirs("results", exist_ok=True)

# ========== 训练参数 ==========
BATCH_SIZE = 8
NUM_EPOCHS = 200
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")