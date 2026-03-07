import torch
import random
import numpy as np

# 固定所有随机源
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.backends.cudnn.deterministic = True

# 默认配置参数
DATA_DIR = "/home/chenxinyan/PythonProject/LungPreprocessed_63/"
BATCH_SIZE = 8
NUM_EPOCHS = 120
LEARNING_RATE = 8e-5
WEIGHT_DECAY = 2e-5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS_DIR = "results"


def update_config(**kwargs):
    """更新配置参数"""
    global DATA_DIR, BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE, WEIGHT_DECAY, DEVICE, RESULTS_DIR

    for key, value in kwargs.items():
        if value is not None:
            if key == 'data_dir':
                DATA_DIR = value
            elif key == 'batch_size':
                BATCH_SIZE = value
            elif key == 'num_epochs':
                NUM_EPOCHS = value
            elif key == 'learning_rate':
                LEARNING_RATE = value
            elif key == 'weight_decay':
                WEIGHT_DECAY = value
            elif key == 'device':
                DEVICE = torch.device(value)
            elif key == 'results_dir':
                RESULTS_DIR = value