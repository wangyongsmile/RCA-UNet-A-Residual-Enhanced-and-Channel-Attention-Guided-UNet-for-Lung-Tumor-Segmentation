import numpy as np


def split_dataset(all_seriesuids, train_ratio=0.7, val_ratio=0.15):
    np.random.seed(42)
    np.random.shuffle(all_seriesuids)
    total = len(all_seriesuids)
    train_size = int(total * train_ratio)
    val_size = int(total * val_ratio)
    train_series = all_seriesuids[:train_size]
    val_series = all_seriesuids[train_size:train_size + val_size]
    test_series = all_seriesuids[train_size + val_size:]
    return train_series, val_series, test_series