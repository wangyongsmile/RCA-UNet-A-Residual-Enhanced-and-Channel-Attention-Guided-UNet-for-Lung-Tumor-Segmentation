import numpy as np


class GradualWarmupScheduler:
    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr_ratio=0.01):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr_ratio = min_lr_ratio
        self.current_epoch = 0
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]

    def step(self):
        self.current_epoch += 1
        lr = self.get_lr()
        for param_group, new_lr in zip(self.optimizer.param_groups, lr):
            param_group['lr'] = new_lr

    def get_lr(self):
        if self.current_epoch <= self.warmup_epochs:
            return [base_lr * (self.current_epoch / self.warmup_epochs)
                    for base_lr in self.base_lrs]
        else:
            progress = (self.current_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            cosine_decay = 0.5 * (1 + np.cos(np.pi * progress))
            return [base_lr * (self.min_lr_ratio + (1 - self.min_lr_ratio) * cosine_decay)
                    for base_lr in self.base_lrs]