import torch
import torch.nn as nn


class StableDiceLoss(nn.Module):
    """稳定的Dice损失"""

    def __init__(self, smooth=1.0):
        super(StableDiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)
        return 1 - dice


class OptimizedCombinedLoss(nn.Module):
    """优化的组合损失"""

    def __init__(self, alpha=0.75, beta=0.25):
        super(OptimizedCombinedLoss, self).__init__()
        self.alpha = alpha  # Dice损失权重
        self.beta = beta  # BCE损失权重
        self.dice_loss = StableDiceLoss()
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, inputs, targets):
        dice_loss = self.dice_loss(inputs, targets)
        bce_loss = self.bce_loss(inputs, targets)

        total_loss = self.alpha * dice_loss + self.beta * bce_loss
        return total_loss