import torch
import torch.nn as nn
import torch.nn.functional as F


class ImprovedFocalDiceLoss(nn.Module):
    def __init__(self, alpha=0.8, gamma=2.0, dice_weight=0.6, boundary_weight=0.2, smooth=1e-6):
        super(ImprovedFocalDiceLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.dice_weight = dice_weight
        self.boundary_weight = boundary_weight
        self.smooth = smooth

    def focal_loss(self, inputs, targets):
        inputs = torch.clamp(inputs, 1e-7, 1.0 - 1e-7)
        bce = F.binary_cross_entropy(inputs, targets, reduction='none')
        p_t = torch.exp(-bce)
        focal_loss = self.alpha * (1 - p_t) ** self.gamma * bce
        return focal_loss.mean()

    def dice_loss(self, inputs, targets):
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)
        return 1 - dice

    def boundary_aware_loss(self, inputs, targets):
        with torch.no_grad():
            kernel = torch.ones(1, 1, 3, 3).to(inputs.device)
            targets_eroded = F.conv2d(targets, kernel, padding=1)
            targets_dilated = F.conv2d(targets, kernel, padding=1)
            boundary_mask = (targets_dilated - targets_eroded).abs()
            boundary_mask = (boundary_mask > 0.1).float()
            weight_map = 1.0 + 4.0 * boundary_mask
        inputs = torch.clamp(inputs, 1e-7, 1.0 - 1e-7)
        bce = F.binary_cross_entropy(inputs, targets, reduction='none')
        weighted_bce = (bce * weight_map).mean()
        return weighted_bce

    def forward(self, inputs, targets):
        inputs = torch.clamp(inputs, 1e-7, 1.0 - 1e-7)
        focal = self.focal_loss(inputs, targets)
        dice = self.dice_loss(inputs, targets)
        boundary = self.boundary_aware_loss(inputs, targets)
        return (1 - self.dice_weight - self.boundary_weight) * focal + \
            self.dice_weight * dice + \
            self.boundary_weight * boundary