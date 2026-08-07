import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt
import random
import warnings
import matplotlib.pyplot as plt
from collections import defaultdict
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts
import cv2

warnings.filterwarnings('ignore')
# 固定所有随机源
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.backends.cudnn.deterministic = True  # 禁用CuDNN的随机优化

# ========== 超参数 ==========
DATA_DIR = "/home/chenxinyan/PythonProject/LungPreprocessed_63/"
BATCH_SIZE = 8
NUM_EPOCHS = 120
LEARNING_RATE = 8e-5
WEIGHT_DECAY = 2e-5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs("results", exist_ok=True)


# ========== 模型架构（保持不变） ==========
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.dropout = nn.Dropout2d(0.4)
        self.extra_dropout = nn.Dropout2d(0.3)

        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.dropout(out)
        out = self.extra_dropout(out)
        out += residual
        out = self.relu(out)
        return out


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class ChannelOnlyAttentionModule(nn.Module):
    def __init__(self, in_channels):
        super(ChannelOnlyAttentionModule, self).__init__()
        self.channel_attention = ChannelAttention(in_channels)

    def forward(self, x):
        ca_out = self.channel_attention(x) * x
        return ca_out


class RCAUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=[64, 128, 256, 512]):
        super(RCAUNet, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.attention_modules = nn.ModuleList()

        for feature in features:
            self.downs.append(ResidualBlock(in_channels, feature))
            in_channels = feature

        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2)
            )
            self.ups.append(ResidualBlock(feature * 2, feature))
            self.attention_modules.append(ChannelOnlyAttentionModule(feature))

        self.bottleneck = ResidualBlock(features[-1], features[-1] * 2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]
            skip_connection = self.attention_modules[idx // 2](skip_connection)

            if x.shape != skip_connection.shape:
                x = F.interpolate(x, size=skip_connection.shape[2:], mode='bilinear', align_corners=False)

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](concat_skip)

        return self.final_conv(x)


# ========== 数据加载类（移除所有数据增强） ==========
class OptimizedLungDataset(Dataset):
    def __init__(self, image_dir, mask_dir, is_training=False):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.is_training = is_training  # 该参数仍保留但不用于增强
        self.image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('_img.npy')])
        self.mask_files = [f.replace('_img.npy', '_label.npy') for f in self.image_files]

        for mask_file in self.mask_files:
            if not os.path.exists(os.path.join(mask_dir, mask_file)):
                raise FileNotFoundError(f"Mask file not found: {os.path.join(mask_dir, mask_file)}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_files[idx])

        image = np.load(img_path)
        mask = np.load(mask_path)

        # 仅保留归一化，删除所有增强操作
        image = self.optimized_normalize(image)

        # 【删除所有训练时的增强代码】
        # 包括翻转、旋转、缩放、模糊、亮度调整等所有数据增强操作

        # 数据格式转换（保持不变）
        image = torch.from_numpy(image).float().unsqueeze(0)
        mask = torch.from_numpy(mask).float().unsqueeze(0)

        return image, mask

    def optimized_normalize(self, image):
        """保持原有的归一化方法不变"""
        p2, p98 = np.percentile(image, [2, 98])
        image = np.clip(image, p2, p98)

        mean = np.mean(image)
        std = np.std(image)
        if std < 1e-8:
            std = 1e-8

        image = (image - mean) / (std + 1e-8)
        return image


# ========== 损失函数 ==========
class StableDiceLoss(nn.Module):
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
    def __init__(self, alpha=0.75, beta=0.25):
        super(OptimizedCombinedLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.dice_loss = StableDiceLoss()
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, inputs, targets):
        dice_loss = self.dice_loss(inputs, targets)
        bce_loss = self.bce_loss(inputs, targets)
        total_loss = self.alpha * dice_loss + self.beta * bce_loss
        return total_loss


# ========== 评估指标计算 ==========
def calculate_metrics(pred, target):
    pred_bin = (pred > 0.5).float()

    tp = torch.sum(pred_bin * target)
    fp = torch.sum(pred_bin * (1 - target))
    tn = torch.sum((1 - pred_bin) * (1 - target))
    fn = torch.sum((1 - pred_bin) * target)

    dice = (2 * tp) / (2 * tp + fp + fn + 1e-8)
    sensitivity = tp / (tp + fn + 1e-8)
    specificity = tn / (tn + fp + 1e-8)
    intersection = torch.sum(pred_bin * target)
    union = torch.sum(pred_bin) + torch.sum(target) - intersection
    iou = intersection / (union + 1e-8)
    precision = tp / (tp + fp + 1e-8)

    try:
        pred_np = pred_bin.squeeze().cpu().numpy().astype(np.uint8)
        target_np = target.squeeze().cpu().numpy().astype(np.uint8)

        if np.sum(pred_np) == 0 and np.sum(target_np) == 0:
            hd95_value = 0.0
            asd_value = 0.0
        elif np.sum(pred_np) > 0 and np.sum(target_np) > 0:
            pred_dist = distance_transform_edt(1 - pred_np)
            target_dist = distance_transform_edt(1 - target_np)
            pred_to_target = pred_dist[target_np > 0]
            target_to_pred = target_dist[pred_np > 0]
            all_distances = np.concatenate([pred_to_target, target_to_pred])
            hd95_value = np.percentile(all_distances, 95) if len(all_distances) > 0 else 0.0
            asd_value = (np.mean(pred_to_target) + np.mean(target_to_pred)) / 2 if len(all_distances) > 0 else 0.0
        else:
            max_dist = np.sqrt(pred_np.shape[0] **2 + pred_np.shape[1]** 2)
            hd95_value = max_dist
            asd_value = max_dist
    except Exception as e:
        hd95_value = 0.0
        asd_value = 0.0

    return dice.item(), sensitivity.item(), specificity.item(), iou.item(), precision.item(), hd95_value, asd_value


# ========== 训练函数（修改结果保存路径和标识） ==========
def train_model():
    if torch.cuda.is_available():
        print(f"使用GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f} GB")
    else:
        print("使用CPU进行训练")

    train_dataset = OptimizedLungDataset(
        image_dir=os.path.join(DATA_DIR, "train"),
        mask_dir=os.path.join(DATA_DIR, "train"),
        is_training=True
    )

    val_dataset = OptimizedLungDataset(
        image_dir=os.path.join(DATA_DIR, "val"),
        mask_dir=os.path.join(DATA_DIR, "val"),
        is_training=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    print(f"训练集样本数: {len(train_dataset)}")
    print(f"验证集样本数: {len(val_dataset)}")
    print(f"批量大小: {BATCH_SIZE}")
    print(f"使用设备: {DEVICE}")

    model = RCAUNet(in_channels=1, out_channels=1).to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数总数: {total_params:,}")

    criterion = OptimizedCombinedLoss(alpha=0.75, beta=0.25)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.999)
    )

    plateau_scheduler = ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=15,
        min_lr=1e-7,
        verbose=True,
        cooldown=5
    )

    cosine_scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=20,
        T_mult=1,
        eta_min=1e-7
    )

    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None

    best_dice = 0.0
    history = defaultdict(list)
    patience_counter = 0
    max_patience = 35
    accumulation_steps = 2
    max_grad_norm = 1.0

    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0.0
        train_dice = 0.0
        train_sensitivity = 0.0
        train_specificity = 0.0
        train_iou = 0.0
        train_precision = 0.0
        train_hd95 = 0.0
        train_asd = 0.0

        train_progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS} [Train]")
        optimizer.zero_grad()

        for i, (images, masks) in enumerate(train_progress):
            images = images.to(DEVICE, non_blocking=True)
            masks = masks.to(DEVICE, non_blocking=True)

            if scaler:
                with torch.cuda.amp.autocast():
                    outputs = model(images)
                    loss = criterion(outputs, masks) / accumulation_steps
            else:
                outputs = model(images)
                loss = criterion(outputs, masks) / accumulation_steps

            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (i + 1) % accumulation_steps == 0:
                if scaler:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

                if scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()

            with torch.no_grad():
                dice, sensitivity, specificity, iou, precision, hd95, asd = calculate_metrics(
                    torch.sigmoid(outputs), masks
                )

            train_loss += loss.item() * accumulation_steps
            train_dice += dice
            train_sensitivity += sensitivity
            train_specificity += specificity
            train_iou += iou
            train_precision += precision
            train_hd95 += hd95
            train_asd += asd

            train_progress.set_postfix({
                "Loss": f"{loss.item() * accumulation_steps:.4f}",
                "Dice": f"{dice:.4f}",
                "Precision": f"{precision:.4f}",
                "IoU": f"{iou:.4f}"
            })

        cosine_scheduler.step()

        train_loss /= len(train_loader)
        train_dice /= len(train_loader)
        train_sensitivity /= len(train_loader)
        train_specificity /= len(train_loader)
        train_iou /= len(train_loader)
        train_precision /= len(train_loader)
        train_hd95 /= len(train_loader)
        train_asd /= len(train_loader)

        model.eval()
        val_loss = 0.0
        val_dice = 0.0
        val_sensitivity = 0.0
        val_specificity = 0.0
        val_iou = 0.0
        val_precision = 0.0
        val_hd95 = 0.0
        val_asd = 0.0

        with torch.no_grad():
            val_progress = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS} [Val]")
            for images, masks in val_progress:
                images = images.to(DEVICE, non_blocking=True)
                masks = masks.to(DEVICE, non_blocking=True)

                if scaler:
                    with torch.cuda.amp.autocast():
                        outputs = model(images)
                        loss = criterion(outputs, masks)
                else:
                    outputs = model(images)
                    loss = criterion(outputs, masks)

                dice, sensitivity, specificity, iou, precision, hd95, asd = calculate_metrics(
                    torch.sigmoid(outputs), masks
                )

                val_loss += loss.item()
                val_dice += dice
                val_sensitivity += sensitivity
                val_specificity += specificity
                val_iou += iou
                val_precision += precision
                val_hd95 += hd95
                val_asd += asd

                val_progress.set_postfix({
                    "Loss": f"{loss.item():.4f}",
                    "Dice": f"{dice:.4f}",
                    "Precision": f"{precision:.4f}",
                    "IoU": f"{iou:.4f}"
                })

        val_loss /= len(val_loader)
        val_dice /= len(val_loader)
        val_sensitivity /= len(val_loader)
        val_specificity /= len(val_loader)
        val_iou /= len(val_loader)
        val_precision /= len(val_loader)
        val_hd95 /= len(val_loader)
        val_asd /= len(val_loader)

        plateau_scheduler.step(val_dice)

        overfit_ratio = train_dice / val_dice if val_dice > 1e-8 else 0.0

        # 【保存路径添加无数据增强标识】
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_dice': best_dice,
            }, "results/best_model_no_augmentation.pth")
            print(f"New best model (无数据增强) saved with Dice: {val_dice:.4f}, IoU: {val_iou:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= max_patience:
            print(f"Early stopping triggered after {epoch + 1} epochs")
            print(f"过拟合比率: {overfit_ratio:.2f}")
            break

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_dice'].append(train_dice)
        history['val_dice'].append(val_dice)
        history['train_iou'].append(train_iou)
        history['val_iou'].append(val_iou)
        history['train_sensitivity'].append(train_sensitivity)
        history['val_sensitivity'].append(val_sensitivity)
        history['train_specificity'].append(train_specificity)
        history['val_specificity'].append(val_specificity)
        history['train_precision'].append(train_precision)
        history['val_precision'].append(val_precision)
        history['train_hd'].append(train_hd95)
        history['val_hd'].append(val_hd95)
        history['train_asd'].append(train_asd)
        history['val_asd'].append(val_asd)
        history['overfit_ratio'].append(overfit_ratio)
        history['learning_rate'].append(optimizer.param_groups[0]['lr'])

        # 【日志添加无数据增强标识】
        print(f"Epoch {epoch + 1}/{NUM_EPOCHS} (无数据增强)")
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        print(f"Train Dice: {train_dice:.4f}, Val Dice: {val_dice:.4f}")
        print(f"Train IoU: {train_iou:.4f}, Val IoU: {val_iou:.4f}")
        print(f"过拟合比率: {overfit_ratio:.2f}")
        print(f"Train Sensitivity: {train_sensitivity:.4f}, Val Sensitivity: {val_sensitivity:.4f}")
        print(f"Train Specificity: {train_specificity:.4f}, Val Specificity: {val_specificity:.4f}")
        print(f"Train Precision: {train_precision:.4f}, Val Precision: {val_precision:.4f}")
        print(f"Train 95% HD: {train_hd95:.2f}, Val 95% HD: {val_hd95:.2f}")
        print(f"Train ASD: {train_asd:.2f}, Val ASD: {val_asd:.2f}")
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Current Learning Rate: {current_lr:.2e}")

        if torch.cuda.is_available():
            print(
                f"GPU Memory: {torch.cuda.memory_allocated() / 1024 ** 3:.2f}GB / {torch.cuda.memory_reserved() / 1024 ** 3:.2f}GB")

        print("-" * 60)

    plot_metrics(history)
    print("Training completed (无数据增强)!")
    print(f"最佳验证Dice (无数据增强): {best_dice:.4f}")
    return best_dice


# ========== 绘图函数 ==========
def plot_metrics(history):
    plt.figure(figsize=(24, 18))

    metrics = [
        ('Loss', ['train_loss', 'val_loss']),
        ('Dice Coefficient', ['train_dice', 'val_dice']),
        ('IoU', ['train_iou', 'val_iou']),
        ('Sensitivity', ['train_sensitivity', 'val_sensitivity']),
        ('Specificity', ['train_specificity', 'val_specificity']),
        ('Precision', ['train_precision', 'val_precision']),
        ('95% Hausdorff Distance', ['train_hd', 'val_hd']),
        ('Average Surface Distance', ['train_asd', 'val_asd']),
        ('Overfitting Ratio', ['overfit_ratio']),
        ('Learning Rate', ['learning_rate'])
    ]

    for i, (title, keys) in enumerate(metrics):
        plt.subplot(4, 3, i + 1)
        for key in keys:
            if key in history and history[key]:
                if key == 'overfit_ratio':
                    plt.plot(history[key], label=key, color='red', linewidth=2)
                    plt.axhline(y=1.15, color='r', linestyle='--', alpha=0.7, label='Good Threshold')
                    plt.axhline(y=1.3, color='orange', linestyle='--', alpha=0.7, label='Warning Threshold')
                elif key == 'learning_rate':
                    plt.semilogy(history[key], label=key, color='purple', linewidth=2)
                else:
                    plt.plot(history[key], label=key, linewidth=2)
        plt.title(title)
        plt.xlabel('Epoch (无数据增强)')  # 【添加无数据增强标识】
        plt.ylabel(title)
        plt.legend()
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    # 【保存路径添加无数据增强标识】
    plt.savefig("results/no_augmentation.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Metrics plot (无数据增强) saved to results/training_metrics_no_augmentation.png")


if __name__ == "__main__":
    best_dice = train_model()
    print(f"最终最佳Dice系数 (无数据增强): {best_dice:.4f}")
