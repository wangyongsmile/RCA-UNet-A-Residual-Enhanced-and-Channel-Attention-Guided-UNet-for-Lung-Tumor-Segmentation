import os
import numpy as np
import pandas as pd
import SimpleITK as sitk
from skimage.transform import resize
import albumentations as A
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt
import random
import warnings
from collections import defaultdict

warnings.filterwarnings('ignore')

# ========== 训练参数 ==========
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.backends.cudnn.deterministic = True

BASE_PATH = "/home/chenxinyan/LUNA"
BATCH_SIZE = 8
NUM_EPOCHS = 200
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(os.path.join(BASE_PATH, "trained_models"), exist_ok=True)
os.makedirs("results", exist_ok=True)


class LUNADataLoader:
    def __init__(self, base_path=BASE_PATH):
        self.base_path = base_path
        self.annotations = pd.read_csv(os.path.join(base_path, "annotations.csv"))
        self.candidates = pd.read_csv(os.path.join(base_path, "candidates.csv"))
        self.all_seriesuids = self.annotations["seriesuid"].unique().tolist()

    def load_ct_scan(self, seriesuid):
        for subset_idx in range(10):
            mhd_path = os.path.join(self.base_path, f"subset{subset_idx}", f"{seriesuid}.mhd")
            if os.path.exists(mhd_path):
                itk_image = sitk.ReadImage(mhd_path)
                ct_array = sitk.GetArrayFromImage(itk_image)
                spacing = np.array(itk_image.GetSpacing())
                origin = np.array(itk_image.GetOrigin())
                return ct_array, spacing, origin
        raise FileNotFoundError(f"未找到seriesuid {seriesuid} 对应的MHD文件")

    @staticmethod
    def world2voxel(world_coord, origin, spacing):
        voxel_coord = np.round((world_coord - origin) / spacing).astype(int)
        return voxel_coord

    @staticmethod
    def voxel2world(voxel_coord, origin, spacing):
        world_coord = origin + voxel_coord * spacing
        return world_coord


class SliceExtractor:
    def __init__(self, margin=1):
        self.margin = margin

    def extract_slices_with_masks(self, ct_array, spacing, annotations_df):
        slices_with_masks = []
        depth, height, width = ct_array.shape

        for _, row in annotations_df.iterrows():
            world_coord = np.array([row["coordX"], row["coordY"], row["coordZ"]])
            if isinstance(row["origin"], str):
                origin = np.array([float(x) for x in row["origin"][1:-1].split()])
            else:
                origin = row["origin"]

            i, j, k = LUNADataLoader.world2voxel(world_coord, origin, spacing)
            radius = row["diameter_mm"] / (2 * spacing[0])

            i = np.clip(i, 0, width - 1)
            j = np.clip(j, 0, height - 1)
            k = np.clip(k, 0, depth - 1)
            radius = max(2, int(np.round(radius)))

            for dz in range(-self.margin, self.margin + 1):
                slice_idx = k + dz
                if 0 <= slice_idx < depth:
                    ct_slice = ct_array[slice_idx]
                    mask = np.zeros((height, width), dtype=np.uint8)

                    y_coords, x_coords = np.ogrid[:height, :width]
                    distance = np.sqrt((x_coords - i) ** 2 + (y_coords - j) ** 2)
                    mask[distance <= radius] = 1

                    if np.sum(mask) > 0:
                        slices_with_masks.append((slice_idx, ct_slice, mask))

        return slices_with_masks


# ========== 数据增强 ==========
class EnhancedPreprocessor:
    def __init__(self, target_size=(256, 256)):
        self.target_size = target_size


        self.train_transform = A.Compose([
            A.Resize(height=target_size[0], width=target_size[1]),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.05,
                scale_limit=0.1,
                rotate_limit=10,
                border_mode=0,
                p=0.5
            ),
            A.OneOf([
                A.RandomBrightnessContrast(
                    brightness_limit=0.1,
                    contrast_limit=0.1,
                    p=0.5
                ),
                A.GaussNoise(var_limit=(5.0, 10.0), p=0.3),
                A.MotionBlur(blur_limit=3, p=0.2),
            ], p=0.5),
            A.CoarseDropout(
                max_holes=8,
                max_height=16,
                max_width=16,
                min_holes=1,
                min_height=8,
                min_width=8,
                fill_value=0,
                p=0.3
            ),
        ])

        self.val_transform = A.Compose([
            A.Resize(height=target_size[0], width=target_size[1])
        ])

    @staticmethod
    def normalize_ct(ct_slice):
        window_center = -600
        window_width = 1500

        window_min = window_center - window_width // 2
        window_max = window_center + window_width // 2

        ct_slice = np.clip(ct_slice, window_min, window_max)
        ct_slice = (ct_slice - window_min) / window_width

        return ct_slice.astype(np.float32)

    def process(self, ct_slice, mask, is_train=True):
        ct_slice = self.normalize_ct(ct_slice)

        ct_slice_uint8 = (ct_slice * 255).astype(np.uint8)
        mask_uint8 = (mask * 255).astype(np.uint8)

        if is_train:
            transformed = self.train_transform(image=ct_slice_uint8, mask=mask_uint8)
        else:
            transformed = self.val_transform(image=ct_slice_uint8, mask=mask_uint8)

        ct_processed = transformed["image"].astype(np.float32) / 255.0
        mask_processed = (transformed["mask"] > 127).astype(np.float32)

        ct_processed = np.expand_dims(ct_processed, axis=0)
        mask_processed = np.expand_dims(mask_processed, axis=0)

        return ct_processed, mask_processed


class AnalyzedLUNADataset(Dataset):
    def __init__(self, seriesuids, data_loader, slice_extractor, preprocessor, is_train=True):
        self.seriesuids = seriesuids
        self.data_loader = data_loader
        self.slice_extractor = slice_extractor
        self.preprocessor = preprocessor
        self.is_train = is_train

        self.raw_samples = self._preload_raw_samples()
        self.sample_weights = self._calculate_sample_weights()
        self._analyze_dataset()

    def _preload_raw_samples(self):
        raw_samples = []
        print(f"加载原始CT切片数据...")

        for idx, seriesuid in enumerate(tqdm(self.seriesuids, desc="Loading raw CT scans")):
            annotations_subset = self.data_loader.annotations[
                self.data_loader.annotations["seriesuid"] == seriesuid
                ].copy()

            if len(annotations_subset) == 0:
                continue

            try:
                ct_array, spacing, origin = self.data_loader.load_ct_scan(seriesuid)
            except FileNotFoundError:
                continue

            annotations_subset["origin"] = [origin] * len(annotations_subset)

            slices_with_masks = self.slice_extractor.extract_slices_with_masks(
                ct_array, spacing, annotations_subset
            )

            if len(slices_with_masks) == 0:
                continue

            for slice_idx, ct_slice, mask in slices_with_masks:
                raw_samples.append((ct_slice, mask))

        print(f"原始样本加载完成：{len(raw_samples)}个样本（{'训练集' if self.is_train else '验证/测试集'}）")
        return raw_samples

    def _calculate_sample_weights(self):
        weights = []
        for ct_slice, mask in self.raw_samples:
            positive_ratio = np.sum(mask > 0.5) / mask.size

            # 更平衡的权重分配
            if positive_ratio > 0.01:
                weight = 5.0
            elif positive_ratio > 0.005:
                weight = 3.0
            elif positive_ratio > 0.001:
                weight = 2.0
            elif positive_ratio > 0.0005:
                weight = 1.5
            else:
                weight = 1.0
            weights.append(weight)

        weights = np.array(weights)
        weights = weights / np.mean(weights)
        return weights.tolist()

    def _analyze_dataset(self):
        positive_pixels = 0
        total_pixels = 0
        sample_positive_ratios = []

        for ct_slice, mask in self.raw_samples:
            positive_in_sample = np.sum(mask > 0.5)
            total_in_sample = mask.size
            positive_pixels += positive_in_sample
            total_pixels += total_in_sample
            sample_positive_ratios.append(positive_in_sample / total_in_sample if total_in_sample > 0 else 0)

        positive_ratio = positive_pixels / total_pixels if total_pixels > 0 else 0

        print(f"数据集分析 - {len(self.raw_samples)}个样本:")
        print(f"  正样本比例: {positive_ratio:.6f}")
        print(f"  正样本像素数: {positive_pixels:,}")
        print(f"  总像素数: {total_pixels:,}")
        print(f"  样本正样本比例范围: {min(sample_positive_ratios):.6f} - {max(sample_positive_ratios):.6f}")
        print(f"  平均样本正样本比例: {np.mean(sample_positive_ratios):.6f}")

        weight_counts = {}
        for weight in self.sample_weights:
            weight_key = f"{weight:.1f}"
            weight_counts[weight_key] = weight_counts.get(weight_key, 0) + 1

        print("  样本权重分布:")
        for weight, count in sorted(weight_counts.items()):
            print(f"    权重 {weight}: {count}个样本 ({count / len(self.sample_weights) * 100:.1f}%)")

    def __len__(self):
        return len(self.raw_samples)

    def __getitem__(self, idx):
        ct_slice, mask = self.raw_samples[idx]

        ct_processed, mask_processed = self.preprocessor.process(
            ct_slice, mask, is_train=self.is_train
        )

        return torch.tensor(ct_processed, dtype=torch.float32), torch.tensor(mask_processed, dtype=torch.float32)


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


# ========== 损失函数==========
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
        # 计算边界权重图
        with torch.no_grad():
            kernel = torch.ones(1, 1, 3, 3).to(inputs.device)
            targets_eroded = F.conv2d(targets, kernel, padding=1)
            targets_dilated = F.conv2d(targets, kernel, padding=1)

            # 边界区域为膨胀后与腐蚀后的差异
            boundary_mask = (targets_dilated - targets_eroded).abs()
            boundary_mask = (boundary_mask > 0.1).float()

            # 给边界区域权重
            weight_map = 1.0 + 4.0 * boundary_mask

        bce = F.binary_cross_entropy(inputs, targets, reduction='none')
        weighted_bce = (bce * weight_map).mean()
        return weighted_bce

    def forward(self, inputs, targets):
        focal = self.focal_loss(inputs, targets)
        dice = self.dice_loss(inputs, targets)
        boundary = self.boundary_aware_loss(inputs, targets)

        return (1 - self.dice_weight - self.boundary_weight) * focal + \
            self.dice_weight * dice + \
            self.boundary_weight * boundary


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.dropout = nn.Dropout2d(0.3)

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

        return torch.sigmoid(self.final_conv(x))


# ========== 训练策略 ==========
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
            # 线性热身
            return [base_lr * (self.current_epoch / self.warmup_epochs)
                    for base_lr in self.base_lrs]
        else:
            # 余弦退火
            progress = (self.current_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            cosine_decay = 0.5 * (1 + np.cos(np.pi * progress))
            return [base_lr * (self.min_lr_ratio + (1 - self.min_lr_ratio) * cosine_decay)
                    for base_lr in self.base_lrs]


class EnhancedEarlyStopping:
    def __init__(self, patience=30, min_delta=1e-4, min_epochs=50, warmup_epochs=20):
        self.patience = patience
        self.min_delta = min_delta
        self.min_epochs = min_epochs
        self.warmup_epochs = warmup_epochs
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_score, current_epoch):
        if current_epoch < self.warmup_epochs:
            return False

        if current_epoch < self.min_epochs:
            return False

        if self.best_score is None:
            self.best_score = val_score
            return False

        if val_score > self.best_score + self.min_delta:
            self.best_score = val_score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                print(f"早停触发！最佳Dice: {self.best_score:.4f}")

        return self.early_stop


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
            max_dist = np.sqrt(pred_np.shape[0] ** 2 + pred_np.shape[1] ** 2)
            hd95_value = max_dist
            asd_value = max_dist
    except Exception as e:
        hd95_value = 0.0
        asd_value = 0.0

    return dice.item(), sensitivity.item(), specificity.item(), iou.item(), precision.item(), hd95_value, asd_value


def train_model():
    if torch.cuda.is_available():
        print(f"使用GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f} GB")
    else:
        print("使用CPU进行训练")

    # 数据加载器创建
    data_loader = LUNADataLoader(base_path=BASE_PATH)
    slice_extractor = SliceExtractor(margin=1)
    preprocessor = EnhancedPreprocessor(target_size=(256, 256))

    train_series, val_series, test_series = split_dataset(data_loader.all_seriesuids)

    train_dataset = AnalyzedLUNADataset(
        train_series, data_loader, slice_extractor, preprocessor, is_train=True
    )
    val_dataset = AnalyzedLUNADataset(
        val_series, data_loader, slice_extractor, preprocessor, is_train=False
    )
    test_dataset = AnalyzedLUNADataset(
        test_series, data_loader, slice_extractor, preprocessor, is_train=False
    )

    print(f"训练集样本数: {len(train_dataset)}")
    print(f"验证集样本数: {len(val_dataset)}")
    print(f"测试集样本数: {len(test_dataset)}")
    print(f"批量大小: {BATCH_SIZE}")
    print(f"使用设备: {DEVICE}")

    # 使用加权采样器处理类别不平衡
    train_weights = torch.DoubleTensor(train_dataset.sample_weights)
    sampler = torch.utils.data.WeightedRandomSampler(train_weights, len(train_weights))

    # 数据加载器
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler,
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=4, pin_memory=True)

    model = RCAUNet(in_channels=1, out_channels=1).to(DEVICE)

    # 打印模型参数数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数总数: {total_params:,}")

    # ========== 使用损失函数 ==========
    criterion = ImprovedFocalDiceLoss(alpha=0.8, gamma=2.0, dice_weight=0.6, boundary_weight=0.2)

    # 优化器
    optimizer = optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.999)
    )

    # ========== 仅使用Cosine Annealing ==========
    warmup_epochs = 10
    scheduler = GradualWarmupScheduler(optimizer, warmup_epochs, NUM_EPOCHS, min_lr_ratio=0.01)
    # 添加CosineAnnealingLR，在热身阶段后使用
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS - warmup_epochs, eta_min=1e-7)

    best_dice = 0.0
    history = defaultdict(list)

    # 早停策略
    early_stopping = EnhancedEarlyStopping(patience=30, min_delta=1e-4, min_epochs=50, warmup_epochs=20)

    # 梯度裁剪
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

        for i, (images, masks) in enumerate(train_progress):
            images = images.to(DEVICE, non_blocking=True)
            masks = masks.to(DEVICE, non_blocking=True)

            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, masks)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

            # 计算所有指标
            with torch.no_grad():
                dice, sensitivity, specificity, iou, precision, hd95, asd = calculate_metrics(outputs, masks)

            train_loss += loss.item()
            train_dice += dice
            train_sensitivity += sensitivity
            train_specificity += specificity
            train_iou += iou
            train_precision += precision
            train_hd95 += hd95
            train_asd += asd

            train_progress.set_postfix({
                "Loss": f"{loss.item():.4f}",
                "Dice": f"{dice:.4f}",
                "Sens": f"{sensitivity:.4f}",
                "Prec": f"{precision:.4f}"
            })

        # ========== 学习率调度策略 ==========
        scheduler.step()
        if epoch >= warmup_epochs:
            cosine_scheduler.step()

        # 训练指标平均
        train_loss /= len(train_loader)
        train_dice /= len(train_loader)
        train_sensitivity /= len(train_loader)
        train_specificity /= len(train_loader)
        train_iou /= len(train_loader)
        train_precision /= len(train_loader)
        train_hd95 /= len(train_loader)
        train_asd /= len(train_loader)

        # 验证阶段
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

                outputs = model(images)
                loss = criterion(outputs, masks)

                dice, sensitivity, specificity, iou, precision, hd95, asd = calculate_metrics(
                    outputs, masks
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
                    "Sens": f"{sensitivity:.4f}",
                    "Prec": f"{precision:.4f}"
                })

        # 验证指标平均
        val_loss /= len(val_loader)
        val_dice /= len(val_loader)
        val_sensitivity /= len(val_loader)
        val_specificity /= len(val_loader)
        val_iou /= len(val_loader)
        val_precision /= len(val_loader)
        val_hd95 /= len(val_loader)
        val_asd /= len(val_loader)

        # 计算过拟合比率
        overfit_ratio = train_dice / val_dice if val_dice > 1e-8 else 0

        # 保存最佳模型
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_dice': best_dice,
            }, os.path.join(BASE_PATH, "trained_models", "best_rca_unet.pth"))
            print(f"New best model saved with Dice: {val_dice:.4f}, IoU: {val_iou:.4f}")

        # 早停检查
        if early_stopping(val_dice, epoch + 1):
            print(f"Early stopping triggered at epoch {epoch + 1}")
            break

        # 记录历史
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

        # 每5轮打印详细日志
        if (epoch + 1) % 5 == 0 or epoch < 10:
            print(f"Epoch {epoch + 1}/{NUM_EPOCHS}")
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

            # 显示GPU内存使用情况
            if torch.cuda.is_available():
                print(f"GPU Memory: {torch.cuda.memory_allocated() / 1024 ** 3:.2f}GB")

            print("-" * 60)

    plot_metrics(history)
    print("Training completed!")
    print(f"最佳验证Dice: {best_dice:.4f}")

    # 测试最佳模型
    test_model(model, test_loader, DEVICE, os.path.join(BASE_PATH, "trained_models", "best_rca_unet.pth"))

    return best_dice


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
        plt.xlabel('Epoch')
        plt.ylabel(title)
        plt.legend()
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("results/training_metrics_rca_unet.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Metrics plot saved to results/training_metrics_rca_unet.png")


def test_model(model, loader, device, model_path):
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    metrics_sum = {
        "Dice": 0.0,
        "IoU": 0.0,
        "Sensitivity": 0.0,
        "Specificity": 0.0,
        "Precision": 0.0,
        "HD95": 0.0,
        "ASD": 0.0
    }

    with torch.no_grad():
        for images, masks in tqdm(loader, desc="Testing"):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            dice, sensitivity, specificity, iou, precision, hd95, asd = calculate_metrics(outputs, masks)

            metrics_sum["Dice"] += dice
            metrics_sum["IoU"] += iou
            metrics_sum["Sensitivity"] += sensitivity
            metrics_sum["Specificity"] += specificity
            metrics_sum["Precision"] += precision
            metrics_sum["HD95"] += hd95
            metrics_sum["ASD"] += asd

    avg_metrics = {k: v / len(loader) for k, v in metrics_sum.items()}
    print("\n" + "=" * 50)
    print("测试集平均评估指标")
    print("=" * 50)
    for key, val in avg_metrics.items():
        print(f"{key:12s}: {val:.4f}")
    print("=" * 50)
    return avg_metrics


if __name__ == "__main__":
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False

    best_dice = train_model()
    print(f"最终最佳Dice系数: {best_dice:.4f}")