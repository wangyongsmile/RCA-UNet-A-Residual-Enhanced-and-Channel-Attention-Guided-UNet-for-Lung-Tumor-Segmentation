import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from collections import defaultdict
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts

from config import BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE, WEIGHT_DECAY, DEVICE, RESULTS_DIR
from losses import OptimizedCombinedLoss
from metrics import calculate_metrics


def plot_metrics(history, model_name="Model", save_path=None):
    """绘制训练指标"""
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
        plt.xlabel(f'Epoch ({model_name})')
        plt.ylabel(title)
        plt.legend()
        plt.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.savefig(f"{RESULTS_DIR}/training_metrics_{model_name}.png", dpi=300, bbox_inches='tight')

    plt.close()
    print(f"Metrics plot ({model_name}) saved")


def train_model(model, model_name, train_dataset, val_dataset, save_path=None, plot_path=None):
    """通用训练函数"""
    # 设置保存路径
    if save_path is None:
        save_path = f"{RESULTS_DIR}/best_model_{model_name}.pth"

    if plot_path is None:
        plot_path = f"{RESULTS_DIR}/training_metrics_{model_name}.png"

    # 打印设备信息
    if torch.cuda.is_available():
        print(f"使用GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f} GB")
    else:
        print("使用CPU进行训练")

    # 创建数据加载器
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

    # 将模型移动到设备
    model = model.to(DEVICE)

    # 打印模型参数数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{model_name} 模型参数总数: {total_params:,}")

    # 初始化损失函数、优化器、调度器
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

    # 混合精度训练
    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None

    # 训练变量初始化
    best_dice = 0.0
    history = defaultdict(list)
    patience_counter = 0
    max_patience = 35
    accumulation_steps = 2
    max_grad_norm = 1.0

    # 训练循环
    for epoch in range(NUM_EPOCHS):
        # 训练阶段
        model.train()
        train_metrics = {
            'loss': 0.0, 'dice': 0.0, 'sensitivity': 0.0, 'specificity': 0.0,
            'iou': 0.0, 'precision': 0.0, 'hd95': 0.0, 'asd': 0.0
        }

        train_progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS} [Train]")
        optimizer.zero_grad()

        for i, (images, masks) in enumerate(train_progress):
            images = images.to(DEVICE, non_blocking=True)
            masks = masks.to(DEVICE, non_blocking=True)

            # 前向传播
            if scaler:
                with torch.cuda.amp.autocast():
                    outputs = model(images)
                    loss = criterion(outputs, masks) / accumulation_steps
            else:
                outputs = model(images)
                loss = criterion(outputs, masks) / accumulation_steps

            # 反向传播
            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            # 梯度累积
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

            # 计算指标
            with torch.no_grad():
                dice, sensitivity, specificity, iou, precision, hd95, asd = calculate_metrics(
                    torch.sigmoid(outputs), masks
                )

            # 累加指标
            train_metrics['loss'] += loss.item() * accumulation_steps
            train_metrics['dice'] += dice
            train_metrics['sensitivity'] += sensitivity
            train_metrics['specificity'] += specificity
            train_metrics['iou'] += iou
            train_metrics['precision'] += precision
            train_metrics['hd95'] += hd95
            train_metrics['asd'] += asd

            # 更新进度条
            train_progress.set_postfix({
                "Loss": f"{loss.item() * accumulation_steps:.4f}",
                "Dice": f"{dice:.4f}",
                "Precision": f"{precision:.4f}",
                "IoU": f"{iou:.4f}"
            })

        # 计算平均训练指标
        train_metrics = {k: v / len(train_loader) for k, v in train_metrics.items()}

        # 余弦退火调度器步进
        cosine_scheduler.step()

        # 验证阶段
        model.eval()
        val_metrics = {
            'loss': 0.0, 'dice': 0.0, 'sensitivity': 0.0, 'specificity': 0.0,
            'iou': 0.0, 'precision': 0.0, 'hd95': 0.0, 'asd': 0.0
        }

        with torch.no_grad():
            val_progress = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS} [Val]")

            for images, masks in val_progress:
                images = images.to(DEVICE, non_blocking=True)
                masks = masks.to(DEVICE, non_blocking=True)

                # 前向传播
                if scaler:
                    with torch.cuda.amp.autocast():
                        outputs = model(images)
                        loss = criterion(outputs, masks)
                else:
                    outputs = model(images)
                    loss = criterion(outputs, masks)

                # 计算指标
                dice, sensitivity, specificity, iou, precision, hd95, asd = calculate_metrics(
                    torch.sigmoid(outputs), masks
                )

                # 累加指标
                val_metrics['loss'] += loss.item()
                val_metrics['dice'] += dice
                val_metrics['sensitivity'] += sensitivity
                val_metrics['specificity'] += specificity
                val_metrics['iou'] += iou
                val_metrics['precision'] += precision
                val_metrics['hd95'] += hd95
                val_metrics['asd'] += asd

                # 更新进度条
                val_progress.set_postfix({
                    "Loss": f"{loss.item():.4f}",
                    "Dice": f"{dice:.4f}",
                    "Precision": f"{precision:.4f}",
                    "IoU": f"{iou:.4f}"
                })

        # 计算平均验证指标
        val_metrics = {k: v / len(val_loader) for k, v in val_metrics.items()}

        # 更新学习率（基于plateau）
        plateau_scheduler.step(val_metrics['dice'])

        # 计算过拟合比率
        overfit_ratio = train_metrics['dice'] / val_metrics['dice'] if val_metrics['dice'] > 1e-8 else 0.0

        # 保存最佳模型
        if val_metrics['dice'] > best_dice:
            best_dice = val_metrics['dice']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_dice': best_dice,
            }, save_path)
            print(f"New best {model_name} model saved with Dice: {val_metrics['dice']:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1

        # 早停检查
        if patience_counter >= max_patience:
            print(f"Early stopping triggered after {epoch + 1} epochs")
            break

        # 记录历史
        history['train_loss'].append(train_metrics['loss'])
        history['val_loss'].append(val_metrics['loss'])
        history['train_dice'].append(train_metrics['dice'])
        history['val_dice'].append(val_metrics['dice'])
        history['train_iou'].append(train_metrics['iou'])
        history['val_iou'].append(val_metrics['iou'])
        history['train_sensitivity'].append(train_metrics['sensitivity'])
        history['val_sensitivity'].append(val_metrics['sensitivity'])
        history['train_specificity'].append(train_metrics['specificity'])
        history['val_specificity'].append(val_metrics['specificity'])
        history['train_precision'].append(train_metrics['precision'])
        history['val_precision'].append(val_metrics['precision'])
        history['train_hd'].append(train_metrics['hd95'])
        history['val_hd'].append(val_metrics['hd95'])
        history['train_asd'].append(train_metrics['asd'])
        history['val_asd'].append(val_metrics['asd'])
        history['overfit_ratio'].append(overfit_ratio)
        history['learning_rate'].append(optimizer.param_groups[0]['lr'])

        # 打印日志
        print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS} ({model_name})")
        print(f"Train Loss: {train_metrics['loss']:.4f}, Val Loss: {val_metrics['loss']:.4f}")
        print(f"Train Dice: {train_metrics['dice']:.4f}, Val Dice: {val_metrics['dice']:.4f}")
        print(f"Train IoU: {train_metrics['iou']:.4f}, Val IoU: {val_metrics['iou']:.4f}")
        print(f"过拟合比率: {overfit_ratio:.2f}")
        print(f"Train Precision: {train_metrics['precision']:.4f}, Val Precision: {val_metrics['precision']:.4f}")
        print(f"Train 95% HD: {train_metrics['hd95']:.2f}, Val 95% HD: {val_metrics['hd95']:.2f}")
        print(f"Current Learning Rate: {optimizer.param_groups[0]['lr']:.2e}")
        print("-" * 60)

    # 绘制训练曲线
    plot_metrics(history, model_name, plot_path)

    return best_dice