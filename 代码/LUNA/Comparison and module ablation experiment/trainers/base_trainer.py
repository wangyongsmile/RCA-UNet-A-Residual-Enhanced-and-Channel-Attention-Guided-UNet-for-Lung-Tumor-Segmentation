import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
from collections import defaultdict

from config import BASE_PATH, DEVICE, BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE, WEIGHT_DECAY
from losses import ImprovedFocalDiceLoss
from utils import GradualWarmupScheduler, EnhancedEarlyStopping, calculate_metrics, plot_metrics


def train_model(model, train_dataset, val_dataset, test_dataset, model_name, model_save_name):
    print(f"训练集样本数: {len(train_dataset)}")
    print(f"验证集样本数: {len(val_dataset)}")
    print(f"测试集样本数: {len(test_dataset)}")
    print(f"批量大小: {BATCH_SIZE}")
    print(f"使用设备: {DEVICE}")

    train_weights = torch.DoubleTensor(train_dataset.sample_weights)
    sampler = WeightedRandomSampler(train_weights, len(train_weights))

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler,
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=4, pin_memory=True)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"{model_name}模型参数总数: {total_params:,}")

    criterion = ImprovedFocalDiceLoss(alpha=0.8, gamma=2.0, dice_weight=0.6, boundary_weight=0.2)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.999)
    )

    warmup_epochs = 10
    scheduler = GradualWarmupScheduler(optimizer, warmup_epochs, NUM_EPOCHS, min_lr_ratio=0.01)
    plateau_scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=15, verbose=True)

    best_dice = 0.0
    history = defaultdict(list)
    early_stopping = EnhancedEarlyStopping(patience=30, min_delta=1e-4, min_epochs=50, warmup_epochs=20)
    max_grad_norm = 1.0

    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss, train_dice, train_sensitivity, train_specificity = 0.0, 0.0, 0.0, 0.0
        train_iou, train_precision, train_hd95, train_asd = 0.0, 0.0, 0.0, 0.0

        train_progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS} [Train]")

        for i, (images, masks) in enumerate(train_progress):
            images = images.to(DEVICE, non_blocking=True)
            masks = masks.to(DEVICE, non_blocking=True)

            outputs = model(images)
            outputs_prob = torch.sigmoid(outputs) if not isinstance(model(torch.randn(1, 1, 256, 256).to(DEVICE)), torch.sigmoid) else outputs
            loss = criterion(outputs_prob, masks)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

            with torch.no_grad():
                dice, sensitivity, specificity, iou, precision, hd95, asd = calculate_metrics(outputs_prob, masks)

            train_loss += loss.item()
            train_dice += dice
            train_sensitivity += sensitivity
            train_specificity += specificity
            train_iou += iou
            train_precision += precision
            train_hd95 += hd95
            train_asd += asd

            train_progress.set_postfix({"Loss": f"{loss.item():.4f}", "Dice": f"{dice:.4f}"})

        scheduler.step()
        if epoch > warmup_epochs:
            plateau_scheduler.step(train_dice / len(train_loader))

        train_loss /= len(train_loader)
        train_dice /= len(train_loader)
        train_sensitivity /= len(train_loader)
        train_specificity /= len(train_loader)
        train_iou /= len(train_loader)
        train_precision /= len(train_loader)
        train_hd95 /= len(train_loader)
        train_asd /= len(train_loader)

        model.eval()
        val_loss, val_dice, val_sensitivity, val_specificity = 0.0, 0.0, 0.0, 0.0
        val_iou, val_precision, val_hd95, val_asd = 0.0, 0.0, 0.0, 0.0

        with torch.no_grad():
            val_progress = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS} [Val]")
            for images, masks in val_progress:
                images, masks = images.to(DEVICE), masks.to(DEVICE)
                outputs = model(images)
                outputs_prob = torch.sigmoid(outputs) if not isinstance(model(torch.randn(1, 1, 256, 256).to(DEVICE)), torch.sigmoid) else outputs
                loss = criterion(outputs_prob, masks)
                dice, sensitivity, specificity, iou, precision, hd95, asd = calculate_metrics(outputs_prob, masks)

                val_loss += loss.item()
                val_dice += dice
                val_sensitivity += sensitivity
                val_specificity += specificity
                val_iou += iou
                val_precision += precision
                val_hd95 += hd95
                val_asd += asd

        val_loss /= len(val_loader)
        val_dice /= len(val_loader)
        val_sensitivity /= len(val_loader)
        val_specificity /= len(val_loader)
        val_iou /= len(val_loader)
        val_precision /= len(val_loader)
        val_hd95 /= len(val_loader)
        val_asd /= len(val_loader)

        overfit_ratio = train_dice / val_dice if val_dice > 1e-8 else 0

        if val_dice > best_dice:
            best_dice = val_dice
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_dice': best_dice,
            }, os.path.join(BASE_PATH, "trained_models", f"best_{model_save_name}.pth"))
            print(f"New best {model_name} model saved with Dice: {val_dice:.4f}, IoU: {val_iou:.4f}")

        if early_stopping(val_dice, epoch + 1):
            print(f"Early stopping triggered at epoch {epoch + 1}")
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

        if (epoch + 1) % 5 == 0 or epoch < 10:
            print(f"Epoch {epoch + 1}/{NUM_EPOCHS} ({model_name})")
            print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            print(f"Train Dice: {train_dice:.4f}, Val Dice: {val_dice:.4f}")
            print(f"过拟合比率: {overfit_ratio:.2f}")
            print(f"Current Learning Rate: {optimizer.param_groups[0]['lr']:.2e}")
            if torch.cuda.is_available():
                print(f"GPU Memory: {torch.cuda.memory_allocated() / 1024 ** 3:.2f}GB")
            print("-" * 60)

    plot_metrics(history, model_name)
    print(f"{model_name} Training completed! Best Dice: {best_dice:.4f}")
    return best_dice