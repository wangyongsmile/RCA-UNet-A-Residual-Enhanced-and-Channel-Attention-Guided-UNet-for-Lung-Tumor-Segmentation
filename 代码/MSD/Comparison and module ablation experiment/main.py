import os
import sys
import argparse
import warnings

warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入所有模型
from models.rca_unet import RCAUNet
from models.att_unet import AttentionUNet
from models.drs_cnn2 import DRS_CNN2
from models.incremental_mrrn import IncrementalMRRN
from models.r2_unet import R2UNet
from models.basic_unet import BasicUNet
from models.segnet import SegNet
from models.no_residual_rca_unet import NoResidualRCAUNet
from models.no_channel_attention_rca_unet import NoAttentionRCAUNet

from config import DATA_DIR, BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE, WEIGHT_DECAY, DEVICE
from trainer import train_model
from data import OptimizedLungDataset


def parse_args():
    parser = argparse.ArgumentParser(description='医学图像分割模型训练')
    # 在 choices 中添加新模型
    parser.add_argument('--model', type=str, required=True,
                        choices=['rca_unet', 'att_unet', 'drs_cnn2',
                                 'incremental_mrrn', 'r2_unet', 'basic_unet',
                                 'segnet','no_residual_rca_unet', 'no_channel_attention_rca_unet'],
                        help='选择要训练的模型')

    parser.add_argument('--data-dir', type=str, default=DATA_DIR,
                        help=f'数据目录路径 (默认: {DATA_DIR})')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE,
                        help=f'批大小 (默认: {BATCH_SIZE})')
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS,
                        help=f'训练轮数 (默认: {NUM_EPOCHS})')
    parser.add_argument('--lr', type=float, default=LEARNING_RATE,
                        help=f'学习率 (默认: {LEARNING_RATE})')
    parser.add_argument('--weight-decay', type=float, default=WEIGHT_DECAY,
                        help=f'权重衰减 (默认: {WEIGHT_DECAY})')
    parser.add_argument('--device', type=str, default=None,
                        help=f'设备: cuda 或 cpu (默认: 自动检测)')
    parser.add_argument('--model-params', type=str, default='',
                        help='模型特定参数 (JSON格式字符串或key=value对)')
    parser.add_argument('--results-dir', type=str, default='results',
                        help='结果保存目录 (默认: results)')

    return parser.parse_args()


def get_model(model_name, model_params=None):
    print(f"正在初始化 {model_name} 模型...")

    if model_params is None:
        model_params = {}

    if model_name == 'rca_unet':
        model = RCAUNet(
            in_channels=1,
            out_channels=1,
            features=model_params.get('features', [64, 128, 256, 512])
        )
    elif model_name == 'att_unet':
        model = AttentionUNet(
            in_channels=1,
            out_channels=1,
            features=model_params.get('features', [64, 128, 256, 512])
        )
    elif model_name == 'drs_cnn2':
        model = DRS_CNN2(
            in_channels=1,
            out_channels=1,
            features=model_params.get('features', [32, 64, 128, 256, 512])
        )
    elif model_name == 'incremental_mrrn':
        model = IncrementalMRRN(
            in_channels=1,
            out_channels=1,
            num_streams=model_params.get('num_streams', 3),
            base_channels=model_params.get('base_channels', 32)
        )
    elif model_name == 'r2_unet':
        model = R2UNet(
            in_channels=1,
            out_channels=1,
            features=model_params.get('features', [64, 128, 256, 512]),
            t=model_params.get('t', 2)
        )
    elif model_name == 'basic_unet':
        model = BasicUNet(
            in_channels=1,
            out_channels=1,
            features=model_params.get('features', [64, 128, 256, 512])
        )
    elif model_name == 'segnet':
        model = SegNet(
            in_channels=1,
            out_channels=1,
            init_channels=model_params.get('init_channels', 64)
        )
    elif model_name == 'no_residual_rca_unet':  # 新增
        model = NoResidualRCAUNet(
            in_channels=1,
            out_channels=1,
            features=model_params.get('features', [64, 128, 256, 512])
        )
    elif model_name == 'no_channel_attention_rca_unet':  # 新增
        model = NoAttentionRCAUNet(
            in_channels=1,
            out_channels=1,
            features=model_params.get('features', [64, 128, 256, 512])
        )
    else:
        raise ValueError(f"不支持的模型: {model_name}")

    return model


def parse_model_params(param_str):
    if not param_str:
        return {}
    params = {}
    if param_str.startswith('{'):
        try:
            import json
            params = json.loads(param_str)
        except json.JSONDecodeError:
            print(f"警告: 无法解析JSON参数: {param_str}")
    else:
        for pair in param_str.split(','):
            if '=' in pair:
                key, value = pair.split('=', 1)
                key = key.strip()
                value = value.strip()
                if value.lower() == 'true':
                    params[key] = True
                elif value.lower() == 'false':
                    params[key] = False
                elif value.isdigit():
                    params[key] = int(value)
                else:
                    try:
                        params[key] = float(value)
                    except ValueError:
                        params[key] = value
    return params


def setup_directories(results_dir, model_name):
    os.makedirs(results_dir, exist_ok=True)
    model_results_dir = os.path.join(results_dir, model_name)
    os.makedirs(model_results_dir, exist_ok=True)
    return model_results_dir


def print_config(args, model_params):
    print("=" * 60)
    print("模型训练配置")
    print("=" * 60)
    print(f"模型: {args.model}")
    print(f"数据目录: {args.data_dir}")
    print(f"批大小: {args.batch_size}")
    print(f"训练轮数: {args.epochs}")
    print(f"学习率: {args.lr}")
    print(f"权重衰减: {args.weight_decay}")
    print(f"设备: {DEVICE if args.device is None else args.device}")
    print(f"结果目录: {args.results_dir}")
    if model_params:
        print(f"模型特定参数: {model_params}")
    print("=" * 60)


def main():
    args = parse_args()
    model_params = parse_model_params(args.model_params)
    print_config(args, model_params)

    model_results_dir = setup_directories(args.results_dir, args.model)
    model = get_model(args.model, model_params)

    print(f"\n正在加载数据集...")
    train_dataset = OptimizedLungDataset(
        image_dir=os.path.join(args.data_dir, "train"),
        mask_dir=os.path.join(args.data_dir, "train"),
        is_training=True
    )
    val_dataset = OptimizedLungDataset(
        image_dir=os.path.join(args.data_dir, "val"),
        mask_dir=os.path.join(args.data_dir, "val"),
        is_training=False
    )
    print(f"训练集样本数: {len(train_dataset)}")
    print(f"验证集样本数: {len(val_dataset)}")

    from config import update_config
    update_config(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        device=args.device,
        results_dir=model_results_dir
    )

    print(f"\n开始训练 {args.model} 模型...")
    best_dice = train_model(
        model=model,
        model_name=args.model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        save_path=os.path.join(model_results_dir, f"best_model_{args.model}.pth"),
        plot_path=os.path.join(model_results_dir, f"training_metrics_{args.model}.png")
    )

    print(f"\n{'=' * 60}")
    print(f"{args.model} 模型训练完成!")
    print(f"最佳验证Dice系数: {best_dice:.4f}")
    print(f"模型已保存至: {os.path.join(model_results_dir, f'best_model_{args.model}.pth')}")
    print(f"训练曲线已保存至: {os.path.join(model_results_dir, f'training_metrics_{args.model}.png')}")
    print('=' * 60)


if __name__ == "__main__":
    main()