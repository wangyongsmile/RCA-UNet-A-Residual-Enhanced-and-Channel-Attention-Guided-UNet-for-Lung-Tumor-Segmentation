import matplotlib.pyplot as plt
from trainers import (
    train_attention_unet,
    train_drs_cnn2,
    train_incremental_mrrn,
    train_r2unet,
    train_segnet,
    train_basic_unet,
    train_remove_channel_rca_unet,
    train_remove_res_rca_unet,
    train_rca_unet
)


def main():
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False

    print("=" * 60)
    print("开始模型对比实验 (9个模型)")
    print("=" * 60)

    results = {}

    print("\n1. 训练 Attention U-Net")
    results['Attention U-Net'] = train_attention_unet()

    print("\n2. 训练 DRS-CNN2")
    results['DRS-CNN2'] = train_drs_cnn2()

    print("\n3. 训练 Incremental-MRRN")
    results['Incremental-MRRN'] = train_incremental_mrrn()

    print("\n4. 训练 R2U-Net")
    results['R2U-Net'] = train_r2unet()

    print("\n5. 训练 SegNet")
    results['SegNet'] = train_segnet()

    print("\n6. 训练 BasicUNet (无残差)")
    results['BasicUNet无残差'] = train_basic_unet()

    print("\n7. 训练 RCAUNet-无通道注意力")
    results['RCAUNet-无通道注意力'] = train_remove_channel_rca_unet()

    print("\n8. 训练 RCAUNet-无残差")
    results['RCAUNet-无残差'] = train_remove_res_rca_unet()

    print("\n9. 训练 RCAUNet")
    results['RCAUNet'] = train_rca_unet()

    print("\n" + "=" * 60)
    print("模型对比实验结果汇总")
    print("=" * 60)
    for model_name, best_dice in results.items():
        print(f"{model_name:25s}: {best_dice:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()