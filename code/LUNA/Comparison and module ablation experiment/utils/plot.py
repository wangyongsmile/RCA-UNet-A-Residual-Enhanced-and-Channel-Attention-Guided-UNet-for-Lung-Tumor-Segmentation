import matplotlib.pyplot as plt


def plot_metrics(history, model_name="Model"):
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
    filename = f"results/training_metrics_{model_name.lower().replace(' ', '_').replace('-', '_')}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Metrics plot saved to {filename}")