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