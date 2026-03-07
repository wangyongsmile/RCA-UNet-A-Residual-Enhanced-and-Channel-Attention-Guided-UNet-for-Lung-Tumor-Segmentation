import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


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