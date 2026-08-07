import os
import cv2
import random
import numpy as np
import torch
from torch.utils.data import Dataset


class OptimizedLungDataset(Dataset):
    """优化的肺部数据集类"""

    def __init__(self, image_dir, mask_dir, is_training=False):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.is_training = is_training

        # 获取图像文件列表
        self.image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('_img.npy')])
        self.mask_files = [f.replace('_img.npy', '_label.npy') for f in self.image_files]

        # 验证掩码文件存在
        for mask_file in self.mask_files:
            if not os.path.exists(os.path.join(mask_dir, mask_file)):
                raise FileNotFoundError(f"Mask file not found: {os.path.join(mask_dir, mask_file)}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # 加载图像和掩码
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_files[idx])

        image = np.load(img_path)
        mask = np.load(mask_path)

        # 标准化
        image = self.optimized_normalize(image)

        # 数据增强
        if self.is_training:
            image, mask = self.apply_augmentations(image, mask)

        # 转换为张量
        image = torch.from_numpy(image).float().unsqueeze(0)
        mask = torch.from_numpy(mask).float().unsqueeze(0)

        return image, mask

    def optimized_normalize(self, image):
        """优化的标准化方法"""
        # 百分位数裁剪
        p2, p98 = np.percentile(image, [2, 98])
        image = np.clip(image, p2, p98)

        # Z-score标准化
        mean = np.mean(image)
        std = np.std(image)
        if std < 1e-8:
            std = 1e-8

        image = (image - mean) / (std + 1e-8)
        return image

    def apply_augmentations(self, image, mask):
        """应用数据增强"""
        # 随机水平翻转
        if random.random() > 0.5:
            image = np.fliplr(image).copy()
            mask = np.fliplr(mask).copy()

        # 随机垂直翻转
        if random.random() > 0.5:
            image = np.flipud(image).copy()
            mask = np.flipud(mask).copy()

        # 随机旋转
        if random.random() > 0.5:
            k = random.randint(1, 3)
            image = np.rot90(image, k).copy()
            mask = np.rot90(mask, k).copy()

        # 随机缩放和裁剪
        if random.random() > 0.4:
            h, w = image.shape
            scale = random.uniform(0.92, 1.08)
            new_h, new_w = int(h * scale), int(w * scale)

            image_resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            mask_resized = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

            if scale > 1.0:
                top = random.randint(0, new_h - h)
                left = random.randint(0, new_w - w)
                image = image_resized[top:top + h, left:left + w]
                mask = mask_resized[top:top + h, left:left + w]
            else:
                pad_h = h - new_h
                pad_w = w - new_w
                top_pad = random.randint(0, pad_h)
                left_pad = random.randint(0, pad_w)
                image = np.pad(image_resized,
                               ((top_pad, pad_h - top_pad), (left_pad, pad_w - left_pad)),
                               mode='constant', constant_values=0)
                mask = np.pad(mask_resized,
                              ((top_pad, pad_h - top_pad), (left_pad, pad_w - left_pad)),
                              mode='constant', constant_values=0)

        # 高斯模糊
        if random.random() > 0.8:
            image = cv2.GaussianBlur(image, (3, 3), 0)

        # 亮度调整
        if random.random() > 0.8:
            brightness_factor = random.uniform(0.97, 1.03)
            image = image * brightness_factor

        # 对比度调整
        if random.random() > 0.8:
            contrast_factor = random.uniform(0.97, 1.03)
            image_mean = np.mean(image)
            image = (image - image_mean) * contrast_factor + image_mean

        # 添加噪声
        if random.random() > 0.9:
            noise_std = random.uniform(0, 0.008)
            noise = np.random.normal(0, noise_std, image.shape)
            image = image + noise

        # 确保数值范围
        image = np.clip(image, -2.5, 2.5)

        return image, mask