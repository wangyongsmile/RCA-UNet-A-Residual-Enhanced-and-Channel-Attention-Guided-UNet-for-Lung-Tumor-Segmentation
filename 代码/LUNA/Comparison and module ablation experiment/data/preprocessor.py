import numpy as np
import albumentations as A


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