import os
import pandas as pd
import numpy as np
import SimpleITK as sitk
from config import BASE_PATH


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