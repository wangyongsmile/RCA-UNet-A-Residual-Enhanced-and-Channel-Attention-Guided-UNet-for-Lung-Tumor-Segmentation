import numpy as np


class SliceExtractor:
    def __init__(self, margin=1):
        self.margin = margin

    def extract_slices_with_masks(self, ct_array, spacing, annotations_df):
        from data.luna_loader import LUNADataLoader

        slices_with_masks = []
        depth, height, width = ct_array.shape

        for _, row in annotations_df.iterrows():
            world_coord = np.array([row["coordX"], row["coordY"], row["coordZ"]])
            if isinstance(row["origin"], str):
                origin = np.array([float(x) for x in row["origin"][1:-1].split()])
            else:
                origin = row["origin"]

            i, j, k = LUNADataLoader.world2voxel(world_coord, origin, spacing)
            radius = row["diameter_mm"] / (2 * spacing[0])

            i = np.clip(i, 0, width - 1)
            j = np.clip(j, 0, height - 1)
            k = np.clip(k, 0, depth - 1)
            radius = max(2, int(np.round(radius)))

            for dz in range(-self.margin, self.margin + 1):
                slice_idx = k + dz
                if 0 <= slice_idx < depth:
                    ct_slice = ct_array[slice_idx]
                    mask = np.zeros((height, width), dtype=np.uint8)

                    y_coords, x_coords = np.ogrid[:height, :width]
                    distance = np.sqrt((x_coords - i) ** 2 + (y_coords - j) ** 2)
                    mask[distance <= radius] = 1

                    if np.sum(mask) > 0:
                        slices_with_masks.append((slice_idx, ct_slice, mask))

        return slices_with_masks