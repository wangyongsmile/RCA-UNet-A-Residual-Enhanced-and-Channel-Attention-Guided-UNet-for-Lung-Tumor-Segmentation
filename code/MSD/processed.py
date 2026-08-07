import os
import nibabel as nib
import numpy as np
import torch
from sklearn.model_selection import train_test_split
import random
from scipy.ndimage import rotate, map_coordinates
from scipy.interpolate import RegularGridInterpolator
import warnings

warnings.filterwarnings('ignore')

# 配置参数
DATA_PATH = "D:/Task06_Lung/"  # MSD数据集根目录
IMAGE_DIR = os.path.join(DATA_PATH, "imagesTr")
LABEL_DIR = os.path.join(DATA_PATH, "labelsTr")
SAVE_DIR = "D:/LungPreprocessed_63/"  # 预处理后保存路径
NUM_CASES = 63 # 使用30例数据
TARGET_SIZE = (256, 256)  # 目标图像尺寸
TRAIN_RATIO = 0.8  # 训练集比例
RANDOM_SEED = 42  # 随机种子
MIN_TUMOR_PIXELS = 50  # 最小肿瘤像素阈值

# 创建保存目录
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(os.path.join(SAVE_DIR, "train"), exist_ok=True)
os.makedirs(os.path.join(SAVE_DIR, "val"), exist_ok=True)


def get_case_list():
    """获取所有可用的病例ID"""
    cases = []
    for filename in os.listdir(IMAGE_DIR):
        if filename.endswith('.nii.gz'):
            case_id = filename.replace('.nii.gz', '')
            # 检查对应的标签文件是否存在
            label_path = os.path.join(LABEL_DIR, filename)
            if os.path.exists(label_path):
                cases.append(case_id)
    return cases


def resize_slice(slice_data, target_size):
    """调整切片尺寸"""
    h, w = slice_data.shape
    if h == target_size[0] and w == target_size[1]:
        return slice_data

    # 创建坐标网格
    x = np.linspace(0, w - 1, target_size[1])
    y = np.linspace(0, h - 1, target_size[0])
    xx, yy = np.meshgrid(x, y)

    # 使用RegularGridInterpolator进行插值
    interp = RegularGridInterpolator((np.arange(h), np.arange(w)), slice_data,
                                     method='linear', bounds_error=False, fill_value=0)
    resized = interp((yy, xx))

    return resized


def normalize_slice(slice_data):
    """CT值标准化（窗宽窗位）"""
    # 肺窗：窗位-600，窗位1500
    window_level = -600
    window_width = 1500
    min_val = window_level - window_width // 2
    max_val = window_level + window_width // 2

    # 限制在窗宽范围内
    slice_data = np.clip(slice_data, min_val, max_val)

    # 归一化到[0, 1]
    if max_val > min_val:
        slice_data = (slice_data - min_val) / (max_val - min_val)

    return slice_data.astype(np.float32)


def augment_slice(img_slice, label_slice):
    """数据增强"""
    # 随机旋转（-10到10度）
    if random.random() > 0.5:
        angle = random.uniform(-10, 10)
        img_slice = rotate(img_slice, angle, reshape=False, order=1, mode='constant', cval=0)
        label_slice = rotate(label_slice, angle, reshape=False, order=0, mode='constant', cval=0)

    # 随机翻转
    if random.random() > 0.5:
        img_slice = np.fliplr(img_slice)
        label_slice = np.fliplr(label_slice)

    if random.random() > 0.5:
        img_slice = np.flipud(img_slice)
        label_slice = np.flipud(label_slice)

    return img_slice, label_slice


def process_case(case_id, is_train=True):
    """处理单个病例"""
    img_path = os.path.join(IMAGE_DIR, f"{case_id}.nii.gz")
    label_path = os.path.join(LABEL_DIR, f"{case_id}.nii.gz")

    # 加载数据
    img = nib.load(img_path).get_fdata()
    label = nib.load(label_path).get_fdata()

    slices = []

    # 遍历所有切片
    for i in range(img.shape[2]):
        img_slice = img[:, :, i]
        label_slice = label[:, :, i]

        # 调整尺寸
        img_slice = resize_slice(img_slice, TARGET_SIZE)
        label_slice = resize_slice(label_slice, TARGET_SIZE)

        # 标准化
        img_slice = normalize_slice(img_slice)

        # 二值化标签
        label_slice = (label_slice > 0).astype(np.uint8)

        # 检查肿瘤像素数量
        tumor_pixels = np.sum(label_slice)
        if tumor_pixels < MIN_TUMOR_PIXELS:
            continue  # 跳过肿瘤太小的切片

        # 数据增强（仅训练集）
        if is_train:
            img_slice, label_slice = augment_slice(img_slice, label_slice)

        slices.append((img_slice, label_slice))

    return slices


def preprocess_data():
    """主预处理函数"""
    # 获取所有可用病例
    all_cases = get_case_list()
    print(f"找到 {len(all_cases)} 个可用病例")

    if len(all_cases) < NUM_CASES:
        raise ValueError(f"数据集中只有 {len(all_cases)} 个病例，少于请求的 {NUM_CASES} 个")

    # 随机选择病例
    selected_cases = random.sample(all_cases, NUM_CASES)
    print(f"选择的病例: {selected_cases}")

    # 划分训练集和验证集
    train_cases, val_cases = train_test_split(
        selected_cases,
        train_size=TRAIN_RATIO,
        random_state=RANDOM_SEED
    )

    print(f"训练集病例数: {len(train_cases)}, 验证集病例数: {len(val_cases)}")

    # 处理训练集
    for i, case_id in enumerate(train_cases):
        print(f"处理训练集病例: {case_id}")
        slices = process_case(case_id, is_train=True)

        for j, (img, label) in enumerate(slices):
            np.save(os.path.join(SAVE_DIR, "train", f"{case_id}_slice{j}_img.npy"), img)
            np.save(os.path.join(SAVE_DIR, "train", f"{case_id}_slice{j}_label.npy"), label)

    # 处理验证集
    for i, case_id in enumerate(val_cases):
        print(f"处理验证集病例: {case_id}")
        slices = process_case(case_id, is_train=False)

        for j, (img, label) in enumerate(slices):
            np.save(os.path.join(SAVE_DIR, "val", f"{case_id}_slice{j}_img.npy"), img)
            np.save(os.path.join(SAVE_DIR, "val", f"{case_id}_slice{j}_label.npy"), label)

    print("预处理完成!")


if __name__ == "__main__":
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    preprocess_data()
