import os
import torch
import numpy as np
from torchvision.transforms import GaussianBlur
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T


def normalize_image(img):
    """归一化图像到[0,1]"""
    img = img.float()
    return (img - img.min()) / (img.max() - img.min() + 1e-6)


def gamma_correction(img, gamma):
    """对单张图像应用伽马变换"""
    return torch.pow(img.clamp(0, 1), gamma)


def denoise_image(img):
    """简单高斯模糊去噪"""
    return GaussianBlur(kernel_size=3, sigma=0.5)(img)


def generate_pseudo_exposures(img, gamma_values=[0.5, 0.8, 1.0, 1.5, 2.0], denoise=False):
    """生成固定 gamma 值的伪多曝光序列"""
    img_normalized = normalize_image(img)
    exposures = []

    for gamma in gamma_values:
        exp = gamma_correction(img_normalized, gamma)
        if denoise and gamma < 0.7:  # 对低伽马图像去噪
            exp = denoise_image(exp)
        exposures.append(exp)

    return exposures, gamma_values  # 直接返回传入的 gamma 值


class LowLightDataset(Dataset):
    """低光图像数据集（单张图像生成伪多曝光序列）"""

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = [os.path.join(root_dir, f) for f in os.listdir(root_dir) if f.endswith(('.jpg', '.png'))]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB')
        original_filename = os.path.basename(img_path)
        if self.transform:
            img = self.transform(img)
        return img, original_filename


# 数据预处理
transform = T.Compose([
    T.Resize((256, 256)),  # 统一尺寸
    T.ToTensor(),  # 转为Tensor
])

# 加载数据集
dataset = LowLightDataset(root_dir='H:/研究生论文/10_image_enhanced/LOLdataset/our485/low/', transform=transform)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

# 输出根目录
output_root = 'output/fixed_gamma_exposures'
os.makedirs(output_root, exist_ok=True)

# 创建分类目录
low_dir = os.path.join(output_root, 'low')
middle_dir = os.path.join(output_root, 'middle')
high_dir = os.path.join(output_root, 'high')
os.makedirs(low_dir, exist_ok=True)
os.makedirs(middle_dir, exist_ok=True)
os.makedirs(high_dir, exist_ok=True)

# 预定义的 gamma 值和对应的分类
gamma_config = [
    {'value': 0.5, 'category': 'low'},
    {'value': 0.8, 'category': 'low'},
    {'value': 1.0, 'category': 'middle'},
    {'value': 1.5, 'category': 'high'},
    {'value': 2.0, 'category': 'high'}
]

# 为每个 gamma 值创建子文件夹
gamma_folders = {}
for config in gamma_config:
    gamma = config['value']
    category = config['category']
    gamma_str = f"gamma_{gamma:.2f}".replace('.', '_')
    folder_path = os.path.join(output_root, category, gamma_str)
    os.makedirs(folder_path, exist_ok=True)
    gamma_folders[gamma] = folder_path


def save_image(tensor, save_path):
    """保存图像（兼容灰度和RGB图像）"""
    if tensor.dim() == 2:
        tensor = tensor.unsqueeze(0)
    if tensor.size(0) == 1:
        tensor = tensor.repeat(3, 1, 1)
    img_np = (tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    Image.fromarray(img_np).save(save_path)


# 批量处理数据集
for idx, (img, original_filename) in enumerate(dataloader):
    print(f"Processing image {idx + 1}/{len(dataloader)}: {original_filename}")
    exposures, gammas = generate_pseudo_exposures(img[0],
                                                gamma_values=[config['value'] for config in gamma_config],
                                                denoise=True)

    if isinstance(original_filename, tuple):
        original_filename = original_filename[0]

    # 保存到对应的 gamma 文件夹
    for config, exp in zip(gamma_config, exposures):
        gamma = config['value']
        gamma_str = f"gamma_{gamma:.2f}".replace('.', '_')
        save_path = os.path.join(gamma_folders[gamma], original_filename)
        save_image(exp, save_path)

print("所有图像处理完成！")