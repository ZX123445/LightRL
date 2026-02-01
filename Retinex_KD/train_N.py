import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import os
from PIL import Image
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn.functional as F
from N import LowLightEnhancer, EnhancementLoss  # 使用统一的损失函数
os.environ["QT_QPA_PLATFORM"] = "offscreen"
from torch.cuda.amp import GradScaler, autocast

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 数据集路径
train_low_dir = "/home/xd508/zyx/LOL/our485/low/"
train_normal_dir = "/home/xd508/zyx/LOL/our485/high/"
val_low_dir = "/home/xd508/zyx/LOL/eval15/low/"
val_normal_dir = "/home/xd508/zyx/LOL/eval15/high/"

# 检查路径是否存在
assert os.path.exists(train_low_dir), f"训练集低光图像路径不存在: {train_low_dir}"
assert os.path.exists(train_normal_dir), f"训练集正常图像路径不存在: {train_normal_dir}"
assert os.path.exists(val_low_dir), f"验证集低光图像路径不存在: {val_low_dir}"
assert os.path.exists(val_normal_dir), f"验证集正常图像路径不存在: {val_normal_dir}"

# 自定义数据集类
class LowLightDataset(Dataset):
    def __init__(self, low_dir, normal_dir, transform=None, phase='train'):
        self.low_dir = low_dir
        self.normal_dir = normal_dir
        self.transform = transform
        self.phase = phase

        # 获取图像列表并确保匹配
        self.low_images = sorted([f for f in os.listdir(low_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
        self.normal_images = sorted([f for f in os.listdir(normal_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])

        # 检查图像数量是否匹配
        assert len(self.low_images) == len(self.normal_images), \
            f"低光图像数量({len(self.low_images)})与正常图像数量({len(self.normal_images)})不匹配"

        # 检查文件名是否对应
        for low_img, normal_img in zip(self.low_images, self.normal_images):
            if low_img != normal_img:
                print(f"警告: 文件名不匹配 - {low_img} vs {normal_img}")

        print(f"加载{phase}数据集: {len(self.low_images)}张图像")

    def __len__(self):
        return len(self.low_images)

    def __getitem__(self, idx):
        low_img_path = os.path.join(self.low_dir, self.low_images[idx])
        normal_img_path = os.path.join(self.normal_dir, self.normal_images[idx])

        low_img = Image.open(low_img_path).convert('RGB')
        normal_img = Image.open(normal_img_path).convert('RGB')

        if self.transform:
            low_img = self.transform(low_img)
            normal_img = self.transform(normal_img)
        else:
            # 默认转换
            to_tensor = transforms.ToTensor()
            low_img = to_tensor(low_img)
            normal_img = to_tensor(normal_img)

        return low_img, normal_img

# 数据预处理
def get_transforms(phase='train', img_size=256):
    if phase == 'train':
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.1),
            transforms.ToTensor(),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])

# 创建数据集和数据加载器
train_transform = get_transforms('train', img_size=128)
val_transform = get_transforms('val', img_size=128)

train_dataset = LowLightDataset(train_low_dir, train_normal_dir, train_transform, 'train')
val_dataset = LowLightDataset(val_low_dir, val_normal_dir, val_transform, 'val')

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=2, pin_memory=True)

# 初始化模型
model = LowLightEnhancer().to(device)

# 模型并行化
if torch.cuda.device_count() > 1:
    print(f"使用 {torch.cuda.device_count()} 个GPU进行训练")
    model = nn.DataParallel(model)

# 优化器和混合精度
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
scaler = GradScaler()

# 损失函数 - 使用统一实现
criterion = EnhancementLoss().to(device)

# 训练参数
num_epochs = 100
best_val_loss = float('inf')
train_loss_history = []
val_loss_history = []
learning_rates = []

# 创建保存目录
os.makedirs('checkpoints_N', exist_ok=True)
os.makedirs('results_N', exist_ok=True)
os.makedirs('loss_plots', exist_ok=True)

# 学习率调度器
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=5, verbose=True
)

# 训练函数
def train_epoch(model, loader, criterion, optimizer, epoch):
    model.train()
    running_loss = 0.0
    loss_components = {
        'l1_final': 0.0,
        'percep_final': 0.0,
        'illum_loss': 0.0,
        'tv_loss': 0.0,
        'l1_ssm': 0.0,
        'ssim_loss': 0.0,
        'weight_smooth': 0.0
    }

    progress_bar = tqdm(loader, desc=f'Epoch {epoch + 1}/{num_epochs}')
    for i, (low_imgs, normal_imgs) in enumerate(progress_bar):
        low_imgs = low_imgs.to(device)
        normal_imgs = normal_imgs.to(device)

        # 使用混合精度训练
        with autocast():
            # 前向传播
            results = model(low_imgs)
            
            # 计算损失 - 使用正常光图像作为target
            total_loss, comp_loss = criterion(results, normal_imgs)

        # 反向传播和优化
        optimizer.zero_grad()
        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # 统计损失
        running_loss += total_loss.item()
        for key in loss_components:
            if key in comp_loss:
                loss_components[key] += comp_loss[key].item()

        # 更新进度条
        avg_loss = running_loss / (i + 1)
        progress_bar.set_postfix({
            'loss': avg_loss,
            'l1_final': loss_components['l1_final'] / (i + 1),
            'percep_final': loss_components['percep_final'] / (i + 1)
        })

    avg_loss = running_loss / len(loader)
    for key in loss_components:
        loss_components[key] /= len(loader)

    return avg_loss, loss_components

# 验证函数
def validate(model, loader, criterion, epoch):
    model.eval()
    running_loss = 0.0
    loss_components = {
        'l1_final': 0.0,
        'percep_final': 0.0,
        'illum_loss': 0.0,
        'tv_loss': 0.0,
        'l1_ssm': 0.0,
        'ssim_loss': 0.0
    }
    
    # 用于计算PSNR和SSIM
    psnr_values = []
    ssim_values = []

    # 保存一些示例结果
    save_examples = []

    with torch.no_grad():
        progress_bar = tqdm(loader, desc=f'Validating Epoch {epoch + 1}')
        for i, (low_imgs, normal_imgs) in enumerate(progress_bar):
            low_imgs = low_imgs.to(device)
            normal_imgs = normal_imgs.to(device)

            # 前向传播
            results = model(low_imgs)

            # 计算损失
            total_loss, comp_loss = criterion(results, normal_imgs)

            # 统计损失
            running_loss += total_loss.item()
            for key in loss_components:
                if key in comp_loss:
                    loss_components[key] += comp_loss[key].item()
            
            # 计算PSNR和SSIM
            final_output = results['output']
            mse = F.mse_loss(final_output, normal_imgs)
            psnr = 10 * torch.log10(1.0 / mse)
            psnr_values.append(psnr.item())
            
            ssim_val = 1 - comp_loss['ssim_loss']  # SSIM损失是1-ssim
            ssim_values.append(ssim_val.cpu().numpy())
            

            # 保存前几个示例
            if i < 2:  # 每个epoch保存前2个批次的结果
                save_examples.append({
                    'low': low_imgs.cpu(),
                    'enhanced': results['enhanced'].cpu(),
                    'output': results['output'].cpu(),
                    'normal': normal_imgs.cpu(),
                    'illum': results['fused_illum'].cpu()
                })

    avg_loss = running_loss / len(loader)
    for key in loss_components:
        loss_components[key] /= len(loader)
    
    avg_psnr = np.mean(psnr_values)
    avg_ssim = np.mean(ssim_values)

    # 保存示例图像
    if save_examples:
        save_example_images(save_examples, epoch)

    return avg_loss, loss_components, avg_psnr, avg_ssim

# 保存示例图像
def save_example_images(examples, epoch):
    for i, batch in enumerate(examples):
        for j in range(min(2, batch['low'].size(0))):  # 每个批次保存2张图像
            fig, axs = plt.subplots(2, 3, figsize=(15, 10))
            fig.suptitle(f'Epoch {epoch+1} - Image {j+1}', fontsize=16)
            
            # 低光图像
            low_img = batch['low'][j].permute(1, 2, 0).numpy()
            low_img = np.clip(low_img, 0, 1)
            axs[0, 0].imshow(low_img)
            axs[0, 0].set_title('Low Light Input')
            axs[0, 0].axis('off')
            
            # 光照图
            illum_img = batch['illum'][j].permute(1, 2, 0).numpy()
            illum_img = np.clip(illum_img, 0, 1)
            axs[0, 1].imshow(illum_img, cmap='viridis')
            axs[0, 1].set_title('Illumination Map')
            axs[0, 1].axis('off')
            
            # 初步增强图像
            enhanced_img = batch['enhanced'][j].permute(1, 2, 0).numpy()
            enhanced_img = np.clip(enhanced_img, 0, 1)
            axs[0, 2].imshow(enhanced_img)
            axs[0, 2].set_title('Enhanced Output')
            axs[0, 2].axis('off')
            
            # 模型最终输出
            output_img = batch['output'][j].permute(1, 2, 0).numpy()
            output_img = np.clip(output_img, 0, 1)
            axs[1, 0].imshow(output_img)
            axs[1, 0].set_title('Final Output')
            axs[1, 0].axis('off')
            
            # 正常光照图像
            normal_img = batch['normal'][j].permute(1, 2, 0).numpy()
            normal_img = np.clip(normal_img, 0, 1)
            axs[1, 1].imshow(normal_img)
            axs[1, 1].set_title('Ground Truth')
            axs[1, 1].axis('off')
            
            # 差异图
            diff_img = np.abs(output_img - normal_img)
            axs[1, 2].imshow(diff_img, cmap='hot')
            axs[1, 2].set_title('Difference Map')
            axs[1, 2].axis('off')

            plt.tight_layout()
            plt.savefig(f'results/epoch_{epoch+1}_batch_{i+1}_img_{j+1}.png', bbox_inches='tight')
            plt.close()

# 绘制损失曲线
def plot_loss_history(train_loss, val_loss, epoch):
    plt.figure(figsize=(12, 6))
    plt.plot(train_loss, label='Training Loss', linewidth=2)
    plt.plot(val_loss, label='Validation Loss', linewidth=2)
    
    # 标记最佳验证点
    min_val_loss = min(val_loss)
    min_idx = val_loss.index(min_val_loss)
    plt.scatter(min_idx, min_val_loss, color='red', s=100, zorder=5)
    plt.annotate(f'Best: {min_val_loss:.4f}', 
                 (min_idx, min_val_loss),
                 xytext=(min_idx+5, min_val_loss+0.05),
                 arrowprops=dict(facecolor='black', shrink=0.05))
    
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'loss_plots/loss_epoch_{epoch+1}.png')
    plt.close()

# 主训练循环
start_time = time.time()

# 第一阶段: 只训练SSM增强器 (前30个epoch)
print("第一阶段训练: 只训练SSM增强器")
for name, param in model.named_parameters():
    if 'retinex_net' in name or 'fuse_net' in name or 'denoiser' in name:
        param.requires_grad = False
    else:
        param.requires_grad = True

# 训练循环
for epoch in range(num_epochs):
    # 训练阶段
    train_loss, train_components = train_epoch(model, train_loader, criterion, optimizer, epoch)
    train_loss_history.append(train_loss)
    
    # 验证阶段
    val_loss, val_components, val_psnr, val_ssim = validate(model, val_loader, criterion, epoch)
    val_loss_history.append(val_loss)
    
    # 更新学习率
    current_lr = optimizer.param_groups[0]['lr']
    learning_rates.append(current_lr)
    scheduler.step(val_loss)
    
    # 打印统计信息
    print(f"\nEpoch {epoch + 1}/{num_epochs} | "
          f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
          f"Val PSNR: {val_psnr:.2f} dB | Val SSIM: {val_ssim:.4f}")
    
    # 打印详细损失组件
    print(f"  L1 Final: {train_components['l1_final']:.4f}/{val_components['l1_final']:.4f} | "
          f"Percep: {train_components['percep_final']:.4f}/{val_components['percep_final']:.4f} | "
          f"Illum: {train_components['illum_loss']:.4f}/{val_components['illum_loss']:.4f} | "
          f"SSIM: {train_components['ssim_loss']:.4f}/{val_components['ssim_loss']:.4f}")
    
    # 每10个epoch绘制损失曲线
    if (epoch + 1) % 5 == 0:
        plot_loss_history(train_loss_history, val_loss_history, epoch)
    
    # 保存最佳模型
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        # 保存完整模型和状态
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_psnr': val_psnr,
            'val_ssim': val_ssim
        }, 'checkpoints/best_model.pth')
        print(f"保存最佳模型, Val Loss: {val_loss:.4f}, PSNR: {val_psnr:.2f} dB")

    # 每10个epoch保存一次
    if (epoch + 1) % 10 == 0:
        torch.save(model.state_dict(), f'checkpoints/model_epoch_{epoch + 1}.pth')
    
    # 第二阶段: 解冻所有层 (从第30个epoch开始)
    if epoch == 30:
        print("第二阶段训练: 解冻所有层")
        for param in model.parameters():
            param.requires_grad = True
        # 重置优化器学习率
        for g in optimizer.param_groups:
            g['lr'] = 1e-5
        print(f"学习率重置为: {optimizer.param_groups[0]['lr']}")

# 保存最终模型
torch.save(model.state_dict(), 'checkpoints/final_model.pth')
print("训练完成! 模型已保存到 checkpoints/ 目录")

# 绘制最终损失曲线
plt.figure(figsize=(12, 6))
plt.plot(train_loss_history, label='Training Loss', linewidth=2)
plt.plot(val_loss_history, label='Validation Loss', linewidth=2)
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig('final_loss_curve.png')

# 绘制学习率曲线
plt.figure(figsize=(12, 6))
plt.plot(learning_rates, label='Learning Rate', color='green', linewidth=2)
plt.title('Learning Rate Schedule')
plt.xlabel('Epoch')
plt.ylabel('Learning Rate')
plt.yscale('log')
plt.legend()
plt.grid(True)
plt.savefig('learning_rate_curve.png')

# 训练时间统计
end_time = time.time()
training_time = end_time - start_time
hours = int(training_time // 3600)
minutes = int((training_time % 3600) // 60)
seconds = int(training_time % 60)
print(f"总训练时间: {hours}小时 {minutes}分钟 {seconds}秒")
print(f"最佳验证损失: {best_val_loss:.4f}")
