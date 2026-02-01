import os
import time
import random
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from skimage.metrics import structural_similarity as ssim
from math import log10
import argparse
from einops import rearrange
import glob

# 定义分解网络
class DecomNet(nn.Module):
    def __init__(self, channel=64, kernel_size=3):
        super(DecomNet, self).__init__()
        self.conv0 = nn.Conv2d(3, channel, kernel_size * 3, padding=4, padding_mode='replicate')
        
        self.convs = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size, padding=1, padding_mode='replicate'),
            nn.ReLU(),
            nn.Conv2d(channel, channel, kernel_size, padding=1, padding_mode='replicate'),
            nn.ReLU(),
            nn.Conv2d(channel, channel, kernel_size, padding=1, padding_mode='replicate'),
            nn.ReLU(),
            nn.Conv2d(channel, channel, kernel_size, padding=1, padding_mode='replicate'),
            nn.ReLU(),
            nn.Conv2d(channel, channel, kernel_size, padding=1, padding_mode='replicate'),
            nn.ReLU()
        )
        
        self.recon = nn.Conv2d(channel, 4, kernel_size, padding=1, padding_mode='replicate')
    
    def forward(self, x):
        feats0 = self.conv0(x)
        featss = self.convs(feats0)
        outs = self.recon(featss)
        R = torch.sigmoid(outs[:, 0:3, :, :])  # 反射图 (3通道)
        L = torch.sigmoid(outs[:, 3:4, :, :])  # 光照图 (1通道)
        return R, L

# 融合模块 - 修复版本
class FusionModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FusionModule, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        )
    
    def forward(self, features):
        # 在通道维度上拼接特征图
        fused = torch.cat(features, dim=1)
        return self.conv(fused)

# 教师扩散模型
class TeacherDiffusionModel(nn.Module):
    def __init__(self, in_channels=3, channels=64):
        super(TeacherDiffusionModel, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels*2, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels*2, channels*4, 3, stride=2, padding=1),
            nn.ReLU()
        )
        self.mid = nn.Sequential(
            nn.Conv2d(channels*4, channels*4, 3, padding=1),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(channels*4, channels*2, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(channels*2, channels, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, in_channels, 3, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.mid(x)
        x = self.decoder(x)
        return x

# 学生扩散模型
class StudentDiffusionModel(nn.Module):
    def __init__(self, in_channels=3, channels=32):
        super(StudentDiffusionModel, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, in_channels, 3, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.model(x)

# 状态空间模型模块
class SSMBlock(nn.Module):
    def __init__(self, dim, d_state=16, expand=2):
        super().__init__()
        self.dim = dim
        self.d_state = d_state
        self.expand = expand
        self.d_inner = int(expand * dim)
        
        self.in_proj = nn.Linear(dim, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=3,
            groups=self.d_inner,
            padding=2,
            padding_mode='replicate'
        )
        self.x_proj = nn.Linear(self.d_inner, d_state + d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.d_inner, d_state, bias=True)
        self.out_proj = nn.Linear(self.d_inner, dim, bias=False)
    
    def forward(self, x):
        B, L, C = x.shape
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)
        x = rearrange(x, 'b l c -> b c l')
        x = self.conv1d(x)[:, :, :L]
        x = rearrange(x, 'b c l -> b l c')
        x_dbl = self.x_proj(x)
        dt, A, B = torch.split(x_dbl, [self.d_state, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt))
        y = torch.sigmoid(A) * x + dt * B
        y = y * torch.sigmoid(z)
        y = self.out_proj(y)
        return y

# 多头注意力机制
class MultiHeadAttention(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
    
    def forward(self, x):
        B, L, C = x.shape
        qkv = self.qkv(x).reshape(B, L, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, L, C)
        x = self.proj(x)
        return x

# Mamba块
class MambaBlock(nn.Module):
    def __init__(self, dim, num_heads=8, d_state=16):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.ssm = SSMBlock(dim, d_state)
        self.norm2 = nn.LayerNorm(dim)
        self.attn = MultiHeadAttention(dim, num_heads)
    
    def forward(self, x):
        x = x + self.ssm(self.norm1(x))
        x = x + self.attn(self.norm2(x))
        return x

# U型网络
class MambaUNet(nn.Module):
    def __init__(self, in_channels=6, out_channels=3, base_dim=32, num_heads=4):
        super().__init__()
        # 编码器
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, base_dim, 3, padding=1),
            nn.ReLU()
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(base_dim, base_dim*2, 3, stride=2, padding=1),
            nn.ReLU()
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(base_dim*2, base_dim*4, 3, stride=2, padding=1),
            nn.ReLU()
        )
        
        # 中间Mamba块
        self.mamba_mid = MambaBlock(base_dim*4 * 16, num_heads)
        
        # 解码器
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(base_dim*4, base_dim*2, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU()
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(base_dim*2*2, base_dim, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU()
        )
        self.dec1 = nn.Conv2d(base_dim*2, out_channels, 3, padding=1)
    
    def forward(self, x):
        # 编码路径
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        
        # Mamba中间处理
        B, C, H, W = enc3.shape
        mamba_in = enc3.view(B, C, H*W).permute(0, 2, 1)
        mamba_out = self.mamba_mid(mamba_in)
        mamba_out = mamba_out.permute(0, 2, 1).view(B, C, H, W)
        
        # 解码路径
        dec3 = self.dec3(mamba_out)
        dec2 = self.dec2(torch.cat([dec3, enc2], dim=1))  # 添加跳跃连接
        dec1 = self.dec1(torch.cat([dec2, enc1], dim=1))  # 添加跳跃连接
        return torch.sigmoid(dec1)

# 完整系统 - 修复融合模块初始化
class MultiExposureEnhancementSystem(nn.Module):
    def __init__(self, num_exposures=5, use_distill=True):
        super().__init__()
        self.num_exposures = num_exposures
        self.use_distill = use_distill
        self.best_psnr = 0.0
        self.best_ssim = 0.0
        self.train_phase = "Decom"  # 默认从分解阶段开始
        
        # 初始化各模块
        self.DecomNet = DecomNet()
        # 修复：光照融合输入通道数 = 曝光数，反射融合输入通道数 = 3 * 曝光数
        self.LightFusion = FusionModule(num_exposures, 1)  # 输入通道数 = 曝光数量
        self.ReflectFusion = FusionModule(3 * num_exposures, 3)  # 输入通道数 = 3 * 曝光数量
        self.TeacherModel = TeacherDiffusionModel()
        self.StudentModel = StudentDiffusionModel()
        self.MambaUNet = MambaUNet()
        
        # 初始化冻结状态
        self._freeze_modules()
    
    def set_train_phase(self, phase):
        """显式设置训练阶段"""
        self.train_phase = phase
        self._freeze_modules()
    
    def _freeze_modules(self):
        """根据训练阶段冻结模块"""
        modules = {
            'DecomNet': self.DecomNet,
            'LightFusion': self.LightFusion,
            'ReflectFusion': self.ReflectFusion,
            'TeacherModel': self.TeacherModel,
            'StudentModel': self.StudentModel,
            'MambaUNet': self.MambaUNet
        }
        
        # 默认冻结所有模块
        for module in modules.values():
            for param in module.parameters():
                param.requires_grad = False
            module.eval()
        
        # 根据阶段解冻需要训练的模块
        if self.train_phase == "Decom":
            for param in self.DecomNet.parameters():
                param.requires_grad = True
            self.DecomNet.train()
        
        elif self.train_phase == "Diffusion":
            for param in self.StudentModel.parameters():
                param.requires_grad = True
            self.StudentModel.train()
            
            if not self.use_distill:
                for param in self.TeacherModel.parameters():
                    param.requires_grad = True
                self.TeacherModel.train()
        
        elif self.train_phase == "UNet":
            for param in self.MambaUNet.parameters():
                param.requires_grad = True
            self.MambaUNet.train()
        
        self._verify_freeze_state()
    
    def _verify_freeze_state(self):
        """验证冻结状态是否符合预期"""
        print(f"\n=== Freeze State Verification ({self.train_phase}) ===")
        modules = {
            'DecomNet': self.DecomNet,
            'LightFusion': self.LightFusion,
            'ReflectFusion': self.ReflectFusion,
            'TeacherModel': self.TeacherModel,
            'StudentModel': self.StudentModel,
            'MambaUNet': self.MambaUNet
        }
        
        for name, module in modules.items():
            status = "Trainable" if any(p.requires_grad for p in module.parameters()) else "Frozen"
            print(f"{name}: {status}")
        print("="*40)
    
    def forward(self, multi_exposure, normal_light):
        """
        multi_exposure: [B, num_exposures, 3, H, W]
        normal_light: [B, 3, H, W]
        """
        # 1. 分解每张曝光图像
        R_list = []
        L_list = []
        for i in range(self.num_exposures):
            R_i, L_i = self.DecomNet(multi_exposure[:, i, :, :, :])
            R_list.append(R_i)
            L_list.append(L_i)
        
        # 保存中间结果用于损失计算
        self.R_list = R_list
        self.L_list = L_list
        self.multi_exposure = multi_exposure
        
        # 2. 融合光照图和反射图
        L_fused = self.LightFusion(L_list)  # [B, 1, H, W]
        R_expo = self.ReflectFusion(R_list)  # [B, 3, H, W]
        
        # 3. 分解正常光图像
        R_normal, L_normal = self.DecomNet(normal_light)
        
        # 4. 生成图像P
        P = R_expo * L_fused.repeat(1, 3, 1, 1)  # [B, 3, H, W]
        
        # 5. 扩散模型处理
        with torch.set_grad_enabled(self.train_phase == "Diffusion"):
            teacher_output = self.TeacherModel(R_normal)
            student_output = self.StudentModel(R_expo)
        
        # 6. U-Net处理
        with torch.set_grad_enabled(self.train_phase == "UNet"):
            unet_input = torch.cat([P, student_output], dim=1)  # [B, 6, H, W]
            enhanced = self.MambaUNet(unet_input)  # [B, 3, H, W]
        
        # 计算损失
        self._calculate_losses(enhanced, normal_light, teacher_output, student_output)
        
        return enhanced
    
    def _calculate_losses(self, enhanced, target, teacher_output, student_output):
        # 重建损失
        self.recon_loss = F.l1_loss(enhanced, target)
        
        # 扩散模型损失
        self.diff_loss = F.mse_loss(student_output, teacher_output.detach())
        
        # 总损失
        if self.train_phase == "Decom":
            # 计算所有曝光图像的重建损失
            self.decom_loss = 0
            for i in range(self.num_exposures):
                recon = self.R_list[i] * self.L_list[i].repeat(1, 3, 1, 1)
                self.decom_loss += F.l1_loss(recon, self.multi_exposure[:, i])
            self.total_loss = self.decom_loss / self.num_exposures
            
        elif self.train_phase == "Diffusion":
            self.total_loss = self.diff_loss
            
        else:  # UNet阶段
            # 组合L1损失和SSIM损失
            ssim_loss = 1 - torch.mean(torch.tensor([
                calculate_ssim(enhanced[i:i+1], target[i:i+1]) 
                for i in range(enhanced.size(0))
            ]))
            self.total_loss = self.recon_loss + 0.5 * ssim_loss
    
    def save_model(self, path):
        state = {
            'decom': self.DecomNet.state_dict(),
            'light_fusion': self.LightFusion.state_dict(),
            'reflect_fusion': self.ReflectFusion.state_dict(),
            'teacher': self.TeacherModel.state_dict(),
            'student': self.StudentModel.state_dict(),
            'unet': self.MambaUNet.state_dict(),
            'phase': self.train_phase,
            'best_psnr': self.best_psnr
        }
        torch.save(state, path)
        print(f"Model saved to {path}")
    
    def load_model(self, path, device):
        state = torch.load(path, map_location=device)
        self.DecomNet.load_state_dict(state['decom'])
        self.LightFusion.load_state_dict(state['light_fusion'])
        self.ReflectFusion.load_state_dict(state['reflect_fusion'])
        self.TeacherModel.load_state_dict(state['teacher'])
        self.StudentModel.load_state_dict(state['student'])
        self.MambaUNet.load_state_dict(state['unet'])
        self.train_phase = state.get('phase', 'Decom')
        self.best_psnr = state.get('best_psnr', 0.0)
        self._freeze_modules()
        print(f"Model loaded from {path}")

# 计算PSNR
def calculate_psnr(img1, img2, max_val=1.0):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    psnr = 20 * log10(max_val / torch.sqrt(mse).item())
    return psnr

# 计算SSIM
def calculate_ssim(img1, img2, data_range=1.0):
    img1_np = img1.permute(1, 2, 0).cpu().numpy()
    img2_np = img2.permute(1, 2, 0).cpu().numpy()
    
    if img1_np.shape[2] == 1:
        img1_np = img1_np[:, :, 0]
        img2_np = img2_np[:, :, 0]
    
    if img1_np.ndim == 3:
        ssim_value = ssim(img1_np, img2_np, data_range=data_range, channel_axis=2)
    else:
        ssim_value = ssim(img1_np, img2_np, data_range=data_range)
    return ssim_value

# 训练一个阶段
def train_phase(system, optimizer, train_loader, val_loader, device, 
                ckpt_dir, phase_name, num_epochs, start_epoch=0):
    best_psnr = system.best_psnr
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=True
    )
    
    for epoch in range(start_epoch, num_epochs):
        # 训练循环
        system.train()
        epoch_loss = 0
        start_time = time.time()
        
        for batch_idx, (multi_exposure, normal_light) in enumerate(train_loader):
            multi_exposure = multi_exposure.to(device)
            normal_light = normal_light.to(device)
            
            # 前向传播
            enhanced = system(multi_exposure, normal_light)
            
            # 反向传播
            optimizer.zero_grad()
            system.total_loss.backward()
            
            # 梯度裁剪防止爆炸
            torch.nn.utils.clip_grad_norm_(system.parameters(), max_norm=1.0)
            
            optimizer.step()
            epoch_loss += system.total_loss.item()
            
            # 打印进度
            if (batch_idx + 1) % 10 == 0:
                avg_loss = epoch_loss / (batch_idx + 1)
                elapsed = time.time() - start_time
                print(f"Epoch [{epoch+1}/{num_epochs}] Batch [{batch_idx+1}/{len(train_loader)}] "
                      f"Loss: {avg_loss:.6f} | Time: {elapsed:.2f}s")
        
        # 验证
        val_psnr, val_ssim = evaluate(system, val_loader, device)
        avg_loss = epoch_loss / len(train_loader)
        elapsed = time.time() - start_time
        
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.6f} | "
              f"Val PSNR: {val_psnr:.2f}, SSIM: {val_ssim:.4f} | "
              f"Time: {elapsed:.2f}s")
        
        # 更新学习率
        scheduler.step(val_psnr)
        
        # 保存检查点
        if (epoch + 1) % 5 == 0:
            ckpt_path = os.path.join(ckpt_dir, f"{phase_name}_epoch{epoch+1}.pth")
            system.save_model(ckpt_path)
        
        # 保存最佳模型
        if val_psnr > best_psnr:
            best_psnr = val_psnr
            system.best_psnr = best_psnr
            best_ckpt = os.path.join(ckpt_dir, f"best_{phase_name}_psnr{val_psnr:.2f}.pth")
            system.save_model(best_ckpt)
            print(f"New best {phase_name} model! PSNR: {val_psnr:.2f}")

# 评估函数
def evaluate(system, val_loader, device):
    system.eval()
    total_psnr = 0
    total_ssim = 0
    count = 0
    
    with torch.no_grad():
        for multi_exposure, normal_light in val_loader:
            multi_exposure = multi_exposure.to(device)
            normal_light = normal_light.to(device)
            
            # 前向传播
            enhanced = system(multi_exposure, normal_light)
            
            # 计算指标
            psnr = calculate_psnr(enhanced, normal_light)
            ssim_val = calculate_ssim(enhanced, normal_light)
            
            total_psnr += psnr
            total_ssim += ssim_val
            count += 1
    
    return total_psnr / count, total_ssim / count

# 完整训练函数
def train_full_system(args):
    # 创建输出目录
    os.makedirs(args.ckpt_dir, exist_ok=True)
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    print(f"Using device: {device}")
    
    # 创建数据集
    train_dataset = MultiExposureDataset(
        base_dir=args.data_dir,
        exposure_paths=args.exposure_paths,
        gt_path=args.gt_path,
        num_exposures=args.num_exposures,
        patch_size=args.patch_size,
        phase='train'
    )
    
    val_dataset = MultiExposureDataset(
        base_dir=args.val_dir,
        exposure_paths=args.val_exposure_paths,
        gt_path=args.val_gt_path,
        num_exposures=args.num_exposures,
        patch_size=args.patch_size,
        phase='val'
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=1, 
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # 初始化系统
    system = MultiExposureEnhancementSystem(
        num_exposures=args.num_exposures,
        use_distill=args.use_distill
    ).to(device)
    
    # 如果指定了恢复路径，加载模型
    start_phase = "Decom"
    if args.resume:
        system.load_model(args.resume, device)
        start_phase = system.train_phase
    
    # 定义各阶段参数
    phases = ["Decom", "Diffusion", "UNet"]
    phase_epochs = {
        "Decom": args.decom_epochs,
        "Diffusion": args.diff_epochs,
        "UNet": args.unet_epochs
    }
    phase_lrs = {
        "Decom": args.decom_lr,
        "Diffusion": args.diff_lr,
        "UNet": args.unet_lr
    }
    
    # 按顺序训练各阶段
    for phase in phases:
        # 如果从中间阶段恢复，跳过已完成阶段
        if phases.index(phase) < phases.index(start_phase):
            print(f"Skipping completed phase: {phase}")
            continue
        
        print(f"\n{'='*60}")
        print(f"Starting {phase} phase training")
        print(f"{'='*60}")
        
        # 设置当前训练阶段
        system.set_train_phase(phase)  # 使用新方法设置阶段
        
        # 准备优化器
        if phase == "Decom":
            optimizer = optim.Adam(system.DecomNet.parameters(), lr=phase_lrs[phase])
        elif phase == "Diffusion":
            params = list(system.StudentModel.parameters())
            if not args.use_distill:
                params += list(system.TeacherModel.parameters())
            optimizer = optim.Adam(params, lr=phase_lrs[phase])
        else:  # "UNet"
            optimizer = optim.Adam(system.MambaUNet.parameters(), lr=phase_lrs[phase])
        
        # 训练当前阶段
        train_phase(
            system, optimizer, train_loader, val_loader, device,
            args.ckpt_dir, phase, phase_epochs[phase]
        )
    
    print("\nTraining completed!")

# 预测函数
def predict(args):
    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    
    # 初始化系统
    system = MultiExposureEnhancementSystem(
        num_exposures=args.num_exposures,
        use_distill=args.use_distill
    ).to(device)
    
    # 加载模型
    system.load_model(args.model_path, device)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 处理每个场景
    scene_dirs = sorted(os.listdir(args.input_dir))
    
    for scene in scene_dirs:
        scene_path = os.path.join(args.input_dir, scene)
        if not os.path.isdir(scene_path):
            continue
        
        print(f"Processing scene: {scene}")
        
        # 加载多曝光图像
        expo_images = []
        for i in range(args.num_exposures):
            img_path = os.path.join(scene_path, f'{i}.jpg')
            if not os.path.exists(img_path):
                print(f"Warning: Missing exposure image {img_path}")
                continue
            
            img = Image.open(img_path).convert('RGB')
            img = np.array(img, dtype=np.float32) / 255.0
            img = np.transpose(img, (2, 0, 1))
            expo_images.append(img)
        
        if len(expo_images) != args.num_exposures:
            print(f"Skipping scene {scene}: incomplete exposures")
            continue
        
        # 创建输入张量
        multi_exposure = np.stack(expo_images, axis=0)
        multi_exposure = torch.tensor(multi_exposure).unsqueeze(0).to(device)
        dummy_normal = torch.zeros(1, 3, *multi_exposure.shape[-2:]).to(device)
        
        # 预测
        with torch.no_grad():
            enhanced = system(multi_exposure, dummy_normal)
        
        # 保存结果
        enhanced_img = enhanced.squeeze(0).permute(1, 2, 0).cpu().numpy()
        enhanced_img = (np.clip(enhanced_img, 0, 1) * 255).astype(np.uint8)
        output_path = os.path.join(args.output_dir, f'{scene}_enhanced.jpg')
        Image.fromarray(enhanced_img).save(output_path)
        print(f"Saved enhanced image: {output_path}")

# 自定义数据集类 - 支持非连续编号图像
class MultiExposureDataset(Dataset):
    def __init__(self, base_dir, exposure_paths, gt_path, num_exposures=5, patch_size=256, phase='train'):
        """
        base_dir: 数据集基础路径
        exposure_paths: 多曝光图像路径列表
        gt_path: 正常光图像路径
        num_exposures: 曝光图像数量
        patch_size: 裁剪尺寸
        phase: 训练或验证阶段
        """
        self.base_dir = base_dir
        self.exposure_paths = exposure_paths
        self.gt_path = gt_path
        self.num_exposures = num_exposures
        self.patch_size = patch_size
        self.phase = phase
        
        # 获取所有图像文件名（不连续编号）
        self._collect_image_files()
        
        # 数据增强
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(90),
        ]) if phase == 'train' else None
    
    def _collect_image_files(self):
        """收集所有可用的图像文件名"""
        # 获取GT文件夹中的所有图像文件
        gt_dir = os.path.join(self.base_dir, self.gt_path)
        gt_files = glob.glob(os.path.join(gt_dir, '*.jpg')) + glob.glob(os.path.join(gt_dir, '*.png'))
        self.gt_files = [os.path.basename(f) for f in gt_files]
        
        # 检查每个曝光路径下的图像文件
        self.exposure_files = []
        for path in self.exposure_paths:
            expo_dir = os.path.join(self.base_dir, path)
            expo_files = glob.glob(os.path.join(expo_dir, '*.jpg')) + glob.glob(os.path.join(expo_dir, '*.png'))
            expo_files = [os.path.basename(f) for f in expo_files]
            self.exposure_files.append(expo_files)
        
        # 确保每个曝光级别都有相同的图像
        common_files = set(self.gt_files)
        for files in self.exposure_files:
            common_files = common_files.intersection(set(files))
        
        # 只保留所有路径都有的图像
        self.valid_files = sorted(list(common_files))
        
        # 训练/验证分割
        if self.phase == 'train':
            self.valid_files = self.valid_files[:int(0.8 * len(self.valid_files))]
        else:
            self.valid_files = self.valid_files[int(0.8 * len(self.valid_files)):]
        
        print(f"Found {len(self.valid_files)} valid images for {self.phase} phase")
    
    def __len__(self):
        return len(self.valid_files)
    
    def __getitem__(self, idx):
        filename = self.valid_files[idx]
        
        # 加载多曝光图像
        expo_images = []
        for i, path in enumerate(self.exposure_paths):
            img_path = os.path.join(self.base_dir, path, filename)
            img = Image.open(img_path).convert('RGB')
            expo_images.append(img)
        
        # 加载正常光图像
        gt_path = os.path.join(self.base_dir, self.gt_path, filename)
        normal_img = Image.open(gt_path).convert('RGB')
        
        # 随机裁剪
        W, H = expo_images[0].size
        if self.phase == 'train':
            x = random.randint(0, H - self.patch_size)
            y = random.randint(0, W - self.patch_size)
        else:
            x = (H - self.patch_size) // 2
            y = (W - self.patch_size) // 2
        
        # 裁剪并应用数据增强
        cropped_expo = []
        for img in expo_images:
            img = img.crop((y, x, y + self.patch_size, x + self.patch_size))
            if self.transform:
                img = self.transform(img)
            img = np.array(img, dtype=np.float32) / 255.0
            img = np.transpose(img, (2, 0, 1))
            cropped_expo.append(img)
        
        normal_img = normal_img.crop((y, x, y + self.patch_size, x + self.patch_size))
        if self.transform:
            normal_img = self.transform(normal_img)
        normal_img = np.array(normal_img, dtype=np.float32) / 255.0
        normal_img = np.transpose(normal_img, (2, 0, 1))
        
        # 转换为张量
        multi_exposure = np.stack(cropped_expo, axis=0)
        return torch.tensor(multi_exposure), torch.tensor(normal_img)

# 主函数
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Exposure Image Enhancement System")
    
    # 通用参数
    parser.add_argument("--mode", type=str, default="train", choices=["train", "predict"],
                        help="Run mode: train or predict")
    parser.add_argument("--cpu", action="store_true", help="Force CPU mode")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    # 训练数据集路径配置
    parser.add_argument("--data_dir", type=str, default="/home/xd508/zyx/LOL/fixed_gamma/", 
                        help="Base directory for training data")
    parser.add_argument("--exposure_paths", nargs='+', 
                        default=["low/gamma_0_50", "low/gamma_0_80", "middle/gamma_1_00", "high/gamma_1_50", "high/gamma_2_00"],
                        help="List of exposure paths relative to data_dir")
    parser.add_argument("--gt_path", type=str, default="/home/xd508/zyx/LOL/fixed_gamma/GT/", 
                        help="Path to ground truth images relative to data_dir")
    
    # 验证数据集路径配置
    parser.add_argument("--val_dir", type=str, default="/home/xd508/zyx/LOL/eval15/", 
                        help="Base directory for validation data")
    parser.add_argument("--val_exposure_paths", nargs='+', 
                        default=["low/gamma_0_50", "low/gamma_0_80", "middle/gamma_1_00", "high/gamma_1_50", "high/gamma_2_00"],
                        help="List of exposure paths relative to val_dir")
    parser.add_argument("--val_gt_path", type=str, default="/home/xd508/zyx/LOL/fixed_gamma/GT/", 
                        help="Path to ground truth images relative to val_dir")
    
    # 训练参数
    parser.add_argument("--ckpt_dir", type=str, default="checkpoints", 
                        help="Directory to save checkpoints")
    parser.add_argument("--num_exposures", type=int, default=5, 
                        help="Number of exposure images per scene")
    parser.add_argument("--batch_size", type=int, default=4, 
                        help="Training batch size")
    parser.add_argument("--patch_size", type=int, default=256, 
                        help="Patch size for training")
    parser.add_argument("--num_workers", type=int, default=4, 
                        help="Number of data loader workers")
    parser.add_argument("--use_distill", action="store_true", 
                        help="Use knowledge distillation for diffusion models")
    parser.add_argument("--resume", type=str, default="", 
                        help="Path to checkpoint to resume training")
    
    # 阶段特定参数
    parser.add_argument("--decom_epochs", type=int, default=30, 
                        help="Number of epochs for decomposition phase")
    parser.add_argument("--diff_epochs", type=int, default=40, 
                        help="Number of epochs for diffusion phase")
    parser.add_argument("--unet_epochs", type=int, default=60, 
                        help="Number of epochs for UNet phase")
    parser.add_argument("--decom_lr", type=float, default=1e-4, 
                        help="Learning rate for decomposition phase")
    parser.add_argument("--diff_lr", type=float, default=5e-5, 
                        help="Learning rate for diffusion phase")
    parser.add_argument("--unet_lr", type=float, default=3e-5, 
                        help="Learning rate for UNet phase")
    
    # 预测参数
    parser.add_argument("--model_path", type=str, default="", 
                        help="Path to model for prediction")
    parser.add_argument("--input_dir", type=str, default="/home/xd508/zyx/LOL/eval15/", 
                        help="Input directory for prediction")
    parser.add_argument("--output_dir", type=str, default="result", 
                        help="Output directory for prediction results")
    
    args = parser.parse_args()
    
    # 打印配置信息
    print("\n" + "="*50)
    print("Multi-Exposure Image Enhancement Configuration")
    print("="*50)
    print(f"Mode: {args.mode}")
    if args.mode == "train":
        print(f"Training Data Directory: {args.data_dir}")
        print(f"Exposure Paths: {args.exposure_paths}")
        print(f"GT Path: {args.gt_path}")
        print(f"Validation Data Directory: {args.val_dir}")
        print(f"Validation Exposure Paths: {args.val_exposure_paths}")
        print(f"Validation GT Path: {args.val_gt_path}")
    elif args.mode == "predict":
        print(f"Input Directory: {args.input_dir}")
        print(f"Output Directory: {args.output_dir}")
    print("="*50 + "\n")
    
    if args.mode == "train":
        train_full_system(args)
    elif args.mode == "predict":
        if not args.model_path:
            raise ValueError("Model path must be specified for prediction mode")
        predict(args)
