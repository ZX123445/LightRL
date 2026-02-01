import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16
from einops import rearrange


class DepthwiseSeparableConv(nn.Module):
    """深度可分离卷积"""

    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size,
                                   padding=kernel_size // 2, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


class LiteRNet(nn.Module):
    """轻量级Retinex分解网络"""

    def __init__(self, in_channels=3, base_channels=32):
        super().__init__()
        # 编码器
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.ReLU()
        )
        self.enc2 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(base_channels, base_channels * 2, 3, padding=1),
            nn.ReLU(),
            DepthwiseSeparableConv(base_channels * 2, base_channels * 2)
        )
        self.enc3 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(base_channels * 2, base_channels * 4, 3, padding=1),
            nn.ReLU(),
            DepthwiseSeparableConv(base_channels * 4, base_channels * 4)
        )

        # 解码器（光照和反射共享）
        self.dec3 = nn.Sequential(
            DepthwiseSeparableConv(base_channels * 4, base_channels * 4),
            nn.ReLU()
        )
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec2 = nn.Sequential(
            DepthwiseSeparableConv(base_channels * 6, base_channels * 2),
            nn.ReLU()
        )
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec1 = nn.Sequential(
            DepthwiseSeparableConv(base_channels * 3, base_channels),
            nn.ReLU()
        )

        # 输出头
        self.illum_head = nn.Conv2d(base_channels, 1, 3, padding=1)
        self.refl_head = nn.Conv2d(base_channels, 3, 3, padding=1)

    def forward(self, x):
        # 输入: [batch, num_imgs, C, H, W]
        batch_size, num_imgs, C, H, W = x.shape
        x = x.view(-1, C, H, W)  # 合并批次和图像维度

        # 编码
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)

        # 解码
        d3 = self.dec3(e3)
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        # 输出分解结果
        illum = torch.sigmoid(self.illum_head(d1))  # [0,1]范围
        refl = torch.sigmoid(self.refl_head(d1))  # [0,1]范围

        # 恢复原始维度
        illum = illum.view(batch_size, num_imgs, 1, H, W)
        refl = refl.view(batch_size, num_imgs, 3, H, W)

        return illum, refl


class LFuseNet(nn.Module):
    """光照图融合网络"""

    def __init__(self, num_inputs, base_channels=32):
        super().__init__()
        self.num_inputs = num_inputs
        self.init_conv = nn.Sequential(
            nn.Conv2d(1, base_channels, 3, padding=1),
            nn.ReLU()
        )
        self.attn_conv = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(base_channels, num_inputs, 1),
            nn.Softmax(dim=1)
        )

    def forward(self, illum_maps):
        # illum_maps: [B, N, 1, H, W]
        batch_size, num_imgs, _, H, W = illum_maps.shape

        # 处理每张光照图
        feats = []
        for i in range(num_imgs):
            img = illum_maps[:, i]  # [B, 1, H, W]
            feat = self.init_conv(img)  # [B, C, H, W]
            weights = self.attn_conv(feat)  # [B, N, H, W]
            feats.append(weights.unsqueeze(1))  # [B, 1, N, H, W]

        # 组合所有权重
        weights = torch.cat(feats, dim=1)  # [B, N, N, H, W]
        weights = weights.mean(dim=1)  # [B, N, H, W] 取平均权重

        # 加权融合
        fused_illum = torch.sum(illum_maps.squeeze(2) * weights, dim=1, keepdim=True)
        return fused_illum  # [B, 1, H, W]


class MobileViTBlock(nn.Module):
    """MobileViT模块 - 高效Transformer结构"""

    def __init__(self, dim, heads=4, expansion=4):
        super().__init__()
        inner_dim = dim * expansion

        # 局部特征提取
        self.local_mlp = nn.Sequential(
            nn.Conv2d(dim, inner_dim, 1),
            nn.GELU(),
            nn.Conv2d(inner_dim, dim, 1)
        )

        # 全局注意力
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)

        # 门控融合
        self.gate = nn.Conv2d(dim * 2, dim, 1)

    def forward(self, x):
        # 局部路径
        local_feat = self.local_mlp(x)

        # 全局路径
        _, _, H, W = x.shape
        global_feat = rearrange(x, 'b c h w -> b (h w) c')
        global_feat = self.norm(global_feat)
        global_feat, _ = self.attn(global_feat, global_feat, global_feat)
        global_feat = rearrange(global_feat, 'b (h w) c -> b c h w', h=H, w=W)

        # 门控融合
        gate_map = torch.sigmoid(self.gate(torch.cat([local_feat, global_feat], dim=1)))
        return gate_map * local_feat + (1 - gate_map) * global_feat


class SobelFilterRGB(nn.Module):
    """Sobel边缘检测器"""

    def __init__(self):
        super().__init__()
        # 定义 Sobel X 核（水平边缘）
        self.register_buffer('sobel_kernel_x',
                             torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3))
        # 定义 Sobel Y 核（垂直边缘）
        self.register_buffer('sobel_kernel_y',
                             torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3))

    def forward(self, x):
        assert x.size(1) == 3, "Input must have 3 channels (RGB)"

        # 初始化输出
        sobel_x = torch.zeros_like(x)
        sobel_y = torch.zeros_like(x)

        # 对每个通道单独计算 Sobel
        for c in range(x.size(1)):
            sobel_x[:, c:c + 1, :, :] = F.conv2d(x[:, c:c + 1, :, :], self.sobel_kernel_x, padding=1)
            sobel_y[:, c:c + 1, :, :] = F.conv2d(x[:, c:c + 1, :, :], self.sobel_kernel_y, padding=1)

        return sobel_x, sobel_y


class SSMEnhancer(nn.Module):
    """端到端低光增强网络"""

    def __init__(self, in_channels=9, base_channels=32):
        super().__init__()
        # 输入: [low_img, enhanced_img, sobel_edge]
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1),
            nn.ReLU()
        )
        self.sobel = SobelFilterRGB()

        # MobileViT主干
        self.stage1 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, 3, stride=2, padding=1),
            MobileViTBlock(base_channels * 2)
        )
        self.stage2 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels * 4, 3, stride=2, padding=1),
            MobileViTBlock(base_channels * 4)
        )
        self.stage3 = MobileViTBlock(base_channels * 4)

        # 上采样路径
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(base_channels * 4, base_channels * 2, 3, padding=1),
            nn.ReLU()
        )
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(base_channels * 4, base_channels, 3, padding=1),
            nn.ReLU()
        )

        # 输出头
        self.illum_head = nn.Conv2d(base_channels, 1, 3, padding=1)
        self.output_head = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(base_channels, 3, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, low_img, enhanced_img):
        # 计算边缘图
        sobel_x, sobel_y = self.sobel(low_img)
        edge_map = torch.sqrt(sobel_x ** 2 + sobel_y ** 2 + 1e-6)

        # 构建输入 [B, 9, H, W]
        x = torch.cat([low_img, enhanced_img, edge_map], dim=1)

        # 主干网络
        x0 = self.stem(x)  # [B, 32, H, W]
        x1 = self.stage1(x0)  # [B, 64, H/2, W/2]
        x2 = self.stage2(x1)  # [B, 128, H/4, W/4]
        x3 = self.stage3(x2)  # [B, 128, H/4, W/4]

        # 上采样
        x2_up = self.up2(x3)  # [B, 64, H/2, W/2]
        x1_up = self.up1(torch.cat([x2_up, x1], dim=1))  # [B, 32, H, W]

        # 输出
        output_img = self.output_head(torch.cat([x1_up, x0], dim=1))
        illum_pred = torch.sigmoid(self.illum_head(x1_up))
        return output_img, illum_pred


class NonLocalBlock(nn.Module):
    """非局部注意力去噪块"""

    def __init__(self, channels):
        super().__init__()
        self.conv_theta = nn.Conv2d(channels, channels // 2, 1)
        self.conv_phi = nn.Conv2d(channels, channels // 2, 1)
        self.conv_g = nn.Conv2d(channels, channels // 2, 1)
        self.conv_out = nn.Conv2d(channels // 2, channels, 1)

    def forward(self, x):
        batch, C, H, W = x.shape
        # 计算theta, phi, g
        theta = self.conv_theta(x)  # [B, C//2, H, W]
        phi = F.max_pool2d(self.conv_phi(x), 2)  # [B, C//2, H//2, W//2]
        g = F.max_pool2d(self.conv_g(x), 2)  # [B, C//2, H//2, W//2]

        # 展平特征
        theta_flat = rearrange(theta, 'b c h w -> b (h w) c')
        phi_flat = rearrange(phi, 'b c h w -> b c (h w)')
        g_flat = rearrange(g, 'b c h w -> b (h w) c')

        # 注意力计算
        attn = torch.matmul(theta_flat, phi_flat)  # [B, (H*W), (H//2*W//2)]
        attn = F.softmax(attn, dim=-1)

        # 加权平均
        y = torch.matmul(attn, g_flat)  # [B, (H*W), C//2]
        y = rearrange(y, 'b (h w) c -> b c h w', h=H, w=W)

        # 输出
        return self.conv_out(y) + x


class LowLightEnhancer(nn.Module):
    """完整的低光增强系统（改进版）"""

    def __init__(self, gamma_values=[0.3, 0.5, 0.8]):
        super().__init__()
        self.gamma_values = gamma_values
        self.num_exposures = len(gamma_values) + 1  # 包含原始图像

        # 存储gamma=0.5的索引
        self.gamma05_index = gamma_values.index(0.5) + 1 if 0.5 in gamma_values else 1

        self.retinex_net = LiteRNet()
        self.fuse_net = LFuseNet(num_inputs=self.num_exposures)
        self.enhancer = SSMEnhancer()
        self.denoiser = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            NonLocalBlock(32),
            nn.Conv2d(32, 3, 3, padding=1)
        )

    def gamma_correction(self, img, gamma):
        """Gamma校正函数"""
        return torch.clamp(img ** gamma, 0, 1)

    def forward(self, low_img):
        # 1. 生成多曝光图像
        exposures = [low_img]  # 原始图像
        gamma05_img = None

        for i, gamma in enumerate(self.gamma_values):
            corrected_img = self.gamma_correction(low_img, gamma)
            exposures.append(corrected_img)

            # 保存gamma=0.3的图像
            if gamma == 0.3:
                gamma03_img = corrected_img
                
            # 保存gamma=0.5的图像
            if gamma == 0.5:
                gamma05_img = corrected_im5
                
            # 保存gamma=0.8的图像
            if gamma == 0.8:
                gamma08_img = corrected_img


        exposures = torch.stack(exposures, dim=1)  # [B, N, 3, H, W]

        # 2. Retinex分解
        illum_maps, refl_maps = self.retinex_net(exposures)

        # 3. 光照融合
        fused_illum = self.fuse_net(illum_maps)  # [B, 1, H, W]

        # 4. 初步增强
        enhanced = torch.clamp(low_img / (fused_illum + 1e-6), 0, 1) ** 0.7

        # 5. 联合去噪
        denoised = self.denoiser(enhanced)

        # 6. SSM精细增强
        output_img, illum_pred = self.enhancer(low_img, denoised)
        final_output = 0.025*output_img +1*gamma03_img+ 1*gamma05_img + 1*gamma08_img

        return {
            'output': final_output,  # 最终输出
            'illum_pred': illum_pred,
            'fused_illum': fused_illum,
            'enhanced': enhanced,
            'gamma05_img': gamma05_img,
            'ssm_output': output_img  # SSM增强器的原始输出
        }


class EnhancementLoss(nn.Module):
    """改进的损失函数，考虑最终融合输出"""

    def __init__(self, alpha=0.7, beta=0.2, gamma=0.05, delta=0.05):
        super().__init__()
        self.alpha = alpha  # 重建损失权重
        self.beta = beta  # 感知损失权重
        self.gamma = gamma  # 光照一致性损失权重
        self.delta = delta  # TV损失权重

        # VGG网络用于感知损失
        self.vgg = vgg16(pretrained=True).features[:16].eval()
        for param in self.vgg.parameters():
            param.requires_grad = False

        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()

    def forward(self, results, targets):
        # 最终输出
        final_output = results['output']

        # 原始SSM输出（可选）
        ssm_output = results['ssm_output']

        # 光照预测
        illum_pred = results['illum_pred']
        fused_illum = results['fused_illum']

        # 1. 最终输出的重建损失
        l1_final = self.l1_loss(final_output, targets)

        # 2. 最终输出的感知损失
        vgg_final = self.vgg(final_output)
        vgg_target = self.vgg(targets)
        percep_final = self.mse_loss(vgg_final, vgg_target)

        # 3. 光照一致性损失
        illum_loss = self.l1_loss(illum_pred, fused_illum)

        # 4. 总变分损失（减少噪声）
        tv_loss = (
                torch.mean(torch.abs(final_output[:, :, :, :-1] - final_output[:, :, :, 1:])) +
                torch.mean(torch.abs(final_output[:, :, :-1, :] - final_output[:, :, 1:, :])))

            # 5. 可选：SSM输出的重建损失
        l1_ssm = self.l1_loss(ssm_output, targets)

        # 总损失
        total_loss = (
                self.alpha * (l1_final + 0.3 * l1_ssm) +
                self.beta * percep_final +
                self.gamma * illum_loss +
                self.delta * tv_loss
        )

        return total_loss, {
            'l1_final': l1_final,
            'percep_final': percep_final,
            'illum_loss': illum_loss,
            'tv_loss': tv_loss,
            'l1_ssm': l1_ssm
        }


# 训练函数示例
def train_model(model, train_loader, val_loader, epochs=100, lr=1e-4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    criterion = EnhancementLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)

    # 训练循环
    for epoch in range(epochs):
        model.train()
        for low_imgs, gt_imgs in train_loader:
            low_imgs, gt_imgs = low_imgs.to(device), gt_imgs.to(device)

            # 前向传播
            results = model(low_imgs)

            # 计算损失
            loss, loss_dict = criterion(results, gt_imgs)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # 验证阶段
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for low_imgs, gt_imgs in val_loader:
                low_imgs, gt_imgs = low_imgs.to(device), gt_imgs.to(device)
                results = model(low_imgs)
                loss, _ = criterion(results, gt_imgs)
                val_loss += loss.item()

        print(
            f'Epoch {epoch + 1}/{epochs} | Train Loss: {loss.item():.4f} | Val Loss: {val_loss / len(val_loader):.4f}')

    # 保存模型
    torch.save(model.state_dict(), 'lowlight_enhancer_final.pth')
