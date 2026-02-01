import os
import time
import random
from glob import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable
import numpy as np
from PIL import Image
from math import log10
from skimage.metrics import structural_similarity as ssim
import copy

gamma_dir = 'output/fixed_gamma_exposures/'


def calculate_psnr(img1, img2, max_val=1.0):
    """计算PSNR（峰值信噪比）"""
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    psnr = 20 * log10(max_val / torch.sqrt(mse).item())
    return psnr


def calculate_ssim(img1, img2, data_range=1.0):
    """计算SSIM（结构相似性）"""
    img1_np = img1.squeeze(0).permute(1, 2, 0).cpu().numpy()
    img2_np = img2.squeeze(0).permute(1, 2, 0).cpu().numpy()

    if img1_np.shape[2] == 1:
        img1_np = img1_np[:, :, 0]
        img2_np = img2_np[:, :, 0]

    if img1_np.ndim == 3:
        ssim_value = ssim(img1_np, img2_np, data_range=data_range, channel_axis=2)
    else:
        ssim_value = ssim(img1_np, img2_np, data_range=data_range)
    return ssim_value


class DecomNet(nn.Module):
    def __init__(self, channel=64, kernel_size=3, num_exposures=5):
        super(DecomNet, self).__init__()
        self.num_exposures = num_exposures
        input_channels = 3 * num_exposures + 1

        self.net1_conv0 = nn.Conv2d(input_channels, channel, kernel_size * 3,
                                    padding=4, padding_mode='replicate')

        self.net1_convs = nn.Sequential(
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

        self.net1_recon = nn.Conv2d(channel, 4, kernel_size, padding=1, padding_mode='replicate')

    def forward(self, input_imgs):
        B, C, H, W = input_imgs.shape
        reshaped = input_imgs.view(B, self.num_exposures, 3, H, W)
        max_channel = torch.max(reshaped, dim=1)[0]
        max_channel = torch.max(max_channel, dim=1, keepdim=True)[0]
        input_with_max = torch.cat((input_imgs, max_channel), dim=1)
        feats0 = self.net1_conv0(input_with_max)
        featss = self.net1_convs(feats0)
        outs = self.net1_recon(featss)
        R = torch.sigmoid(outs[:, 0:3, :, :])
        L = torch.sigmoid(outs[:, 3:4, :, :])
        return R, L


class TeacherRelightNet(nn.Module):
    def __init__(self, channel=128, kernel_size=3):
        super(TeacherRelightNet, self).__init__()
        self.relu = nn.ReLU()

        self.net2_conv0_1 = nn.Conv2d(4, channel, kernel_size, padding=1, padding_mode='replicate')

        self.net2_conv1_1 = nn.Conv2d(channel, channel, kernel_size, stride=2, padding=1, padding_mode='replicate')
        self.net2_conv1_2 = nn.Conv2d(channel, channel, kernel_size, stride=2, padding=1, padding_mode='replicate')
        self.net2_conv1_3 = nn.Conv2d(channel, channel, kernel_size, stride=2, padding=1, padding_mode='replicate')
        self.net2_conv1_4 = nn.Conv2d(channel, channel, kernel_size, stride=2, padding=1, padding_mode='replicate')

        self.net2_deconv1_1 = nn.Conv2d(channel * 2, channel, kernel_size, padding=1, padding_mode='replicate')
        self.net2_deconv1_2 = nn.Conv2d(channel * 2, channel, kernel_size, padding=1, padding_mode='replicate')
        self.net2_deconv1_3 = nn.Conv2d(channel * 2, channel, kernel_size, padding=1, padding_mode='replicate')
        self.net2_deconv1_4 = nn.Conv2d(channel * 2, channel, kernel_size, padding=1, padding_mode='replicate')

        self.net2_fusion = nn.Conv2d(channel * 4, channel, kernel_size=1, padding=1, padding_mode='replicate')
        self.net2_output = nn.Conv2d(channel, 1, kernel_size=3, padding=0)

    def forward(self, input_L, input_R):
        input_img = torch.cat((input_R, input_L), dim=1)
        out0 = self.net2_conv0_1(input_img)
        out1 = self.relu(self.net2_conv1_1(out0))
        out2 = self.relu(self.net2_conv1_2(out1))
        out3 = self.relu(self.net2_conv1_3(out2))
        out4 = self.relu(self.net2_conv1_4(out3))

        out4_up = F.interpolate(out4, size=out3.shape[2:])
        deconv1 = self.relu(self.net2_deconv1_1(torch.cat((out4_up, out3), dim=1)))
        deconv1_up = F.interpolate(deconv1, size=out2.shape[2:])
        deconv2 = self.relu(self.net2_deconv1_2(torch.cat((deconv1_up, out2), dim=1)))
        deconv2_up = F.interpolate(deconv2, size=out1.shape[2:])
        deconv3 = self.relu(self.net2_deconv1_3(torch.cat((deconv2_up, out1), dim=1)))
        deconv3_up = F.interpolate(deconv3, size=out0.shape[2:])
        deconv4 = self.relu(self.net2_deconv1_4(torch.cat((deconv3_up, out0), dim=1)))

        deconv1_rs = F.interpolate(deconv1, size=input_R.shape[2:])
        deconv2_rs = F.interpolate(deconv2, size=input_R.shape[2:])
        deconv3_rs = F.interpolate(deconv3, size=input_R.shape[2:])
        deconv4_rs = F.interpolate(deconv4, size=input_R.shape[2:])
        feats_all = torch.cat((deconv1_rs, deconv2_rs, deconv3_rs, deconv4_rs), dim=1)

        feats_fus = self.net2_fusion(feats_all)
        output = self.net2_output(feats_fus)
        return output


class RelightNet(nn.Module):
    def __init__(self, channel=64, kernel_size=3, use_distill=False):
        super(RelightNet, self).__init__()
        self.use_distill = use_distill
        self.relu = nn.ReLU()

        self.net2_conv0_1 = nn.Conv2d(4, channel, kernel_size, padding=1, padding_mode='replicate')
        self.net2_conv1_1 = nn.Conv2d(channel, channel, kernel_size, stride=2, padding=1, padding_mode='replicate')
        self.net2_conv1_2 = nn.Conv2d(channel, channel, kernel_size, stride=2, padding=1, padding_mode='replicate')
        self.net2_conv1_3 = nn.Conv2d(channel, channel, kernel_size, stride=2, padding=1, padding_mode='replicate')

        self.net2_deconv1_1 = nn.Conv2d(channel * 2, channel, kernel_size, padding=1, padding_mode='replicate')
        self.net2_deconv1_2 = nn.Conv2d(channel * 2, channel, kernel_size, padding=1, padding_mode='replicate')
        self.net2_deconv1_3 = nn.Conv2d(channel * 2, channel, kernel_size, padding=1, padding_mode='replicate')

        self.net2_fusion = nn.Conv2d(channel * 3, channel, kernel_size=1, padding=1, padding_mode='replicate')
        self.net2_output = nn.Conv2d(channel, 1, kernel_size=3, padding=0)

        if self.use_distill:
            self.feature_adapters = nn.ModuleDict({
                'out1': nn.Conv2d(channel, 128, 1),
                'out2': nn.Conv2d(channel, 128, 1),
                'out3': nn.Conv2d(channel, 128, 1),
                'feats_fus': nn.Conv2d(channel, 128, 1)
            })

    def forward(self, input_L, input_R, return_features=False):
        features = {}
        input_img = torch.cat((input_R, input_L), dim=1)
        out0 = self.net2_conv0_1(input_img)
        out1 = self.relu(self.net2_conv1_1(out0))
        out2 = self.relu(self.net2_conv1_2(out1))
        out3 = self.relu(self.net2_conv1_3(out2))

        features['out1'] = out1
        features['out2'] = out2
        features['out3'] = out3

        out3_up = F.interpolate(out3, size=out2.shape[2:])
        deconv1 = self.relu(self.net2_deconv1_1(torch.cat((out3_up, out2), dim=1)))
        deconv1_up = F.interpolate(deconv1, size=out1.shape[2:])
        deconv2 = self.relu(self.net2_deconv1_2(torch.cat((deconv1_up, out1), dim=1)))
        deconv2_up = F.interpolate(deconv2, size=out0.shape[2:])
        deconv3 = self.relu(self.net2_deconv1_3(torch.cat((deconv2_up, out0), dim=1)))

        features['deconv1'] = deconv1
        features['deconv2'] = deconv2
        features['deconv3'] = deconv3

        deconv1_rs = F.interpolate(deconv1, size=input_R.shape[2:])
        deconv2_rs = F.interpolate(deconv2, size=input_R.shape[2:])
        feats_all = torch.cat((deconv1_rs, deconv2_rs, deconv3), dim=1)
        feats_fus = self.net2_fusion(feats_all)

        features['feats_fus'] = feats_fus

        output = self.net2_output(feats_fus)

        if return_features:
            return output, features
        return output


class RetinexNet(nn.Module):
    def __init__(self, num_exposures=5, use_distill=False):
        super(RetinexNet, self).__init__()
        self.num_exposures = num_exposures
        self.use_distill = use_distill
        self.best_psnr = 0.0
        self.best_ssim = 0.0
        self.best_weights = None
        self.train_phase = "Decom"  # 默认为分解阶段

        self.DecomNet = DecomNet(num_exposures=num_exposures)
        self.RelightNet = RelightNet(use_distill=use_distill)

        if use_distill:
            self.TeacherRelightNet = TeacherRelightNet()
            for param in self.TeacherRelightNet.parameters():
                param.requires_grad = False

    def forward(self, input_low, input_high):
        if isinstance(input_low, np.ndarray):
            input_low = torch.from_numpy(input_low).float()
        if isinstance(input_high, np.ndarray):
            input_high = torch.from_numpy(input_high).float()

        if torch.cuda.is_available():
            input_low = input_low.cuda()
            input_high = input_high.cuda()

        # 根据训练阶段决定是否计算分解网络的梯度
        decom_grad_enabled = self.train_phase == "Decom"
        with torch.set_grad_enabled(decom_grad_enabled):
            # 分解网络
            R_low, I_low = self.DecomNet(input_low)
            input_high_rep = input_high.repeat(1, self.num_exposures, 1, 1)
            R_high, I_high = self.DecomNet(input_high_rep)

        # 重光照网络
        if self.use_distill and self.train_phase == "Relight":
            I_delta, student_features = self.RelightNet(I_high, R_low, return_features=True)
        else:
            I_delta = self.RelightNet(I_high, R_low)

        # 教师模型前向传播
        if self.use_distill and self.train_phase == "Relight":
            with torch.no_grad():
                teacher_output = self.TeacherRelightNet(I_high, R_low)

        # 准备输出
        I_low_3 = torch.cat((I_low, I_low, I_low), dim=1)
        I_high_3 = torch.cat((I_high, I_high, I_high), dim=1)
        I_delta_3 = torch.cat((I_delta, I_delta, I_delta), dim=1)

        # 计算损失
        self.recon_loss_low = F.l1_loss(R_low * I_low_3, input_low[:, :3, :, :])
        self.recon_loss_high = F.l1_loss(R_high * I_high_3, input_high)
        self.recon_loss_mutal_low = F.l1_loss(R_high * I_low_3, input_low[:, :3, :, :])
        self.recon_loss_mutal_high = F.l1_loss(R_low * I_high_3, input_high)
        self.equal_R_loss = F.l1_loss(R_low, R_high.detach())
        self.relight_loss = F.l1_loss(R_low * I_delta_3, input_high)

        self.Ismooth_loss_low = self.smooth(I_low, R_low)
        self.Ismooth_loss_high = self.smooth(I_high, R_high)
        self.Ismooth_loss_delta = self.smooth(I_delta, R_low)

        # 分解损失
        self.loss_Decom = (self.recon_loss_low +
                           self.recon_loss_high +
                           0.001 * self.recon_loss_mutal_low +
                           0.001 * self.recon_loss_mutal_high +
                           0.1 * self.Ismooth_loss_low +
                           0.1 * self.Ismooth_loss_high +
                           0.01 * self.equal_R_loss)

        self.light_consistency = F.mse_loss(I_low.mean(dim=[2, 3]), I_high.mean(dim=[2, 3]))
        self.loss_Decom += 0.05 * self.light_consistency

        # 重光照损失
        self.loss_Relight = (self.relight_loss + 3 * self.Ismooth_loss_delta)

        # 知识蒸馏损失
        self.distill_loss = 0
        if self.use_distill and self.train_phase == "Relight":
            output_distill_loss = F.mse_loss(I_delta, teacher_output)

            feature_distill_loss = 0
            with torch.no_grad():
                _, teacher_features = self.TeacherRelightNet(I_high, R_low, return_features=True)

            for key in ['out1', 'out2', 'out3', 'feats_fus']:
                adapted_feature = self.RelightNet.feature_adapters[key](student_features[key])
                feature_distill_loss += F.mse_loss(adapted_feature, teacher_features[key])

            self.distill_loss = 0.7 * output_distill_loss + 0.3 * feature_distill_loss
            self.loss_Relight += 0.5 * self.distill_loss

        # 保存输出
        self.output_R_low = R_low.detach().cpu()
        self.output_I_low = I_low_3.detach().cpu()
        self.output_I_delta = I_delta_3.detach().cpu()
        self.output_S = R_low.detach().cpu() * I_delta_3.detach().cpu()

        # 保存教师输出
        if self.use_distill and self.train_phase == "Relight":
            self.teacher_output = teacher_output.detach().cpu()

    def gradient(self, input_tensor, direction):
        smooth_kernel_x = torch.FloatTensor([[0, 0], [-1, 1]]).view((1, 1, 2, 2))
        smooth_kernel_y = torch.transpose(smooth_kernel_x, 2, 3)
        if torch.cuda.is_available():
            smooth_kernel_x = smooth_kernel_x.cuda()
            smooth_kernel_y = smooth_kernel_y.cuda()
        if direction == "x":
            kernel = smooth_kernel_x
        elif direction == "y":
            kernel = smooth_kernel_y
        grad_out = torch.abs(F.conv2d(input_tensor, kernel, stride=1, padding=1))
        return grad_out

    def ave_gradient(self, input_tensor, direction):
        return F.avg_pool2d(self.gradient(input_tensor, direction),
                            kernel_size=3, stride=1, padding=1)

    def smooth(self, input_I, input_R):
        input_R = 0.299 * input_R[:, 0, :, :] + 0.587 * input_R[:, 1, :, :] + 0.114 * input_R[:, 2, :, :]
        input_R = torch.unsqueeze(input_R, dim=1)
        return torch.mean(self.gradient(input_I, "x") * torch.exp(-10 * self.ave_gradient(input_R, "x")) +
                          self.gradient(input_I, "y") * torch.exp(-10 * self.ave_gradient(input_R, "y")))

    def calculate_metrics(self, enhanced, gt):
        """计算PSNR和SSIM指标"""
        psnr = calculate_psnr(enhanced, gt)
        ssim_val = calculate_ssim(enhanced, gt)
        return psnr, ssim_val

    def evaluate(self, epoch_num, eval_multi_names, eval_high_names, vis_dir, train_phase, num_exposures=5):
        """修改后的评估函数，接受多曝光序列和对应的正常光图像路径"""
        print(f"Evaluating for phase {train_phase} / epoch {epoch_num}...")
        total_psnr = 0.0
        total_ssim = 0.0
        count = 0

        # 遍历评估场景
        for idx in range(len(eval_multi_names)):
            multi_exposure = eval_multi_names[idx]  # 多曝光序列
            gt_path = eval_high_names[idx]  # 对应的正常光图像路径

            # 加载多曝光图像
            exposure_images = []
            for exp_path in multi_exposure:
                try:
                    img = Image.open(exp_path)
                    img = np.array(img, dtype="float32") / 255.0
                    img = np.transpose(img, (2, 0, 1))
                    exposure_images.append(img)
                except:
                    print(f"无法加载图像: {exp_path}")
                    continue

            # 确保有足够的曝光图像
            if len(exposure_images) < num_exposures:
                print(f"场景 {idx} 只有 {len(exposure_images)} 张图像，跳过")
                continue

            # 创建多曝光输入
            input_multi = np.concatenate(exposure_images, axis=0)
            input_multi = np.expand_dims(input_multi, axis=0)
            placeholder_high = np.expand_dims(exposure_images[0], axis=0)

            # 加载真实正常光图像
            try:
                gt_img = Image.open(gt_path)
                gt_img = np.array(gt_img, dtype="float32") / 255.0
                gt_img = np.transpose(gt_img, (2, 0, 1))
                gt_tensor = torch.from_numpy(np.expand_dims(gt_img, axis=0)).float()
            except:
                print(f"无法加载真实图像: {gt_path}")
                continue

            # 前向传播
            self.forward(input_multi, placeholder_high)

            if train_phase == "Decom":
                result_1 = self.output_R_low
                result_2 = self.output_I_low
                input = np.squeeze(input_multi[:, :3, :, :])
                result_1 = np.squeeze(result_1)
                result_2 = np.squeeze(result_2)
                cat_image = np.concatenate([input, result_1, result_2], axis=2)

            elif train_phase == "Relight" or train_phase == "Teacher":
                result_1 = self.output_R_low
                result_2 = self.output_I_low
                result_3 = self.output_I_delta
                result_4 = self.output_S
                input = np.squeeze(input_multi[:, :3, :, :])
                result_1 = np.squeeze(result_1)
                result_2 = np.squeeze(result_2)
                result_3 = np.squeeze(result_3)
                result_4 = np.squeeze(result_4)
                cat_image = np.concatenate([input, result_1, result_2, result_3, result_4], axis=2)

                # 计算指标
                enhanced_tensor = torch.from_numpy(np.expand_dims(result_4, axis=0)).float()
                psnr, ssim_val = self.calculate_metrics(enhanced_tensor, gt_tensor)
                total_psnr += psnr
                total_ssim += ssim_val
                count += 1

                if self.use_distill and train_phase == "Relight":
                    teacher_out = np.squeeze(self.teacher_output.numpy())
                    teacher_out = np.repeat(teacher_out, 3, axis=0)
                    cat_image = np.concatenate([cat_image, teacher_out], axis=2)

            # 保存可视化结果
            cat_image = np.transpose(cat_image, (1, 2, 0))
            im = Image.fromarray(np.clip(cat_image * 255.0, 0, 255.0).astype('uint8'))
            filename = os.path.basename(gt_path).split('.')[0]
            filepath = os.path.join(vis_dir, f'eval_{train_phase}_{filename}_ep{epoch_num}.jpg')
            im.save(filepath)
            print(f"Saved visualization: {filepath}")

        if count > 0:
            avg_psnr = total_psnr / count
            avg_ssim = total_ssim / count
            print(f"Validation Metrics - PSNR: {avg_psnr:.4f}, SSIM: {avg_ssim:.4f}")
            return avg_psnr, avg_ssim
        return 0.0, 0.0

    def save(self, iter_num, ckpt_dir):
        save_dir = os.path.join(ckpt_dir, self.train_phase)
        os.makedirs(save_dir, exist_ok=True)
        save_name = os.path.join(save_dir, f'{iter_num}.tar')

        if self.train_phase == 'Decom':
            torch.save(self.DecomNet.state_dict(), save_name)
        elif self.train_phase == 'Relight':
            state = {
                'model': self.RelightNet.state_dict(),
                'teacher': self.TeacherRelightNet.state_dict() if self.use_distill else None
            }
            torch.save(state, save_name)
        print(f"Model saved: {save_name}")

    def save_best(self, ckpt_dir, metric_name, metric_value):
        """保存最优模型"""
        save_dir = os.path.join(ckpt_dir, self.train_phase, 'best')
        os.makedirs(save_dir, exist_ok=True)
        save_name = os.path.join(save_dir, f'best_{metric_name}_{metric_value:.4f}.pth')

        if self.train_phase == 'Decom':
            torch.save(self.DecomNet.state_dict(), save_name)
        elif self.train_phase == 'Relight':
            state = {
                'model': self.RelightNet.state_dict(),
                'teacher': self.TeacherRelightNet.state_dict() if self.use_distill else None
            }
            torch.save(state, save_name)
        print(f"Best model saved: {save_name}")

    def load(self, ckpt_dir):
        load_dir = os.path.join(ckpt_dir, self.train_phase)
        if not os.path.exists(load_dir):
            print(f"Checkpoint directory not found: {load_dir}")
            return False, 0

        ckpts = glob(os.path.join(load_dir, '*.tar'))
        if not ckpts:
            print(f"No checkpoints found in {load_dir}")
            return False, 0

        ckpts.sort(key=os.path.getmtime)
        ckpt_path = ckpts[-1]
        global_step = int(os.path.basename(ckpt_path).split('.')[0])

        try:
            ckpt = torch.load(ckpt_path)
            if self.train_phase == 'Decom':
                # 检查输入通道数是否匹配
                if 'net1_conv0.weight' in ckpt:
                    saved_weight = ckpt['net1_conv0.weight']
                    current_weight = self.DecomNet.net1_conv0.weight

                    if saved_weight.shape[1] != current_weight.shape[1]:
                        print(
                            f"输入通道数不匹配: 预训练模型为{saved_weight.shape[1]}, 当前模型为{current_weight.shape[1]}")
                        print("由于num_exposures参数变化，将跳过预训练权重加载")
                        return False, 0

                self.DecomNet.load_state_dict(ckpt)
                # 确保分解网络在重光照阶段冻结
                if self.train_phase == 'Relight':
                    for param in self.DecomNet.parameters():
                        param.requires_grad = False

            elif self.train_phase == 'Relight':
                if 'model' in ckpt:
                    self.RelightNet.load_state_dict(ckpt['model'])
                if self.use_distill and 'teacher' in ckpt and ckpt['teacher'] is not None:
                    self.TeacherRelightNet.load_state_dict(ckpt['teacher'])

                # 确保分解网络冻结
                for param in self.DecomNet.parameters():
                    param.requires_grad = False

            print(f"Model loaded from {ckpt_path}")
            return True, global_step
        except Exception as e:
            print(f"Error loading model from {ckpt_path}: {str(e)}")
            return False, 0

    def train(self,
              train_multi_names,  # 训练多曝光序列
              train_high_names,  # 训练正常光图像
              eval_multi_names,  # 评估多曝光序列
              eval_high_names,  # 评估正常光图像
              batch_size,
              patch_size, epoch,
              lr,
              vis_dir,
              ckpt_dir,
              eval_every_epoch,
              train_phase):

        self.train_phase = train_phase
        load_model_status, global_step = self.load(ckpt_dir)
        if load_model_status:
            iter_num = global_step
            start_epoch = global_step // (len(train_multi_names) // batch_size)
            start_step = global_step % (len(train_multi_names) // batch_size)
            print("Model restore success!")
        else:
            iter_num = 0
            start_epoch = 0
            start_step = 0
            print("No pretrained model to restore!")

        print("Start training for phase %s, with start epoch %d start iter %d : " %
              (self.train_phase, start_epoch, iter_num))

        start_time = time.time()
        image_id = 0

        # 初始化优化器
        if train_phase == "Decom":
            optimizer = optim.Adam(self.DecomNet.parameters(), lr=lr[0], betas=(0.9, 0.999))
            # 确保分解网络梯度开启
            for param in self.DecomNet.parameters():
                param.requires_grad = True
        else:
            # 冻结分解网络参数
            for param in self.DecomNet.parameters():
                param.requires_grad = False

            # 仅优化重光照网络
            optimizer = optim.Adam(self.RelightNet.parameters(), lr=lr[0], betas=(0.9, 0.999))

            # 如果使用蒸馏，确保教师模型冻结
            if self.use_distill:
                for param in self.TeacherRelightNet.parameters():
                    param.requires_grad = False

        # 设置分解网络为评估模式（关闭Dropout等）
        if train_phase != "Decom":
            self.DecomNet.eval()

        # 初始化最优指标
        best_psnr = 0.0
        best_ssim = 0.0

        for epoch_idx in range(start_epoch, epoch):
            # 更新学习率
            current_lr = lr[epoch_idx]
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr

            # 打乱数据
            combined = list(zip(train_multi_names, train_high_names))
            random.shuffle(combined)
            train_multi_names, train_high_names = zip(*combined)

            numBatch = len(train_multi_names) // batch_size

            for batch_id in range(start_step, numBatch):
                # 创建批处理数据
                batch_input_multi = np.zeros((batch_size, 3 * self.num_exposures, patch_size, patch_size),
                                             dtype="float32")
                batch_input_high = np.zeros((batch_size, 3, patch_size, patch_size), dtype="float32")

                for patch_id in range(batch_size):
                    scene_exposures = train_multi_names[image_id]
                    normal_light_path = train_high_names[image_id]

                    # 加载多曝光图像
                    exposure_images = []
                    for exp_path in scene_exposures:
                        try:
                            img = Image.open(exp_path)
                            img = np.array(img, dtype='float32') / 255.0
                            if img.ndim == 2:  # 灰度图处理
                                img = np.stack([img] * 3, axis=-1)
                            h, w, _ = img.shape
                            x = random.randint(0, h - patch_size)
                            y = random.randint(0, w - patch_size)
                            img = img[x:x + patch_size, y:y + patch_size, :]

                            # 数据增强
                            if random.random() < 0.5:
                                img = np.flipud(img)
                            if random.random() < 0.5:
                                img = np.fliplr(img)
                            rot_type = random.randint(1, 4)
                            if random.random() < 0.5:
                                img = np.rot90(img, rot_type)

                            img = np.transpose(img, (2, 0, 1))
                            exposure_images.append(img)
                        except Exception as e:
                            print(f"Error loading image {exp_path}: {str(e)}")
                            continue

                    if not exposure_images:
                        continue

                    multi_exposure = np.concatenate(exposure_images, axis=0)
                    batch_input_multi[patch_id] = multi_exposure

                    # 加载正常光图像
                    try:
                        high_img = Image.open(normal_light_path)
                        high_img = np.array(high_img, dtype='float32') / 255.0
                        if high_img.ndim == 2:  # 灰度图处理
                            high_img = np.stack([high_img] * 3, axis=-1)
                        h, w, _ = high_img.shape
                        x = random.randint(0, h - patch_size)
                        y = random.randint(0, w - patch_size)
                        high_img = high_img[x:x + patch_size, y:y + patch_size, :]
                        high_img = np.transpose(high_img, (2, 0, 1))
                        batch_input_high[patch_id] = high_img
                    except Exception as e:
                        print(f"Error loading normal light image {normal_light_path}: {str(e)}")
                        continue

                    image_id = (image_id + 1) % len(train_multi_names)

                # 前向传播
                self.forward(batch_input_multi, batch_input_high)

                # 计算损失
                if train_phase == "Decom":
                    loss = self.loss_Decom
                else:
                    loss = self.loss_Relight

                # 反向传播和优化
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # 打印日志
                elapsed = time.time() - start_time
                log_str = "%s Epoch: [%2d/%2d] [%4d/%4d] time: %4.2f, lr: %.6f, loss: %.6f" % \
                          (train_phase, epoch_idx + 1, epoch, batch_id + 1, numBatch, elapsed, current_lr, loss.item())

                if self.use_distill and train_phase == "Relight":
                    log_str += f", distill: {self.distill_loss.item():.6f}"

                print(log_str)

                iter_num += 1

            # 评估和保存
            if (epoch_idx + 1) % eval_every_epoch == 0:
                print(f"Evaluating model at epoch {epoch_idx + 1}...")
                psnr, ssim_val = self.evaluate(epoch_idx + 1,
                                               eval_multi_names,
                                               eval_high_names,
                                               vis_dir=vis_dir,
                                               train_phase=train_phase,
                                               num_exposures=self.num_exposures)

                # 保存当前模型
                self.save(iter_num, ckpt_dir)

                # 检查PSNR和SSIM是否改善
                psnr_improved = False
                ssim_improved = False

                if psnr > best_psnr:
                    best_psnr = psnr
                    psnr_improved = True
                    print(f"New best PSNR: {best_psnr:.4f}")

                if ssim_val > best_ssim:
                    best_ssim = ssim_val
                    ssim_improved = True
                    print(f"New best SSIM: {best_ssim:.4f}")

                # 只有当PSNR和SSIM都改善时才保存模型
                if psnr_improved and ssim_improved:
                    self.save_best(ckpt_dir, 'psnr_ssim', best_psnr)
                    print(f"New best model for both PSNR and SSIM: PSNR={best_psnr:.4f}, SSIM={best_ssim:.4f}")

        print("Finished training for phase %s." % train_phase)
        print(f"Best PSNR: {best_psnr:.4f}, Best SSIM: {best_ssim:.4f}")

    def predict(self,
                test_multi_data,
                res_dir,
                ckpt_dir,
                num_exposures=5):
        self.train_phase = 'Decom'
        load_model_status, _ = self.load(ckpt_dir)
        if load_model_status:
            print(self.train_phase, "  : Model restore success!")
        else:
            print("No pretrained model to restore!")
            raise Exception

        self.train_phase = 'Relight'
        load_model_status, _ = self.load(ckpt_dir)
        if load_model_status:
            print(self.train_phase, ": Model restore success!")
        else:
            print("No pretrained model to restore!")
            raise Exception

        # 确保分解网络冻结
        for param in self.DecomNet.parameters():
            param.requires_grad = False
        self.DecomNet.eval()

        save_R_L = False
        os.makedirs(res_dir, exist_ok=True)

        for scene_paths in test_multi_data:
            exposure_images = []
            for exp_path in scene_paths:
                img = Image.open(exp_path)
                img = np.array(img, dtype="float32") / 255.0
                img = np.transpose(img, (2, 0, 1))
                exposure_images.append(img)

            input_multi = np.concatenate(exposure_images, axis=0)
            input_multi = np.expand_dims(input_multi, axis=0)
            placeholder_high = np.expand_dims(exposure_images[0], axis=0)

            self.forward(input_multi, placeholder_high)
            result_1 = self.output_R_low
            result_2 = self.output_I_low
            result_3 = self.output_I_delta
            result_4 = self.output_S

            input_img = np.squeeze(input_multi[:, :3, :, :])
            result_1 = np.squeeze(result_1)
            result_2 = np.squeeze(result_2)
            result_3 = np.squeeze(result_3)
            result_4 = np.squeeze(result_4)

            if save_R_L:
                cat_image = np.concatenate([input_img, result_1, result_2, result_3, result_4], axis=2)
            else:
                cat_image = np.concatenate([input_img, result_4], axis=2)

            cat_image = np.transpose(cat_image, (1, 2, 0))
            im = Image.fromarray(np.clip(cat_image * 255.0, 0, 255.0).astype('uint8'))
            scene_name = os.path.basename(scene_paths[0]).split('_')[0]
            filepath = os.path.join(res_dir, f'{scene_name}_enhanced.jpg')
            im.save(filepath)
            print(f"Enhanced image saved: {filepath}")