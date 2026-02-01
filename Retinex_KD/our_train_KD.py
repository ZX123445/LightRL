import os
import argparse
import random
import logging
from glob import glob
import numpy as np
import torch
from torch import optim
import torch.nn.functional as F
from Retinex_KD import RetinexNet  # 确保文件名匹配
from PIL import Image
import time
import shutil

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def parse_args():
    parser = argparse.ArgumentParser(description='Multi-Exposure RetinexNet Training')
    parser.add_argument('--gpu_id', dest='gpu_id', default="0", help='GPU ID (-1 for CPU)')
    parser.add_argument('--epochs', dest='epochs', type=int, default=100, help='number of total epochs')
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=16, help='number of samples in one batch')
    parser.add_argument('--patch_size', dest='patch_size', type=int, default=112, help='patch size for training')
    parser.add_argument('--num_exposures', dest='num_exposures', type=int, default=5,
                        help='number of exposures to use per scene')
    parser.add_argument('--lr', dest='lr', type=float, default=0.001, help='initial learning rate')
    parser.add_argument('--data_dir', dest='data_dir',
                        default=r'H:/研究生论文/10_image_enhanced/LOLdataset/our485/',
                        help='directory storing the original data')
    parser.add_argument('--gamma_dir', dest='gamma_data_dir',
                        default=r'C:/Users/Lenovo/Desktop/gamma/output/fixed_gamma_exposures/',
                        help='directory storing the gamma-corrected exposures')
    parser.add_argument('--ckpt_dir', dest='ckpt_dir', default=r'./ckpts/', help='directory for checkpoints')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--use_distill', action='store_true', help='whether to use knowledge distillation')
    parser.add_argument('--teacher_epochs', type=int, default=500, help='number of epochs to pre-train teacher model')
    parser.add_argument('--distill_weight', type=float, default=0.7, help='weight for distillation loss')
    parser.add_argument('--teacher_lr', type=float, default=0.0001, help='learning rate for teacher model')
    parser.add_argument('--fresh_start', action='store_true',
                        help='start training from scratch, ignoring existing checkpoints')
    parser.add_argument('--eval_with_train_data', action='store_true',
                        help='use training data for evaluation instead of separate validation set')
    return parser.parse_args()


def get_scene_pairs(gamma_dir, data_dir, num_exposures=5, eval_with_train_data=False):
    """创建训练对：每个场景的多曝光输入和对应的正常光图像"""
    logger.info(f"开始从目录 {gamma_dir} 加载多曝光数据，从 {data_dir} 加载正常光图像...")

    # 规范化路径
    gamma_dir = os.path.normpath(gamma_dir)
    data_dir = os.path.normpath(data_dir)

    # 查找正常光图像目录
    high_dirs = [
        os.path.join(data_dir, 'high'),
        os.path.join(data_dir, 'high_png'),
        os.path.join(data_dir, 'normal'),
        os.path.join(data_dir, 'normal_light')
    ]

    high_dir = None
    for dir_path in high_dirs:
        if os.path.exists(dir_path):
            high_dir = dir_path
            break

    if high_dir is None:
        logger.error("未找到正常光图像目录！请检查以下路径:")
        for dir_path in high_dirs:
            logger.error(f" - {dir_path}")
        return [], [], [], []

    logger.info(f"使用正常光图像目录: {high_dir}")

    # 收集所有多曝光图像
    all_exposures = []
    exp_levels = ['low', 'middle', 'high']
    for exp_level in exp_levels:
        exp_dir = os.path.join(gamma_dir, exp_level)
        if not os.path.exists(exp_dir):
            logger.warning(f"曝光级别目录不存在: {exp_dir}")
            continue

        # 递归搜索图像文件
        exp_paths = glob(os.path.join(exp_dir, '**', '*.png'), recursive=True) + \
                    glob(os.path.join(exp_dir, '**', '*.jpg'), recursive=True) + \
                    glob(os.path.join(exp_dir, '**', '*.jpeg'), recursive=True)

        if exp_paths:
            logger.info(f"在 {exp_dir} 中找到 {len(exp_paths)} 张图像")
            all_exposures.extend(exp_paths)
        else:
            logger.warning(f"在 {exp_dir} 中未找到图像文件")

    if not all_exposures:
        logger.error("错误：未找到任何多曝光图像！")
        return [], [], [], []

    logger.info(f"共找到 {len(all_exposures)} 张多曝光图像")

    # 提取场景ID - 改进匹配逻辑
    scene_dict = {}
    for path in all_exposures:
        filename = os.path.basename(path)
        # 尝试多种可能的命名方式
        if '_' in filename:
            # 处理类似 "scene1_0.5.png" 的格式
            scene_id = filename.split('_')[0]
        elif '-' in filename:
            # 处理类似 "scene1-05.jpg" 的格式
            scene_id = filename.split('-')[0]
        else:
            # 默认使用文件名前缀
            scene_id = filename.split('.')[0][:6]

        if scene_id not in scene_dict:
            scene_dict[scene_id] = []
        scene_dict[scene_id].append(path)

    scene_ids = list(scene_dict.keys())
    logger.info(f"共找到 {len(scene_ids)} 个场景")

    # 收集正常光图像
    high_images = glob(os.path.join(high_dir, '*.png')) + \
                  glob(os.path.join(high_dir, '*.jpg')) + \
                  glob(os.path.join(high_dir, '*.jpeg'))

    if not high_images:
        logger.error(f"在 {high_dir} 中未找到任何正常光图像")
        return [], [], [], []

    logger.info(f"共找到 {len(high_images)} 张正常光图像")

    # 创建场景ID到正常光图像的映射
    high_dict = {}
    for path in high_images:
        filename = os.path.basename(path)
        # 匹配逻辑与多曝光图像一致
        if '_' in filename:
            scene_id = filename.split('_')[0]
        elif '-' in filename:
            scene_id = filename.split('-')[0]
        else:
            scene_id = filename.split('.')[0][:6]
        high_dict[scene_id] = path

    # 匹配多曝光和正常光图像
    train_multi_names = []
    train_high_names = []
    eval_multi_names = []  # 修改：存储多曝光序列
    eval_high_names = []  # 修改：存储对应的正常光图像

    # 分离训练集和验证集 (80% 训练, 20% 验证)
    all_scenes = list(scene_dict.items())
    random.shuffle(all_scenes)

    if eval_with_train_data:
        # 使用训练数据进行评估
        logger.info("评估模式：使用训练数据进行评估")
        train_scenes = all_scenes
    else:
        # 常规分离
        split_idx = int(len(all_scenes) * 0.8)
        train_scenes = all_scenes[:split_idx]
        eval_scenes = all_scenes[split_idx:]

    # 处理训练集
    for scene_id, exp_paths in train_scenes:
        if len(exp_paths) < num_exposures:
            logger.debug(f"场景 {scene_id} 只有 {len(exp_paths)} 张多曝光图像，需要 {num_exposures} 张，跳过")
            continue

        if scene_id in high_dict:
            selected_exposures = random.sample(exp_paths, num_exposures)
            train_multi_names.append(selected_exposures)
            train_high_names.append(high_dict[scene_id])
        else:
            # 尝试模糊匹配
            matched = False
            for key in high_dict:
                # 场景ID前4个字符匹配
                if key.startswith(scene_id[:4]):
                    selected_exposures = random.sample(exp_paths, num_exposures)
                    train_multi_names.append(selected_exposures)
                    train_high_names.append(high_dict[key])
                    matched = True
                    logger.info(f"模糊匹配: {scene_id} -> {key}")
                    break
            if not matched:
                logger.debug(f"场景 {scene_id} 未找到匹配的正常光图像，跳过")

    # 处理验证集（如果不使用训练数据进行评估）
    if not eval_with_train_data:
        for scene_id, exp_paths in eval_scenes:
            if len(exp_paths) < num_exposures:
                continue

            if scene_id in high_dict:
                # 选择多曝光序列
                selected_exposures = random.sample(exp_paths, num_exposures)
                eval_multi_names.append(selected_exposures)
                eval_high_names.append(high_dict[scene_id])
            else:
                # 尝试模糊匹配
                for key in high_dict:
                    if key.startswith(scene_id[:4]):
                        selected_exposures = random.sample(exp_paths, num_exposures)
                        eval_multi_names.append(selected_exposures)
                        eval_high_names.append(high_dict[key])
                        break

    if not train_multi_names or not train_high_names:
        logger.error("错误：没有找到有效的训练场景！")
        return [], [], [], []

    if not eval_with_train_data and (not eval_multi_names or not eval_high_names):
        logger.warning("警告：验证集为空，将使用训练数据进行验证")
        eval_multi_names = train_multi_names
        eval_high_names = train_high_names

    logger.info(f"训练集: {len(train_multi_names)} 个场景")

    if eval_with_train_data:
        logger.info(f"评估集: {len(train_multi_names)} 个场景（使用训练数据）")
        return train_multi_names, train_high_names, train_multi_names, train_high_names
    else:
        logger.info(f"验证集: {len(eval_multi_names)} 个场景")
        return train_multi_names, train_high_names, eval_multi_names, eval_high_names


def train_teacher_model(model, train_multi_names, train_high_names, eval_multi_names, eval_high_names, args):
    """单独训练教师模型"""
    logger.info("开始教师模型训练...")
    model.train_phase = 'Teacher'

    # 确保分解网络冻结
    for param in model.DecomNet.parameters():
        param.requires_grad = False
    model.DecomNet.eval()

    teacher_optim = optim.Adam(model.TeacherRelightNet.parameters(), lr=args.teacher_lr, betas=(0.9, 0.999))

    # 教师模型学习率调度
    teacher_lr = args.teacher_lr * np.ones([args.teacher_epochs])
    teacher_lr[int(args.teacher_epochs * 0.6):] = args.teacher_lr / 10.0

    numBatch = max(1, len(train_multi_names) // args.batch_size)

    # 训练教师模型
    for epoch_idx in range(args.teacher_epochs):
        # 更新学习率
        for param_group in teacher_optim.param_groups:
            param_group['lr'] = teacher_lr[epoch_idx]

        # 打乱数据
        combined = list(zip(train_multi_names, train_high_names))
        random.shuffle(combined)
        shuffled_multi, shuffled_high = zip(*combined)

        image_id = 0
        total_loss = 0

        for batch_id in range(numBatch):
            batch_input_multi = np.zeros((args.batch_size, 3 * args.num_exposures, args.patch_size, args.patch_size),
                                         dtype="float32")
            batch_input_high = np.zeros((args.batch_size, 3, args.patch_size, args.patch_size), dtype="float32")

            for patch_id in range(args.batch_size):
                if image_id >= len(shuffled_multi):
                    image_id = 0

                scene_exposures = shuffled_multi[image_id]
                normal_light_path = shuffled_high[image_id]

                # 加载多曝光图像
                exposure_images = []
                for exp_path in scene_exposures:
                    try:
                        img = Image.open(exp_path)
                        img = np.array(img, dtype='float32') / 255.0
                        if img.ndim == 2:  # 灰度图处理
                            img = np.stack([img] * 3, axis=-1)
                        h, w, _ = img.shape
                        x = random.randint(0, h - args.patch_size)
                        y = random.randint(0, w - args.patch_size)
                        img = img[x:x + args.patch_size, y:y + args.patch_size, :]

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
                        logger.error(f"加载图像 {exp_path} 出错: {str(e)}")
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
                    x = random.randint(0, h - args.patch_size)
                    y = random.randint(0, w - args.patch_size)
                    high_img = high_img[x:x + args.patch_size, y:y + args.patch_size, :]
                    high_img = np.transpose(high_img, (2, 0, 1))
                    batch_input_high[patch_id] = high_img
                except Exception as e:
                    logger.error(f"加载正常光图像 {normal_light_path} 出错: {str(e)}")
                    continue

                image_id += 1

            # 转换为tensor并移动到设备
            batch_input_multi = torch.from_numpy(batch_input_multi).float().cuda()
            batch_input_high = torch.from_numpy(batch_input_high).float().cuda()

            # 分解网络前向传播（不计算梯度）
            with torch.no_grad():
                R_low, I_low = model.DecomNet(batch_input_multi)
                R_high, I_high = model.DecomNet(batch_input_high.repeat(1, args.num_exposures, 1, 1))

            # 教师模型前向传播
            teacher_output = model.TeacherRelightNet(R_high, I_low)

            # 计算教师损失
            I_teacher_3 = torch.cat((teacher_output, teacher_output, teacher_output), dim=1)
            teacher_relight_loss = F.l1_loss(R_low * I_teacher_3, batch_input_high)

            # 反向传播和优化
            teacher_optim.zero_grad()
            teacher_relight_loss.backward()
            teacher_optim.step()

            total_loss += teacher_relight_loss.item()
            avg_loss = total_loss / (batch_id + 1)

            if (batch_id + 1) % 10 == 0:
                logger.info(f"Teacher Epoch: [{epoch_idx + 1}/{args.teacher_epochs}] "
                            f"[{batch_id + 1}/{numBatch}] loss: {avg_loss:.6f}")

        # 评估教师模型
        if (epoch_idx + 1) % max(1, args.teacher_epochs // 5) == 0:
            logger.info(f"评估教师模型 (Epoch {epoch_idx + 1})...")
            # 使用相同的评估数据集
            model.evaluate(epoch_idx + 1, eval_multi_names, eval_high_names,
                           vis_dir=args.vis_dir,
                           train_phase="Teacher",
                           num_exposures=args.num_exposures)

    # 保存教师模型
    teacher_ckpt_dir = os.path.join(args.ckpt_dir, 'teacher')
    os.makedirs(teacher_ckpt_dir, exist_ok=True)
    teacher_save_path = os.path.join(teacher_ckpt_dir, 'teacher_model.pth')
    torch.save(model.TeacherRelightNet.state_dict(), teacher_save_path)
    logger.info(f"教师模型已保存至 {teacher_save_path}")

    # 冻结教师模型参数
    for param in model.TeacherRelightNet.parameters():
        param.requires_grad = False

    return model


def main():
    args = parse_args()
    setup_seed(args.seed)

    # 创建必要的目录
    args.vis_dir = os.path.join(args.ckpt_dir, 'visuals')
    os.makedirs(args.ckpt_dir, exist_ok=True)
    os.makedirs(args.vis_dir, exist_ok=True)

    # 如果启用全新训练，删除旧检查点
    if args.fresh_start:
        logger.info("删除旧检查点，开始全新训练...")
        shutil.rmtree(args.ckpt_dir, ignore_errors=True)
        os.makedirs(args.ckpt_dir, exist_ok=True)
        os.makedirs(args.vis_dir, exist_ok=True)

    # 设备设置
    use_gpu = args.gpu_id != "-1" and torch.cuda.is_available()
    if use_gpu:
        device = torch.device(f"cuda:{args.gpu_id}")
        logger.info(f"使用 GPU: {args.gpu_id}")
    else:
        device = torch.device("cpu")
        logger.warning("使用CPU训练")

    # 初始化模型
    model = RetinexNet(num_exposures=args.num_exposures, use_distill=args.use_distill)
    model = model.to(device)

    # 学习率调度
    lr = args.lr * np.ones([args.epochs])
    lr[int(args.epochs * 0.6):] = args.lr / 10.0

    # 获取训练数据
    logger.info("准备训练数据...")
    try:
        # 修改：获取四组数据
        train_multi_names, train_high_names, eval_multi_names, eval_high_names = get_scene_pairs(
            args.gamma_data_dir,
            args.data_dir,
            args.num_exposures,
            args.eval_with_train_data
        )
    except Exception as e:
        logger.error(f"准备训练数据时出错: {str(e)}")
        return

    if not train_multi_names or not train_high_names:
        logger.error("错误：没有可用的训练数据！")
        logger.error(f"找到的多曝光场景数: {len(train_multi_names)}")
        logger.error(f"找到的正常光图像数: {len(train_high_names)}")
        return

    logger.info(f'训练场景数量: {len(train_multi_names)}')
    logger.info(f'评估场景数量: {len(eval_multi_names)}')

    # 分解阶段训练
    logger.info("开始分解阶段训练...")
    model.train_phase = 'Decom'
    model.train(
        train_multi_names,
        train_high_names,
        eval_multi_names,
        eval_high_names,
        batch_size=args.batch_size,
        patch_size=args.patch_size,
        epoch=args.epochs,
        lr=lr,
        vis_dir=args.vis_dir,
        ckpt_dir=args.ckpt_dir,
        eval_every_epoch=5,
        train_phase="Decom"
    )

    # 如果启用知识蒸馏，训练教师模型
    if args.use_distill:
        logger.info("启用知识蒸馏训练")
        model = train_teacher_model(model, train_multi_names, train_high_names,
                                    eval_multi_names, eval_high_names, args)

    # 重光照阶段训练
    logger.info("开始重光照阶段训练...")
    model.train_phase = 'Relight'

    # 确保分解网络冻结
    for param in model.DecomNet.parameters():
        param.requires_grad = False
    model.DecomNet.eval()

    model.train(
        train_multi_names,
        train_high_names,
        eval_multi_names,
        eval_high_names,
        batch_size=args.batch_size,
        patch_size=args.patch_size,
        epoch=args.epochs,
        lr=lr,
        vis_dir=args.vis_dir,
        ckpt_dir=args.ckpt_dir,
        eval_every_epoch=5,
        train_phase="Relight"
    )

    logger.info("训练完成！")
    logger.info(f"模型和日志保存在: {args.ckpt_dir}")


if __name__ == '__main__':
    main()