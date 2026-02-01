import torch
import numpy as np
from PIL import Image
import os
import argparse
import time
import cv2
import matplotlib.pyplot as plt
from torchvision.transforms import ToTensor
from New_My_Network import LowLightEnhancer, enhance_image, quantize_model


def main():
    # 参数解析
    parser = argparse.ArgumentParser(description='低光照图像增强测试')
    parser.add_argument('--model_path', type=str, default='best_enhancer.pth',
                        help='训练好的模型路径')
    parser.add_argument('--input_dir', type=str, default='F:/研究生论文/10_image_enhanced/LOLdataset/test/low/',
                        help='测试图像目录')
    parser.add_argument('--output_dir', type=str, default='results/',
                        help='结果保存目录')
    parser.add_argument('--quantized', action='store_true',
                        help='是否使用量化模型')
    parser.add_argument('--hdr', action='store_true',
                        help='是否保存HDR版本')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='批处理大小')
    parser.add_argument('--visualize', action='store_true',
                        help='是否可视化结果')
    args = parser.parse_args()

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 加载模型
    model = LowLightEnhancer().to(device)

    try:
        if args.model_path:
            model.load_state_dict(torch.load(args.model_path, map_location=device))
            print(f"成功加载模型: {args.model_path}")
    except Exception as e:
        print(f"加载模型失败: {e}")
        print("请确保模型路径正确且与当前架构兼容")
        return

    # 量化模型（如果指定）
    if args.quantized:
        model = quantize_model(model)
        print("已使用量化模型进行推理")

    # 收集测试图像
    image_paths = []
    for f in os.listdir(args.input_dir):
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            image_paths.append(os.path.join(args.input_dir, f))

    if not image_paths:
        print(f"在 {args.input_dir} 中没有找到图像文件")
        return

    print(f"找到 {len(image_paths)} 张测试图像")

    # 准备输出路径
    output_paths = [os.path.join(args.output_dir, os.path.basename(p)) for p in image_paths]

    # 测试模型
    total_time = 0
    enhanced_images = []

    # 批处理推理
    for i in range(0, len(image_paths), args.batch_size):
        batch_paths = image_paths[i:i + args.batch_size]
        batch_outputs = output_paths[i:i + args.batch_size]

        start_time = time.time()
        enhanced = enhance_image(model, batch_paths, batch_outputs, hdr_output=args.hdr)
        batch_time = time.time() - start_time

        total_time += batch_time
        enhanced_images.extend(enhanced)

        print(
            f"已处理 {min(i + args.batch_size, len(image_paths))}/{len(image_paths)} 张图像, 批处理时间: {batch_time:.2f}秒")

    # 计算性能指标
    avg_time_per_image = total_time / len(image_paths)
    print(f"\n处理完成! 平均每张图像处理时间: {avg_time_per_image:.4f}秒")
    print(f"结果已保存至: {args.output_dir}")

    # 可视化结果（如果指定）
    if args.visualize:
        visualize_results(image_paths, enhanced_images, args.output_dir)


def visualize_results(input_paths, enhanced_images, output_dir):
    """可视化输入图像与增强结果"""
    print("\n可视化结果...")

    # 创建可视化目录
    vis_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)

    for i, (input_path, enhanced_img) in enumerate(zip(input_paths, enhanced_images)):
        # 加载原始图像
        orig_img = Image.open(input_path).convert('RGB')

        # 创建对比图
        plt.figure(figsize=(15, 7))

        # 原始图像
        plt.subplot(1, 2, 1)
        plt.imshow(orig_img)
        plt.title("原始低光照图像")
        plt.axis('off')

        # 增强结果
        plt.subplot(1, 2, 2)
        plt.imshow(enhanced_img)
        plt.title("增强结果")
        plt.axis('off')

        # 保存可视化结果
        vis_path = os.path.join(vis_dir, f"comparison_{i + 1}.png")
        plt.savefig(vis_path, bbox_inches='tight')
        plt.close()

        # 创建直方图对比
        plot_histogram_comparison(orig_img, enhanced_img, i + 1, vis_dir)


def plot_histogram_comparison(orig_img, enhanced_img, idx, output_dir):
    """绘制原始图像与增强结果的直方图对比"""
    # 转换为numpy数组
    orig_arr = np.array(orig_img).astype(np.float32) / 255.0
    enhanced_arr = np.array(enhanced_img).astype(np.float32) / 255.0

    # 创建直方图
    plt.figure(figsize=(10, 6))

    # 绘制RGB通道直方图
    colors = ['r', 'g', 'b']
    for j, color in enumerate(colors):
        # 原始图像直方图
        plt.hist(orig_arr[:, :, j].flatten(), bins=50, alpha=0.5,
                 color=color, label=f"原始 {color.upper()}")

        # 增强图像直方图
        plt.hist(enhanced_arr[:, :, j].flatten(), bins=50, alpha=0.3,
                 color=color, label=f"增强 {color.upper()}",
                 histtype='step', linewidth=2)

    plt.title("颜色直方图对比")
    plt.xlabel("像素值")
    plt.ylabel("频率")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 保存直方图
    hist_path = os.path.join(output_dir, f"histogram_{idx}.png")
    plt.savefig(hist_path, bbox_inches='tight')
    plt.close()


def analyze_model_performance(model, device):
    """分析模型性能"""
    print("\n模型性能分析:")

    # 计算模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")

    # 测量推理时间
    dummy_input = torch.randn(1, 3, 256, 256).to(device)

    # 预热
    for _ in range(5):
        with torch.no_grad():
            _ = model(dummy_input)

    # 正式测量
    timings = []
    for _ in range(20):
        start_time = time.perf_counter()
        with torch.no_grad():
            _ = model(dummy_input)
        timings.append(time.perf_counter() - start_time)

    avg_time = sum(timings) / len(timings) * 1000  # 转换为毫秒
    fps = 1000 / avg_time

    print(f"平均推理时间: {avg_time:.2f} ms")
    print(f"帧率(FPS): {fps:.2f}")

    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "avg_inference_time": avg_time,
        "fps": fps
    }


if __name__ == "__main__":
    main()