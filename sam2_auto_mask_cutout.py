import torch
from PIL import Image
import numpy as np
import argparse
import os
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor  # 使用此类来生成掩码
from pathlib import Path
from hydra import initialize, compose
from hydra.core.global_hydra import GlobalHydra


def load_image(image_path):
    """加载图片，并转换为 RGB 格式"""
    image = Image.open(image_path).convert("RGB")
    return image

def save_cutout(image, mask, output_path):
    """将掩码应用于原图，并保存为透明背景 PNG 图像"""
    image_np = np.array(image)  # 转换为 NumPy 数组
    alpha = (mask * 255).astype(np.uint8)  # 掩码作为 Alpha 通道

    # 创建 RGBA 图像，并替换 Alpha 通道
    result_np = np.dstack((image_np, alpha))
    result_image = Image.fromarray(result_np, mode="RGBA")

    # 保存结果图像
    result_image.save(output_path, format="PNG")
    print(f"抠图完成，已保存为：{output_path}")



def main(args):
    # 清理 Hydra 实例，确保不会重复初始化
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()

    # 获取配置文件目录
    config_path = Path(args.config).parent
    print(f"配置路径: {config_path}")

    # 切换到脚本所在目录，确保路径一致
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    print(f"切换到工作目录: {script_dir}")

    # 初始化 Hydra
    with initialize(config_path=str(config_path), job_name="sam2"):
        config_name = Path(args.config).stem  # 获取配置文件名（不带 .yaml 后缀）
        cfg = compose(config_name=config_name)

        # 加载模型和权重
        print("加载模型...")
        model = build_sam2(cfg, args.checkpoint)
        device = torch.device(args.device)
        model.to(device)

        # 初始化预测器并加载图片
        predictor = SAM2ImagePredictor(model)
        image = load_image(args.input)

        # 生成掩码
        print("生成掩码...")
        with torch.no_grad():
            predictor.set_image(image)
            masks, _, _ = predictor.predict(None)

        # 保存最大掩码的抠图结果
        if masks:
            largest_mask = masks[0]
            save_cutout(image, largest_mask, args.output)
        else:
            print("未检测到有效掩码。")

if __name__ == "__main__":
    import argparse

    # 定义命令行参数
    parser = argparse.ArgumentParser(description="SAM2 图像自动抠图工具")
    parser.add_argument("--input", type=str, required=True, help="输入图片路径")
    parser.add_argument("--output", type=str, required=True, help="输出图片路径")
    parser.add_argument("--checkpoint", type=str, required=True, help="模型权重文件路径")
    parser.add_argument("--config", type=str, required=True, help="模型配置文件路径")
    parser.add_argument("--device", type=str, default="cuda", help="运行设备：cuda 或 cpu")
    args = parser.parse_args()

    main(args)
