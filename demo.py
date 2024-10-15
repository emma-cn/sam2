import torch
from PIL import Image  # 确保导入 PIL
import numpy as np
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import warnings

# 忽略不必要的警告
warnings.filterwarnings("ignore", category=UserWarning)

# 加载模型的 checkpoint 和 config 文件路径
checkpoint = "./checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

# 初始化预测器
print("加载模型...")
predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))
print("模型加载完成")

# 替换为您想要处理的图片路径
image_path = "./test.jpg"
image = Image.open(image_path).convert("RGB")

# 使用推理模式，并将张量转换为混合精度 (bfloat16)
with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    predictor.set_image(image)
    masks, _, _ = predictor.predict(None)

# 打印掩码数组的结构以确保格式正确
print(f"Masks shape: {np.array(masks).shape}")

# 找到面积最大的掩码（假设它是人像掩码）
max_area = 0
best_mask = None

for mask in masks:
    # 计算掩码的非零像素数量（作为面积）
    area = np.sum(mask)

    if area > max_area:
        max_area = area
        best_mask = mask

# 如果找到了最佳掩码，则使用它抠出人物区域
if best_mask is not None:
    # 将掩码转换为 uint8 格式，并确保数组的类型符合 PIL 的要求
    mask_array = (best_mask * 255).astype(np.uint8)
    mask_image = Image.fromarray(mask_array)

    # 将掩码应用于原始图片，并创建一个带有 Alpha 通道的图像
    person_array = np.array(image).astype(np.uint8)

    # 创建一个空白的 Alpha 通道，初始值为 0（完全透明）
    alpha_channel = (best_mask * 255).astype(np.uint8)

    # 将原始图片和 Alpha 通道合并，生成 RGBA 图片
    rgba_image = Image.fromarray(np.dstack([person_array, alpha_channel]), mode="RGBA")

    # 保存抠出的人物图片（带透明背景）
    rgba_image.save("./person.png")
    print("已保存抠出的人物图片为带透明背景的 person.png。")

    # 打开图片并检查透明度
    result_image = Image.open("./person.png")
    if result_image.mode in ("RGBA", "LA"):
        print("图片具有透明背景")
    else:
        print("图片没有透明背景")
else:
    print("未找到合适的掩码。")
    