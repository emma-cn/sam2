import torch
from PIL import Image  # 确保导入 PIL
import numpy as np
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# 加载模型的 checkpoint 和 config 文件路径
checkpoint = "./checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

# 初始化预测器
predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))

# 替换为您想要处理的图片路径
image_path = "./test.jpg"
image = Image.open(image_path).convert("RGB")

# 使用推理模式，并将张量转换为混合精度 (bfloat16)
with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    predictor.set_image(image)
    masks, _, _ = predictor.predict(None)

# 打印 `masks` 的结构，确保我们理解其形状
print(f"Masks shape: {np.array(masks).shape}")

# 遍历生成的掩码，并将每个掩码保存为 PNG 文件
for i, mask in enumerate(masks):
    # `mask` 本身是一个 NumPy 数组，直接进行操作
    mask_array = (mask * 255).astype(np.uint8)

    # 将掩码数组转换为 PIL Image
    mask_image = Image.fromarray(mask_array)

    # 保存掩码为 PNG 文件，文件名格式为 mask_0.png, mask_1.png 等
    mask_image.save(f"./mask_{i}.png")

print("掩码图像已成功保存。")