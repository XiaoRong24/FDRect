

import torch
from thop import profile
from net.Flow_DistillModel import FDRect


# 检查是否有可用的CUDA设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 创建模型实例并移动到GPU
model = FDRect().to(device)

# 生成随机输入并移动到GPU（根据你的实际输入尺寸调整）
batch_size = 1
height, width = 384, 512  # 假设输入图像尺寸
input_img = torch.randn(batch_size, 3, height, width).to(device)
mask_img = torch.randn(batch_size, 3, height, width).to(device)  # 假设mask与输入图像尺寸相同

# 计算FLOPS和参数数量
flops, params = profile(model, inputs=(input_img, mask_img))

# 格式化输出结果
print(f"FLOPS: {flops / 1e9:.2f} G")  # 转换为千兆次浮点运算
print(f"Params: {params / 1e6:.2f} M")  # 转换为百万个参数