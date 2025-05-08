import torch
from net.DistillModel_Mamba import RectanglingNetwork

def compute_erf(model, input_size=32):
    # 生成输入（1张随机图像）
    input_img = torch.randn(1, 3, 384, 512).requires_grad_(True)
    mask_img = torch.randn(1, 3, 384, 512).requires_grad_(True)
    output = model.forward(input_img, mask_img)  # 前向传播

    # 选择中心像素的响应（假设输出与输入尺寸相同）
    center_y, center_x = input_size // 2, input_size // 2
    target = output[0, :, center_y, center_x].sum()  # 取中心点所有通道的和

    # 反向传播获取梯度
    target.backward()
    erf = input_img.grad.abs().sum(dim=1).squeeze()  # 梯度绝对值求和（近似 ERF）
    return erf.numpy()


# 示例：计算 ResNet-50 的 ERF
erf_resnet = compute_erf(RectanglingNetwork)