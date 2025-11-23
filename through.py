

import torch
from thop import profile
from net.Flow_DistillModel import FDRect
import numpy as np
import time




# --- 2. 吞吐量测量代码 ---
def measure_throughput():
    # 实例化模型并移至GPU
    if not torch.cuda.is_available():
        print("未找到可用的CUDA设备。请确保你的GPU和驱动已正确配置。")
        return

    model = FDRect().eval().cuda()

    # 创建模拟输入数据
    batch_size = 8
    height, width = 384, 512
    input_img = torch.randn(batch_size, 3, height, width).cuda()
    mask_img = torch.randn(batch_size, 3, height, width).cuda()

    # 预热GPU
    print("开始GPU预热...")
    for _ in range(10):
        _ = model(input_img, mask_img)
    print("GPU预热完成。")
    print("-" * 30)

    # 计时
    num_iterations = 100
    total_time = 0

    print("开始吞吐量测试...")
    with torch.no_grad():
        for _ in range(num_iterations):
            # 等待所有CUDA操作完成，以获取准确的计时
            torch.cuda.synchronize()
            start_time = time.time()

            _ = model(input_img, mask_img)

            torch.cuda.synchronize()
            end_time = time.time()
            total_time += (end_time - start_time)

    # 计算结果
    average_time_per_batch = total_time / num_iterations
    throughput = batch_size / average_time_per_batch

    print("吞吐量测试完成。")
    print("-" * 30)
    print(f"每批次 ({batch_size} 张图像) 的平均处理时间: {average_time_per_batch:.4f} 秒")
    print(f"模型的吞吐量: {throughput:.2f} 张/秒")


if __name__ == "__main__":
    measure_throughput()