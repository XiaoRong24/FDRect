import os
import time
import argparse
import torch
import numpy as np
from net.Flow_DistillModel_Lite import FDRect   # 你的模型定义

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def measure_cpu_latency_fps(model, input_tensor, mask_tensor, warmup=10, repeats=50):
    """
    返回 CPU 平均推理延迟 (ms) 和 FPS
    """
    model.eval()
    # 预热
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(input_tensor, mask_tensor)

    # 计时
    start = time.time()
    with torch.no_grad():
        for _ in range(repeats):
            _ = model(input_tensor, mask_tensor)
    end = time.time()

    avg_time_s = (end - start) / repeats
    latency_ms = avg_time_s * 1000.0
    fps = 1.0 / avg_time_s
    return latency_ms, fps

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_h', type=int, default=384)
    parser.add_argument('--img_w', type=int, default=512)
    parser.add_argument('--save_model_name', type=str, default='../model/distill_model_epoch200.pkl')
    parser.add_argument('--device', type=str, default='cpu', help='Force CPU (default: cpu)')
    args = parser.parse_args()

    resize_h, resize_w = args.img_h, args.img_w

    # 加载模型到 CPU
    model = FDRect()
    pretrain_model = torch.load(args.save_model_name, map_location='cpu')
    model_dict = model.state_dict()
    state_dict = {k: v for k, v in pretrain_model.items() if k in model_dict.keys()}
    model_dict.update(state_dict)
    model.load_state_dict(model_dict)
    model = model.cpu()
    model.eval()

    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model parameters: {total_params:.2f} M")

    # 构造 dummy 输入
    dummy_input = torch.randn(1, 3, resize_h, resize_w)
    dummy_mask  = torch.randn(1, 3, resize_h, resize_w)

    # 测量 CPU 性能
    latency_ms, fps = measure_cpu_latency_fps(model, dummy_input, dummy_mask,
                                              warmup=10, repeats=50)
    print(f"CPU Latency: {latency_ms:.2f} ms")
    print(f"CPU FPS:     {fps:.2f}")

if __name__ == '__main__':
    main()
