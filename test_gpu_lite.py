import os
import time
import numpy as np
import cv2
import argparse
import torch
from PIL import Image
from torchvision import transforms
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from thop import profile, clever_format
from net.Flow_DistillModel_Lite import FDRect  # 你的模型定义

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch.nn.functional as F
import utils.torch_tps2flow as torch_tps2flow
import utils.constant as constant
grid_w = constant.GRID_W
grid_h = constant.GRID_H
gpu_device = constant.GPU_DEVICE

def shift2mesh0(mesh_shift, height,width):
    device = mesh_shift.device
    batch_size = mesh_shift.shape[0]
    h = height / grid_h
    w = width / grid_w
    ori_pt = []
    for i in range(grid_h + 1):
        for j in range(grid_w + 1):
            ww = j * w
            hh = i * h
            p = torch.FloatTensor([ww, hh])
            ori_pt.append(p.unsqueeze(0))
    ori_pt = torch.cat(ori_pt,dim=0)
    # print(ori_pt.shape)
    ori_pt = ori_pt.view(grid_h + 1, grid_w + 1, 2)
    # print(ori_pt)
    ori_pt = torch.tile(ori_pt.unsqueeze(0), [batch_size, 1, 1, 1])
    ori_pt = ori_pt.to(gpu_device)
    # print("ori_pt:",ori_pt.shape)
    # print("mesh_shift:", mesh_shift.shape)
    tar_pt = ori_pt + mesh_shift
    return tar_pt

def get_rigid_mesh(batch_size, height, width):

    ww = torch.matmul(torch.ones([grid_h+1, 1]), torch.unsqueeze(torch.linspace(0., float(width), grid_w+1), 0))
    hh = torch.matmul(torch.unsqueeze(torch.linspace(0.0, float(height), grid_h+1), 1), torch.ones([1, grid_w+1]))
    ww = ww.to(gpu_device)
    hh = hh.to(gpu_device)

    ori_pt = torch.cat((ww.unsqueeze(2), hh.unsqueeze(2)),2) # (grid_h+1)*(grid_w+1)*2
    ori_pt = ori_pt.unsqueeze(0).expand(batch_size, -1, -1, -1)

    return ori_pt

def get_norm_mesh(mesh, height, width):
    batch_size = mesh.size()[0]
    mesh_w = mesh[...,0]*2./float(width) - 1.
    mesh_h = mesh[...,1]*2./float(height) - 1.
    norm_mesh = torch.stack([mesh_w, mesh_h], 3) # bs*(grid_h+1)*(grid_w+1)*2
    # norm_mesh = torch.stack([mesh_h, mesh_w], 3)  # bs*(grid_h+1)*(grid_w+1)*2
    # print("norm_mesh:",norm_mesh.shape)
    return norm_mesh.reshape([batch_size, -1, 2]) # bs*-1*2

def warp_with_flow(img, flow):
    #initilize grid_coord
    batch, C, H, W = img.shape
    coords0 = torch.meshgrid(torch.arange(H).cuda(), torch.arange(W).cuda())
    coords0 = torch.stack(coords0[::-1], dim=0).float()
    coords0 = coords0[None].repeat(batch, 1, 1, 1)  # bs, 2, h, w

    # target coordinates
    target_coord = coords0 + flow

    # normalization
    target_coord_w = target_coord[:,0,:,:]*2./float(W) - 1.
    target_coord_h = target_coord[:,1,:,:]*2./float(H) - 1.
    target_coord_wh = torch.stack([target_coord_w, target_coord_h], 1)

    # warp
    warped_img = F.grid_sample(img, target_coord_wh.permute(0,2,3,1), align_corners=True)

    return warped_img


# ---------- 辅助函数 ----------
def get_model_params(model):
    """参数量 (M)"""
    return sum(p.numel() for p in model.parameters()) / 1e6


def get_flops(model, input_tensor, mask_tensor):
    """FLOPs (G)"""
    flops, _ = profile(model, inputs=(input_tensor, mask_tensor), verbose=False)
    return flops / 1e9


def measure_gpu_memory_and_latency(model, input_tensor, mask_tensor, warmup=10, repeats=100):
    """
    返回 GPU 峰值显存 (MB)、平均推理延迟 (ms) 和 FPS
    """
    model.eval()
    # 预热
    for _ in range(warmup):
        _ = model(input_tensor, mask_tensor)

    # 重置峰值显存统计
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    # 计时事件
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    torch.cuda.synchronize()
    start_event.record()
    for _ in range(repeats):
        _ = model(input_tensor, mask_tensor)
    end_event.record()
    torch.cuda.synchronize()

    avg_latency_ms = start_event.elapsed_time(end_event) / repeats
    peak_mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

    # 计算 FPS (每秒处理帧数)
    fps = 1000.0 / avg_latency_ms  # 将 ms 转换为秒的倒数

    return peak_mem_mb, avg_latency_ms, fps


def inference_func(pathInput, pathMask, pathGT, model_path, args):
    resize_h, resize_w = args.img_h, args.img_w
    transform_op = transforms.Compose([
        transforms.Resize([resize_h, resize_w]),
        transforms.ToTensor(),
    ])

    index_all = sorted([x.split('.')[0] for x in os.listdir(pathInput)])

    # ---------- 加载模型 ----------
    model = FDRect()
    pretrain_model = torch.load(model_path, map_location='cpu')
    model_dict = model.state_dict()
    state_dict = {k: v for k, v in pretrain_model.items() if k in model_dict.keys()}
    model_dict.update(state_dict)
    model.load_state_dict(model_dict)
    device = torch.device(f'cuda:{args.device_ids[0]}')
    model = model.to(device)
    model.eval()

    # ---------- 模型复杂度与性能指标 ----------
    # 构造 dummy 输入（与实际尺寸一致）
    dummy_input = torch.randn(1, 3, resize_h, resize_w).to(device)
    dummy_mask = torch.randn(1, 3, resize_h, resize_w).to(device)

    params_m = get_model_params(model)
    flops_g = get_flops(model, dummy_input, dummy_mask)
    gpu_mem_mb, gpu_latency_ms, fps = measure_gpu_memory_and_latency(
        model, dummy_input, dummy_mask, warmup=10, repeats=30
    )

    print(f"Params: {params_m:.2f} M")
    print(f"FLOPs: {flops_g:.3f} G")
    print(f"GPU Peak Memory: {gpu_mem_mb:.2f} MB")
    print(f"GPU Latency: {gpu_latency_ms:.2f} ms")
    print(f"FPS: {fps:.2f} frames/sec")

    # ---------- 图像质量评估 ----------
    psnr_list = []
    ssim_list = []
    length = len(index_all)  # 根据实际图片数量自动调整

    for i in range(length):
        idx = index_all[i]
        input_img = cv2.imread(os.path.join(pathInput, f'{idx}.jpg'))
        mask_img = cv2.imread(os.path.join(pathMask, f'{idx}.jpg'))
        gt_img = cv2.imread(os.path.join(pathGT, f'{idx}.jpg'))

        input_img = Image.fromarray(input_img)
        mask_img = Image.fromarray(mask_img)
        gt_img = Image.fromarray(gt_img)

        test_input = transform_op(input_img).unsqueeze(0).float().to(gpu_device)
        test_mask = transform_op(mask_img).unsqueeze(0).float().to(gpu_device)
        test_gt = transform_op(gt_img).unsqueeze(0).float().to(gpu_device)

        with torch.no_grad():
            mesh_motion = model(test_input, test_mask)

        # build model
        batch_size, _, height, width = test_input.shape
        '''convert TPS deformation to optical flows (image resolution: 384*512)'''
        rigid_mesh = get_rigid_mesh(batch_size, height, width)
        norm_rigid_mesh = get_norm_mesh(rigid_mesh, height, width)
        pre_mesh = rigid_mesh + mesh_motion
        norm_pre_mesh = get_norm_mesh(pre_mesh, height, width)
        delta_flow = torch_tps2flow.transformer(test_input, norm_rigid_mesh, norm_pre_mesh, (height, width))
        warp_flow1 = warp_with_flow(delta_flow, delta_flow)
        flow1 = delta_flow + warp_flow1
        final_image = warp_with_flow(test_input, flow1)
        final_mask = warp_with_flow(test_mask, flow1)

        warp_np = final_image.squeeze(0).permute(1, 2, 0).cpu().numpy()
        gt_np = test_gt.squeeze(0).permute(1, 2, 0).cpu().numpy()

        psnr = compare_psnr(warp_np, gt_np, data_range=1)
        ssim = compare_ssim(warp_np, gt_np, data_range=1, channel_axis=2)

        psnr_list.append(psnr)
        ssim_list.append(ssim)
        print(f'i = {i + 1}/{length}, PSNR = {psnr:.4f}, SSIM = {ssim:.4f}')

    # ---------- 汇总结果 ----------
    print("\n=================== Results ===================")
    print(f"Average PSNR:           {np.mean(psnr_list):.4f}")
    print(f"Average SSIM:           {np.mean(ssim_list):.4f}")
    print(f"Params (M):             {params_m:.2f}")
    print(f"FLOPs (G):              {flops_g:.3f}")
    print(f"GPU Mem (MB):           {gpu_mem_mb:.2f}")
    print(f"GPU Latency (ms):       {gpu_latency_ms:.2f}")
    print(f"FPS:                    {fps:.2f}")

    # 可选：保存结果到文件
    if hasattr(args, 'save_result') and args.save_result:
        with open('evaluation_results.txt', 'w') as f:
            f.write(f"Average PSNR: {np.mean(psnr_list):.4f}\n")
            f.write(f"Average SSIM: {np.mean(ssim_list):.4f}\n")
            f.write(f"Params (M): {params_m:.2f}\n")
            f.write(f"FLOPs (G): {flops_g:.3f}\n")
            f.write(f"GPU Mem (MB): {gpu_mem_mb:.2f}\n")
            f.write(f"GPU Latency (ms): {gpu_latency_ms:.2f}\n")
            f.write(f"FPS: {fps:.2f}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='../DIR-D/')
    parser.add_argument('--device_ids', type=list, default=[0])
    parser.add_argument('--img_h', type=int, default=384)
    parser.add_argument('--img_w', type=int, default=512)
    parser.add_argument('--save_model_name', type=str, default='../model/distill_model_epoch200.pkl')
    parser.add_argument('--save_result', action='store_true', help='Save results to file')
    args = parser.parse_args()

    pathGT = os.path.join(args.path, 'testing/gt')
    pathInput = os.path.join(args.path, 'testing/input')
    pathMask = os.path.join(args.path, 'testing/mask')

    inference_func(pathInput, pathMask, pathGT, args.save_model_name, args)