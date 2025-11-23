import torch
import numpy as np
import torch.nn as nn
from math import exp
import torch.nn.functional as F
from torchvision.models import vgg16, vgg19
from torch.autograd import Variable

import utils.constant as constant

grid_w = constant.GRID_W
grid_h = constant.GRID_H
gpu_device = constant.GPU_DEVICE

min_w = (512 / grid_w) / 8
min_h = (384 / grid_h) / 8


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


class TV_Loss(torch.nn.Module):

    def __init__(self):
        super(TV_Loss, self).__init__()

    def forward(self, IA, IF):
        r = IA - IF
        h = r.shape[2]
        w = r.shape[3]
        tv1 = torch.pow((r[:, :, 1:, :] - r[:, :, :h - 1, :]), 2).mean()
        tv2 = torch.pow((r[:, :, :, 1:] - r[:, :, :, :w - 1]), 2).mean()

        return tv1 + tv2


class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()
        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)
            window = window.to(img1.get_device())
            window = window.type_as(img1)
            self.window = window
            self.channel = channel
        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)


class GradLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, img1, img2):
        device = img1.device
        b, c, h, w = img1.shape
        kernel = torch.FloatTensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]]) \
            .to(device).reshape((1, 1, 3, 3)).repeat(c, 1, 1, 1)
        f1_grad = F.conv2d(img1, kernel, padding=1, groups=c)
        f2_grad = F.conv2d(img2, kernel, padding=1, groups=c)
        totalGradLoss = intensity_loss(gen_frames=f1_grad, gt_frames=f2_grad, l_num=2)
        return totalGradLoss


class VGG(nn.Module):
    def __init__(self, layer_indexs):
        super(VGG, self).__init__()
        layers = []
        in_dim = 3
        out_dim = 64
        self.layer_indexs = layer_indexs
        for i in range(16):
            layers += [nn.Conv2d(in_dim, out_dim, 3, 1, 1), nn.ReLU(inplace=True)]
            in_dim = out_dim
            if i == 1 or i == 3 or i == 7 or i == 11 or i == 15:
                layers += [nn.MaxPool2d(2, 2)]
                if i != 11:
                    out_dim *= 2
        self.features = nn.Sequential(*layers)

    def forward(self, x):
        out = []
        for i in range(len(self.layer_indexs)):
            if i == 0:
                x = self.features[:self.layer_indexs[0] + 1](x)
            else:
                x = self.features[self.layer_indexs[i - 1] + 1:self.layer_indexs[i] + 1](x)
            out.append(x)
        # print("out:",len(out))
        return out


class PerceptualLoss(nn.Module):
    def __init__(self, weights=[1.0 / 2, 1.0], layer_indexs=[5, 22]):
        super(PerceptualLoss, self).__init__()
        self.criterion = nn.L1Loss().to(gpu_device)
        self.weights = weights
        self.layer_indexs = layer_indexs
        self.vgg = VGG(self.layer_indexs)
        self.vgg.features.load_state_dict(vgg19(pretrained=True).features.state_dict())
        self.vgg.to(gpu_device)
        self.vgg.eval()

        for parm in self.vgg.parameters():
            parm.requires_grad = False

    def forward(self, yPred, yGT):
        yPred = yPred.to(gpu_device)
        yGT = yGT.to(gpu_device)
        yPred_vgg, yGT_vgg = self.vgg(yPred), self.vgg(yGT)
        loss = 0
        for i in range(len(yPred_vgg)):
            loss += self.weights[i] * intensity_loss(yPred_vgg[i], yGT_vgg[i], l_num=2)
        return loss


def intensity_loss(gen_frames, gt_frames, l_num):
    return torch.mean(torch.abs((gen_frames - gt_frames) ** l_num))


# intra-grid constraint
def intra_grid_loss(pts):
    batch_size = pts.shape[0]

    delta_x = pts[:, :, 0:grid_w, 0] - pts[:, :, 1:grid_w + 1, 0]
    delta_y = pts[:, 0:grid_h, :, 1] - pts[:, 1:grid_h + 1, :, 1]

    loss_x = F.relu(delta_x + min_w)
    loss_y = F.relu(delta_y + min_h)

    loss = torch.mean(loss_x) + torch.mean(loss_y)
    return loss


# inter-grid constraint
def inter_grid_loss(train_mesh):
    w_edges = train_mesh[:, :, 0:grid_w, :] - train_mesh[:, :, 1:grid_w + 1, :]
    cos_w = torch.sum(w_edges[:, :, 0:grid_w - 1, :] * w_edges[:, :, 1:grid_w, :], 3) / \
            (torch.sqrt(torch.sum(w_edges[:, :, 0:grid_w - 1, :] * w_edges[:, :, 0:grid_w - 1, :], 3))
             * torch.sqrt(torch.sum(w_edges[:, :, 1:grid_w, :] * w_edges[:, :, 1:grid_w, :], 3)))
    # print("cos_w.shape")
    # print(cos_w.shape)
    delta_w_angle = 1 - cos_w

    h_edges = train_mesh[:, 0:grid_h, :, :] - train_mesh[:, 1:grid_h + 1, :, :]
    cos_h = torch.sum(h_edges[:, 0:grid_h - 1, :, :] * h_edges[:, 1:grid_h, :, :], 3) / \
            (torch.sqrt(torch.sum(h_edges[:, 0:grid_h - 1, :, :] * h_edges[:, 0:grid_h - 1, :, :], 3))
             * torch.sqrt(torch.sum(h_edges[:, 1:grid_h, :, :] * h_edges[:, 1:grid_h, :, :], 3)))
    delta_h_angle = 1 - cos_h

    loss = torch.mean(delta_w_angle) + torch.mean(delta_h_angle)
    return loss


def intensity_weight_loss(gen_frames, gt_frames, weight, l_num=1):
    # print(gen_frames.shape, gt_frames.shape)
    return torch.mean((torch.abs((gen_frames - gt_frames) * weight) ** l_num))


class Distill_loss_2Teacher_Weight(nn.Module):
    def __init__(self, lam_appearance, lam_perception, lam_mask, lam_mesh):
        super().__init__()
        self.lam_appearance = lam_appearance
        self.lam_ssim = 5e-6
        self.lam_mask = lam_mask
        self.lam_mesh = lam_mesh
        self.lam_primary_weight = 2
        self.lam_distill_weight = 0.01
        # weights = [0.1, 0.1, 1.0, 1.0, 1.0]
        # layer_indexs = [2, 7, 16, 25, 34]
        weights = [1.0]
        layer_indexs = [22]
        self.perceptual_loss = PerceptualLoss(weights=weights, layer_indexs=layer_indexs).to(gpu_device)
        self.grad_loss = GradLoss().to(gpu_device)
        self.l1_loss = nn.L1Loss().to(gpu_device)

    def forward(self, mesh_final, warp_image_final, warp_mask_final, ds_mesh1, ds_mesh2, weight1, weight2, img_gt):
        # mesh loss
        mesh_loss = intra_grid_loss(mesh_final) + inter_grid_loss(mesh_final)
        # appearance loss
        appearance_loss = self.l1_loss(warp_image_final, img_gt)
        # perception loss
        perception_loss = self.perceptual_loss(warp_image_final * 255., img_gt * 255.)
        # mask loss
        # mask_loss = self.l1_loss(warp_mask_final,torch.ones_like(warp_mask_final))

        # distill loss
        ds_w1 = weight1 * 1.0 / (weight1 + weight2)
        ds_w2 = weight2 * 1.0 / (weight1 + weight2)
        ds_w1 = ds_w1.unsqueeze(2).unsqueeze(3)
        ds_w2 = ds_w2.unsqueeze(2).unsqueeze(3)

        ds_loss1 = intensity_weight_loss(mesh_final, ds_mesh1, ds_w1, l_num=2)
        ds_loss2 = intensity_weight_loss(mesh_final, ds_mesh2, ds_w2, l_num=2)

        ds_loss = ds_loss1 + ds_loss2

        '''
        + ds_loss * self.lam_distill_weight
        ,ds_loss * self.lam_distill_weight *10
        '''

        # total loss
        primary_img_loss = appearance_loss * self.lam_appearance + perception_loss * self.lam_ssim + mesh_loss * self.lam_mesh  # + mask_loss * self.lam_appearance
        total_loss = primary_img_loss * self.lam_primary_weight
        return total_loss * 10, primary_img_loss * self.lam_primary_weight * 10


# L2损失
def L2_loss(gen_frames, gt_frames, l_num=2):
    return torch.mean((torch.abs(gen_frames - gt_frames) ** l_num))


# 均方误差（MSE）损失
def mse_loss(flow, final_flow):
    return torch.mean((flow - final_flow) ** 2)


# 端点误差（End - Point Error，EPE）损失
def epe_loss(flow, final_flow):
    return torch.mean(torch.sqrt((flow[..., 0] - final_flow[..., 0]) ** 2 + (flow[..., 1] - final_flow[..., 1]) ** 2))


# 设置训练的损失函数权重调整策略
# def get_weights(epoch):
#     # 阶段划分：阶段一：0-66轮，阶段二：67-133轮，阶段三：134-200轮
#     if epoch < 67:
#         # 第一阶段：主要计算 flow1 的损失
#         w1 = 1 - epoch / 200  # 从 1 减小到接近 0.7
#         w2 = epoch / 66 * 0.3  # 从 0 增加到接近 0.3
#         w3 = 0
#     elif epoch < 134:
#         # 第二阶段：主要计算 flow2 的损失
#         w1 = (134 - epoch) / 66 * 0.3  # 从 0.3 减小到接近 0
#         w2 = 1 - (epoch - 67) / 200  # 从 1 减小到接近 0.7
#         w3 = (epoch - 67) / 66 * 0.3  # 从 0 增加到接近 0.3
#     else:
#         # 第三阶段：主要计算 flow3 的损失
#         w1 = 0
#         w2 = (200 - epoch) / 66 * 0.2  # 从 0.2 减小到接近 0
#         w3 = 0.8 + (epoch - 134) / 200  # 从 0.8 增加到接近 1
#
#     return w1, w2, w3


def get_weight(n, start_weight=0.1, end_weight=0.01, total_rounds=50):
    """
    根据当前训练轮数n计算权重，权重从start_weight递减到end_weight。
    返回：当前训练轮数n的权重（保留两位小数）
    """
    # 计算每轮的递减量
    decrement_per_round = (start_weight - end_weight) / total_rounds

    # 计算当前轮数n对应的权重
    current_weight = start_weight - decrement_per_round * n

    # 保留两位小数
    weight = round(current_weight, 3)

    return weight


class Distill_loss_Flow(nn.Module):
    def __init__(self, lam_appearance, lam_perception, lam_mask, lam_mesh):
        super().__init__()
        # 初始化损失权重
        self.lam_appearance = lam_appearance
        self.lam_ssim = 5e-6
        self.lam_mask = lam_mask
        self.lam_mesh = lam_mesh
        self.lam_primary_weight = 2
        self.weak_distill_weight = 0.01
        self.strong_distill_weight = 0.01

        # 感知损失配置
        weights = [1.0]
        layer_indexs = [22]
        self.perceptual_loss = PerceptualLoss(weights=weights, layer_indexs=layer_indexs).to(gpu_device)

        # 其他损失函数
        self.grad_loss = GradLoss().to(gpu_device)
        self.l1_loss = nn.L1Loss().to(gpu_device)

        # 目标性能指标
        self.target_perf_ssim = 0.7952
        self.target_perf_psnr = 22.5402

        # 自适应蒸馏权重（初始值为1.0）
        self.ATdistill_weight = 1.0

    def update_distillation_strength(self, actual_time_perf):
        """
        根据实际性能动态调整蒸馏强度权重

        参数:
            actual_time_perf: 列表，包含最近的测试性能数据，每个元素是(psnr, ssim)元组
        """
        if not actual_time_perf:
            return

        # 计算最近的平均性能
        avg_psnr = sum([x[0] for x in actual_time_perf]) / len(actual_time_perf)
        avg_ssim = sum([x[1] for x in actual_time_perf]) / len(actual_time_perf)

        # 计算与目标性能的相对差距
        psnr_gap = (avg_psnr - self.target_perf_psnr) / self.target_perf_psnr
        ssim_gap = (avg_ssim - self.target_perf_ssim) / self.target_perf_ssim

        # 计算综合性能差距（加权平均，可根据需要调整权重）
        combined_gap = 0.7 * psnr_gap + 0.3 * ssim_gap  # 更重视PSNR

        # 动态调整蒸馏权重（使用平滑调整因子0.1）
        new_weight = self.ATdistill_weight * (1 + 0.1 * combined_gap)

        # 限制权重范围在[0.5, 1.5]
        self.ATdistill_weight = max(0.5, min(1.5, new_weight))

        # 打印调试信息
        print(f"Updated distill weight: {self.ATdistill_weight:.4f} "
              f"(PSNR: {avg_psnr:.2f}/{self.target_perf_psnr:.2f}, "
              f"SSIM: {avg_ssim:.4f}/{self.target_perf_ssim:.4f})")

    def forward(self, flow, warp_mask_final, warp_image_final, ds_flow3, epoch, img_gt):

        # 外观损失（L1）
        appearance_loss = self.l1_loss(warp_image_final, img_gt)

        # 感知损失（使用预定义的感知损失）
        perception_loss = self.perceptual_loss(warp_image_final * 255., img_gt * 255.)

        # 蒸馏损失（L2 + 动态权重）
        ds_loss = L2_loss(flow, ds_flow3) * self.ATdistill_weight

        # 主要图像损失组合
        primary_img_loss = appearance_loss * self.lam_appearance + perception_loss * self.lam_ssim

        if epoch > 199:
            # 总损失
            total_loss = primary_img_loss * self.lam_primary_weight + ds_loss * self.strong_distill_weight
        else:
            total_loss = primary_img_loss * self.lam_primary_weight + ds_loss * self.weak_distill_weight

        # 返回各项损失（乘以10作为最终缩放）
        return total_loss * 10, primary_img_loss * self.lam_primary_weight * 10, ds_loss * 10


class Distill_loss_residue_Weight(nn.Module):
    def __init__(self, lam_appearance, lam_perception, lam_mask, lam_mesh):
        super().__init__()
        self.lam_appearance = lam_appearance
        self.lam_ssim = 5e-6
        self.lam_mask = lam_mask
        self.lam_mesh = lam_mesh
        self.lam_primary_weight = 2
        self.lam_distill_weight = 0.01
        # weights = [0.1, 0.1, 1.0, 1.0, 1.0]
        # layer_indexs = [2, 7, 16, 25, 34]
        weights = [1.0]
        layer_indexs = [22]
        self.perceptual_loss = PerceptualLoss(weights=weights, layer_indexs=layer_indexs).to(gpu_device)
        self.grad_loss = GradLoss().to(gpu_device)
        self.l1_loss = nn.L1Loss().to(gpu_device)

    def forward(self, flow, warp_image_final, ds_flow3, epoch, img_gt):
        # mesh loss
        # appearance loss
        appearance_loss = self.l1_loss(warp_image_final, img_gt)
        # perception loss
        perception_loss = self.perceptual_loss(warp_image_final * 255., img_gt * 255.)
        # mask loss
        # mask_loss = self.l1_loss(warp_mask_final,torch.ones_like(warp_mask_final))

        # distill loss

        ds_loss = L2_loss(flow, ds_flow3) * self.lam_distill_weight

        '''
        loss_epe = epe_loss(flow, ds_flow3)
            ds_loss = loss_l2 + loss_epe
        '''

        # total loss
        primary_img_loss = appearance_loss * self.lam_appearance + perception_loss * self.lam_ssim  # + mask_loss * self.lam_appearance
        total_loss = primary_img_loss * self.lam_primary_weight + ds_loss
        return total_loss * 10, primary_img_loss * self.lam_primary_weight * 10, ds_loss * 10

