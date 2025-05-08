import os
from PIL import Image
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import numpy as np
import cv2
import argparse
import torch
from thop import profile
from net.Flow_DistillModel import FRKD
import torchvision.transforms as transforms
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
import torch.nn.functional as F
# import lpips
import time
from scipy.linalg import sqrtm
from torchvision.models import inception_v3
from torch.nn.functional import interpolate
from torch.nn.functional import adaptive_avg_pool2d
# os.environ['CUDA_DEVICES_ORDER'] = "PCI_BUS_ID"
# os.environ['CUDA_VISIBLE_DEVICES'] = "0"

def make_colorwheel():
    """
    Generates a color wheel for optical flow visualization as presented in:
        Baker et al. "A Database and Evaluation Methodology for Optical Flow" (ICCV, 2007)
        URL: http://vision.middlebury.edu/flow/flowEval-iccv07.pdf

    Code follows the original C++ source code of Daniel Scharstein.
    Code follows the the Matlab source code of Deqing Sun.

    Returns:
        np.ndarray: Color wheel
    """

    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros((ncols, 3))
    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.floor(255*np.arange(0,RY)/RY)
    col = col+RY
    # YG
    colorwheel[col:col+YG, 0] = 255 - np.floor(255*np.arange(0,YG)/YG)
    colorwheel[col:col+YG, 1] = 255
    col = col+YG
    # GC
    colorwheel[col:col+GC, 1] = 255
    colorwheel[col:col+GC, 2] = np.floor(255*np.arange(0,GC)/GC)
    col = col+GC
    # CB
    colorwheel[col:col+CB, 1] = 255 - np.floor(255*np.arange(CB)/CB)
    colorwheel[col:col+CB, 2] = 255
    col = col+CB
    # BM
    colorwheel[col:col+BM, 2] = 255
    colorwheel[col:col+BM, 0] = np.floor(255*np.arange(0,BM)/BM)
    col = col+BM
    # MR
    colorwheel[col:col+MR, 2] = 255 - np.floor(255*np.arange(MR)/MR)
    colorwheel[col:col+MR, 0] = 255
    return colorwheel


def flow_uv_to_colors(u, v, convert_to_bgr=False):
    """
    Applies the flow color wheel to (possibly clipped) flow components u and v.

    According to the C++ source code of Daniel Scharstein
    According to the Matlab source code of Deqing Sun

    Args:
        u (np.ndarray): Input horizontal flow of shape [H,W]
        v (np.ndarray): Input vertical flow of shape [H,W]
        convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.

    Returns:
        np.ndarray: Flow visualization image of shape [H,W,3]
    """
    flow_image = np.zeros((u.shape[0], u.shape[1], 3), np.uint8)
    colorwheel = make_colorwheel()  # shape [55x3]
    ncols = colorwheel.shape[0]
    rad = np.sqrt(np.square(u) + np.square(v))
    a = np.arctan2(-v, -u)/np.pi
    fk = (a+1) / 2*(ncols-1)
    k0 = np.floor(fk).astype(np.int32)
    k1 = k0 + 1
    k1[k1 == ncols] = 0
    f = fk - k0
    for i in range(colorwheel.shape[1]):
        tmp = colorwheel[:,i]
        col0 = tmp[k0] / 255.0
        col1 = tmp[k1] / 255.0
        col = (1-f)*col0 + f*col1
        idx = (rad <= 1)
        col[idx]  = 1 - rad[idx] * (1-col[idx])
        col[~idx] = col[~idx] * 0.75   # out of range
        # Note the 2-i => BGR instead of RGB
        ch_idx = 2-i if convert_to_bgr else i
        flow_image[:,:,ch_idx] = np.floor(255 * col)
    return flow_image


def flow_to_image(flow_uv, clip_flow=None, convert_to_bgr=False):
    """
    Expects a two dimensional flow image of shape.

    Args:
        flow_uv (np.ndarray): Flow UV image of shape [H,W,2]
        clip_flow (float, optional): Clip maximum of flow values. Defaults to None.
        convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.

    Returns:
        np.ndarray: Flow visualization image of shape [H,W,3]
    """
    assert flow_uv.ndim == 3, 'input flow must have three dimensions'
    assert flow_uv.shape[2] == 2, 'input flow must have shape [H,W,2]'
    if clip_flow is not None:
        flow_uv = np.clip(flow_uv, 0, clip_flow)
    u = flow_uv[:,:,0]
    v = flow_uv[:,:,1]
    rad = np.sqrt(np.square(u) + np.square(v))
    rad_max = np.max(rad)
    epsilon = 1e-5
    u = u / (rad_max + epsilon)
    v = v / (rad_max + epsilon)
    return flow_uv_to_colors(u, v, convert_to_bgr)



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

def calculate_lpips(real_images, generated_images, net_type='alex', device='cuda'):
    """
    计算两组图像之间的LPIPS（越低越好）
    :param real_images: 真实图像张量 [N,C,H,W]，范围[0,1]
    :param generated_images: 生成图像张量 [N,C,H,W]，范围[0,1]
    :param net_type: 'alex' | 'vgg' | 'squeeze'
    :return: LPIPS均值
    """
    loss_fn = lpips.LPIPS(net=net_type).to(device)
    with torch.no_grad():
        return loss_fn(real_images, generated_images).mean().item()


def calculate_params_flops(model, input_size, mask_size, device='cuda'):
    """
    计算模型的参数量（M）和计算量（GFLOPs）
    :param model: PyTorch模型
    :param input_size: 输入张量尺寸 (C,H,W)
    :return: (参数量(M), GFLOPs)
    """
    # 生成模拟输入张量（随机值）
    # dummy_input = torch.randn(*input_size).to(device)  # 例如形状 (1,3,512,384)
    # dummy_mask = torch.randn(*mask_size).to(device)  # 例如形状 (1,3,512,384)

    # 预热（避免首次推理的额外开销）
    with torch.no_grad():
        for _ in range(3):
            _ = model(input_size, mask_size)

    flops, params = profile(model, inputs=(input_size,mask_size), custom_ops={torch.nn.functional.interpolate: None}, verbose=False)
    if flops <= 0:
        raise ValueError(f"Invalid FLOPs value: {flops}. Check model or input dimensions.")

    # 计算实际推理FLOPs（考虑batch_size）
    batch_flops = float(flops) / 1e9  # 转换为 GFLOPs
    return params / 1e6, batch_flops  # 返回参数量(M)和单样本FLOPs(G)


def measure_inference_time(model, input_size, mask_size, warmup=10, repeats=10, device='cuda'):
    """
    测量模型平均推理时间（秒）
    :param model: PyTorch模型
    :param warmup: 预热次数（避免冷启动误差）
    :param repeats: 重复测量次数
    :return: 平均时间（秒）
    """
    model.eval()


    # 预热
    for _ in range(warmup):
        _ = model(input_size,mask_size)

    # 正式测量
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(repeats):
        _ = model(input_size,mask_size)
    torch.cuda.synchronize()
    return (time.time() - start) / repeats

def inference_func(pathInput2,pathMask2,pathGT2,model_path):
    resize_w, resize_h = args.img_w,args.img_h
    _origin_transform = transforms.Compose([
        transforms.Resize([resize_h, resize_w]),
        transforms.ToTensor(),
    ])

    index_all = list(sorted([x.split('.')[0] for x in os.listdir(pathInput2)]))
    # loading model
    model = FRKD()
    # model.load_state_dict(torch.load(model_path))

    pretrain_model = torch.load(model_path, map_location='cpu')
    # Extract K,V from the existing model
    model_dict = model.state_dict()
    # Create a new weight dictionary and update it
    state_dict = {k: v for k, v in pretrain_model.items() if k in model_dict.keys()}
    # Update the weight dictionary of the existing model
    model_dict.update(state_dict)
    # Loading the updated weight dictionary
    model.load_state_dict(model_dict)
    # loading model to device 0
    model = model.cuda(device=args.device_ids[0])
    # model.featureExtrator.fuse()
    # model.meshRegression.fuse()
    # print(model)
    model.eval()
    total = sum([param.nelement() for param in model.parameters()])
    print("Number of parameter: %.2fM" % (total / 1e6))

    psnr_list = []
    ssim_list = []
    lpips_list = []
    params_m_list = []
    gflops_list = []
    fid_list = []
    timess_list = []
    length = 519  # 665
    for i in range(0, length):
        idx = index_all[i]

        input_img = cv2.imread(os.path.join(pathInput2, str(idx) + '.jpg'))
        mask_img = cv2.imread(os.path.join(pathMask2, str(idx) + '.jpg'))
        gt_img = cv2.imread(os.path.join(pathGT2, str(idx) + '.jpg'))


        input_img = Image.fromarray(input_img)
        mask_img = Image.fromarray(mask_img)
        gt_img = Image.fromarray(gt_img)
        ###
        test_input = _origin_transform(input_img).unsqueeze(0).float().to(args.device_ids[0])
        test_mask = _origin_transform(mask_img).unsqueeze(0).float().to(args.device_ids[0])
        test_gt = _origin_transform(gt_img).unsqueeze(0).float().to(args.device_ids[0])


        flow, warp_mask_final, final_image = model.forward(test_input, test_mask)

        # mesh_final, total_flow,warp_image_final, warp_mask_final, final_image, mesh_v = model.forward(test_input, test_mask)

        # delta_flow = total_flow[0]
        # residual_flow = total_flow[1]
        # flow = total_flow[2]
        #
        # lpips = calculate_lpips(test_gt, final_image)
        # lpips_list.append(lpips)
        # params_m, gflops = calculate_params_flops(model,test_input,test_mask)
        # params_m_list.append(params_m)
        # gflops_list.append(gflops)
        # if i < 11:
        #     timess = measure_inference_time(model, test_input, test_mask)
        #     timess_list.append(timess)

        warp_image = final_image.squeeze(0).permute(1,2,0).cpu().detach().numpy()
        test_gt = test_gt.squeeze(0).permute(1,2,0).cpu().detach().numpy()

        I1 = warp_image
        I2 = test_gt
        psnr = compare_psnr(I1, I2 , data_range=1)
        ssim = compare_ssim(I1 , I2, data_range=1, channel_axis=2)

        print('i = {} / {}, psnr = {:.6f}, ssim = {:.6f}'.format(i + 1, length, psnr, ssim))
        psnr_list.append(psnr)
        ssim_list.append(ssim)




        # fusion = np.zeros_like(test_gt, dtype=np.float64)
        # fusion[..., 0] = warp_image[..., 0]
        # fusion[..., 1] = test_gt[..., 1] * 0.5 + warp_image[..., 1] * 0.5
        # fusion[..., 2] = test_gt[..., 2]
        # fusion = np.clip(fusion, 0, 1)
        # path1 = "result/final_fusion/" + str(i + 1).zfill(5) + ".jpg"
        # cv2.imwrite(path1, fusion * 255.)


        #
        path2 = "result/Final_Image/" + str(i + 1).zfill(5) + ".jpg"
        cv2.imwrite(path2, warp_image * 255.)

        ''' Image generation'''
        # final_mesh = mesh_final[0].cpu().detach().numpy()
        # path2 = "result/mesh/" + str(i + 1).zfill(5) + ".jpg"
        # cv2.imwrite(path2, mesh_v)
        #


        # f_image = warp_with_flow(test_input, flow)
        # f_image = f_image.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
        # path2 = "result/Final_Image/" + str(i + 1).zfill(5) + ".jpg"
        # cv2.imwrite(path2, warp_image * 255.)

        '''Tps_fusion image'''
        # warp_image_Tps = warp_image_Tps.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
        # fusion_Tps = np.zeros_like(test_gt, dtype=np.float64)
        # fusion_Tps[..., 0] = warp_image_Tps[..., 0]
        # fusion_Tps[..., 1] = test_gt[..., 1] * 0.5 + warp_image_Tps[..., 1] * 0.5
        # fusion_Tps[..., 2] = test_gt[..., 2]
        # fusion_Tps = np.clip(fusion_Tps, 0, 1)
        # path2 = "result/Tps_Image/" + str(i + 1).zfill(5) + ".jpg"
        # # warp_image = cv2.resize(warp_image, (512, 384))
        # cv2.imwrite(path2, fusion_Tps * 255.)

        # warp_mask_final = warp_mask_final.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
        # path2 = "result/Tps_mask/" + str(i + 1).zfill(5) + ".jpg"
        # # warp_image = cv2.resize(warp_image, (512, 384))
        # cv2.imwrite(path2, warp_mask_final * 255.)
        #
        # ''' Flow visualization '''
        # delta_flow = (delta_flow[0]).cpu().detach().numpy().transpose(1, 2, 0)
        # delta_flow = flow_to_image(delta_flow)
        # path3 = "result/delta_flow/" + str(i + 1).zfill(5) + ".jpg"
        # cv2.imwrite(path3, delta_flow)
        #
        # residual_flow = (residual_flow[0]).cpu().detach().numpy().transpose(1, 2, 0)
        # residual_flow = flow_to_image(residual_flow)
        # path3 = "result/residual_flow/" + str(i + 1).zfill(5) + ".jpg"
        # cv2.imwrite(path3, residual_flow)
        #
        # flow = (flow[0]).cpu().detach().numpy().transpose(1, 2, 0)
        # flow = flow_to_image(flow)
        # path3 = "result/Final_flow/" + str(i + 1).zfill(5) + ".jpg"
        # cv2.imwrite(path3, flow)


    print("===================Results Analysis==================")
    print('average psnr:', np.mean(psnr_list))
    print('average ssim:', np.mean(ssim_list))
    print('average lpips:', np.mean(lpips_list))
    print('average params_m:', np.mean(params_m_list))
    print('average gflops:', np.mean(gflops_list))
    # print('average time:', np.mean(timess_list))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='./DIR-D/')
    parser.add_argument('--device_ids', type=list, default=[0]) 
    parser.add_argument('--img_h', type=int, default=384)
    parser.add_argument('--img_w', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--max_epoch', type=int, default=30)
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--save_model_name', type=str, default='model/distill_model_epoch200.pkl')
    parser.add_argument('--lam_perception', type=float, default=5e-6)
    parser.add_argument('--lam_mask', type=float, default=1)
    parser.add_argument('--lam_mesh', type=float, default=1)
    parser.add_argument('--lam_appearance', type=float, default=1)
    print('<==================== Loading data ===================>\n')

    args = parser.parse_args()
    print(args)
    ##############
    pathGT2 = os.path.join(args.path, 'testing/gt')
    pathInput2 = os.path.join(args.path, 'testing/input')
    pathMask2 = os.path.join(args.path, 'testing/mask')
    model_path = args.save_model_name
    # test
    inference_func(pathInput2,pathMask2,pathGT2,model_path)

        
        
        


    






