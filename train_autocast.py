import argparse
import torch
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import torch
import argparse
import time
from tqdm import tqdm
import numpy as np
import random

import torch.nn.functional as F
from torch.cuda.amp import autocast

from torch.utils.data import DataLoader
import torchvision.models as models

from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr

from net.DistillModel import StudentNetwork
from net.builder import build_model
from net.loss_functions import Studentloss
from net.loss import *

from utils.dataSet import TrainDataset, TestDataSet
from utils.learningRateScheduler import warmUpLearningRate

from warmup_scheduler import GradualWarmupScheduler
from torch.optim.lr_scheduler import StepLR

from early_stopping import EarlyStopping






def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)

setup_seed(2023)




def train_once(net,each_data_batch,epoch,epochs,criterion, optimizer):
    net.train()
    with tqdm(total=each_data_batch, desc=f'Epoch {epoch + 1}/{epochs}', postfix=dict, mininterval=0.3) as pbar:
        each_batch_all_loss = 0
        each_batch_primary_loss = 0
        each_batch_super_loss = 0

        print("Start Train")
        for i, batch_value in enumerate(train_loader):
            input_tesnor = batch_value[0].float().cuda(device=args.device_ids[0])
            gt_tesnor = batch_value[1].float().cuda(device=args.device_ids[0])
            mask_tensor = batch_value[2].float().cuda(device=args.device_ids[0])
            # 多任务学习中，用于区分不同的任务
            task_id_tensor = batch_value[3].float().cuda(device=args.device_ids[0])
            ds_flow = batch_value[4].float().cuda(device=args.device_ids[0])


            # input_tesnor torch.Size([8, 3, 256, 256])
            # mask_tensor torch.Size([8, 1, 256, 256])
            # gt_tesnor torch.Size([8, 3, 256, 256])
            # task_id_tensor torch.Size([8])

            optimizer.zero_grad()
            with autocast():
                flow, warp_image_final, final_image = net.forward(input_tesnor, mask_tensor)

                total_loss, primary_img_loss, super_img_loss = criterion(flow, warp_image_final, final_image,ds_flow, gt_tesnor)


            scaler.scale(total_loss).backward()
            # clip the gradient
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=3, norm_type=2)
            scaler.step(optimizer)
            scaler.update()

            # for param in net.meshRegression.parameters():
            #     if param.requires_grad and param.grad is not None:
            #         print(param.grad.norm())
                    # print(param.grad)

            each_batch_all_loss += total_loss.item() / args.train_batch_size
            each_batch_primary_loss += primary_img_loss.item() / args.train_batch_size
            each_batch_super_loss += super_img_loss.item() / args.train_batch_size
            pbar.set_postfix({'Lsum': each_batch_all_loss / (i + 1),
                              'Lpri': each_batch_primary_loss / (i + 1),
                              'Lsuper': each_batch_super_loss / (i + 1),
                              'lr': scheduler.get_last_lr()[0]})
            pbar.update(1)

        print("\nFinish Train")
        return each_batch_all_loss / each_data_batch



def val_once(net,epoch,epochs,criterion,optimizer):
    net.eval()
    ssim_list = []
    psnr_list = []
    test_num = len(test_loader_list)
    print('test_num:', test_num)
    for index in range(test_num):
        print("Task ID:", index)
        test_each_data_batch = len(test_loader_list[index])
        with tqdm(total=test_each_data_batch, desc=f'Epoch {epoch + 1}/{epochs}', postfix=dict, mininterval=0.3) as t_pbar:
            each_batch_psnr = 0
            each_batch_ssim = 0
            print("Start Test")
            with torch.no_grad():
                # 加载当前任务的数据。
                test_loader = test_loader_list[index]
                for i, batch_value in enumerate(test_loader):
                    input_tesnor = batch_value[0].float().cuda(device=args.device_ids[0])
                    gt_tesnor = batch_value[1].float().cuda(device=args.device_ids[0])
                    mask_tensor = batch_value[2].float().cuda(device=args.device_ids[0])
                    # 多任务学习中，用于区分不同的任务
                    task_id_tensor = batch_value[3].float().cuda(device=args.device_ids[0])

                    optimizer.zero_grad()
                    flow, warp_mask_final, final_image = net.forward(input_tesnor, mask_tensor)

                    I1 = final_image.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
                    I2 = gt_tesnor.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()

                    psnr = compare_psnr(I1, I2, data_range=1)
                    ssim = compare_ssim(I1, I2, data_range=1, channel_axis=2)

                    ssim_list.append(ssim)
                    psnr_list.append(psnr)

                    each_batch_psnr += psnr / args.test_batch_size
                    each_batch_ssim += ssim / args.test_batch_size
                    t_pbar.set_postfix({'average psnr': each_batch_psnr / (i + 1),
                                        'average ssim': each_batch_ssim / (i + 1)})
                    t_pbar.update(1)
    print("\nFinish Test")
    print('average psnr:', np.mean(psnr_list))
    print('average ssim:', np.mean(ssim_list))




def train(net,saveModelName,criterion,optimizer,scheduler,start_epochs=0, end_epochs=1):

    loss_history = []
    switch_epoch = 50
    ssim_temp = 0

    train_each_data_batch = len(train_loader)

    # save_path = '../model/' # 当前目录下
    # early_stopping = EarlyStopping(save_path)

    for epoch in range(start_epochs,end_epochs):
        # 开始计时
        start_time = time.time()  # 记录训练开始时间
        # training
        each_batch_all_loss = train_once(net,train_each_data_batch,epoch, end_epochs,criterion,optimizer)

        # testing
        if epoch % 1 == 0:
            val_once(net, epoch, end_epochs, criterion, optimizer)

            # 结束计时
        end_time = time.time()  # 记录训练结束时间

        # 计算训练时长
        elapsed_time = end_time - start_time  # 训练用时（秒）
        # 转换为时分秒格式
        hours = int(elapsed_time // 3600)
        minutes = int((elapsed_time % 3600) // 60)
        seconds = int(elapsed_time % 60)
        # 输出训练时间
        print(f"Training Time: {hours:02}:{minutes:02}:{seconds:02}")

        # learning rate scheduler
        scheduler.step()
        # print("epoch:",epoch,"lr:",scheduler.get_last_lr())
        loss_history.append(each_batch_all_loss)

        if (epoch + 1) % 10 ==0 or epoch >= int(end_epochs-5):
            torch.save(net.state_dict(), saveModelName + "_" + "epoch" + str(epoch + 1) + ".pkl")
            np.save(saveModelName + "_" + "epoch" + str(epoch + 1) + "_" + "TrainLoss" +
            str(round(each_batch_all_loss, 3)), np.array(loss_history))

        # if early_stopping.state:
        #     final_ssim = val_once(net, epoch, end_epochs, criterion, optimizer)
        #     # 早停止
        #     early_stopping(final_ssim, net)
        #     # 达到早停止条件时，early_stop会被置为True
        #     if early_stopping.early_stop:
        #         print("Early stopping")
        #         break  # 跳出迭代，结束训练





if __name__ == "__main__":

    print('<==================== setting arguments ===================>\n')

    parser = argparse.ArgumentParser()
    '''Implementation details'''
    parser.add_argument('--train_batch_size', type=int, default=8)
    parser.add_argument('--test_batch_size', type=int, default=1)
    parser.add_argument('--start_epochs', type=int, default=0)
    parser.add_argument('--end_epochs', type=int, default=200)
    parser.add_argument('--device_ids', type=list, default=[0])
    parser.add_argument('-w', '--warmup', type=bool, default=True)
    parser.add_argument('--warmup_epochs', type=int, default=3, help='epochs for warmup')
    parser.add_argument("--eta_min", type=float, default=1e-6, help="final learning rate")



    '''Network details'''
    parser.add_argument('--img_h', type=int, default=384) # 384
    parser.add_argument('--img_w', type=int, default=512)
    parser.add_argument('--lr', type=float, default=2e-4, help='learning rate')
    parser.add_argument('--save_model_name', type=str, default='../model/distill_model')
    parser.add_argument('--lam_perception', type=float, default=0.2)
    parser.add_argument('--lam_mask', type=float, default=1)
    parser.add_argument('--lam_mesh', type=float, default=1)
    parser.add_argument('--lam_appearance', type=float, default=1)
    parser.add_argument('--resume', type=bool, default=False)
    parser.add_argument('--resume_epoch', type=int, default=0)

    '''Dataset settings'''
    parser.add_argument('--train_path', type=str,
                        default=['../../Dataset/DIR-D/training/'])
    parser.add_argument('--test_path', type=str,
                        default=['../../Dataset/DIR-D/testing/'])

    args = parser.parse_args()
    print(args)



    train_dataset = TrainDataset(args.train_path)
    train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=4, pin_memory=True)

    test_loader_list = [DataLoader(dataset=TestDataSet(test_path, i), batch_size=args.test_batch_size, num_workers=4, shuffle=False, pin_memory=True, drop_last=False) \
                        for i, test_path in enumerate(args.test_path)]
    for i, test_path in enumerate(args.test_path):
        print(f'Task ID:{i}对应任务:{test_path}')

    # define somethings
    criterion = Studentloss(args.lam_appearance, args.lam_perception, args.lam_mask,
                                                args.lam_mesh).cuda(device=args.device_ids[0])
    net = StudentNetwork()
    print(net)
    # loading model to device 0
    net = net.to(device=args.device_ids[0])
    # vgg_model = models.vgg19(weights=models.VGG19_Weights.DEFAULT)
    # vgg_model = vgg_model.to(device=args.device_ids[0])
    # vgg_model.eval()

    optimizer = torch.optim.AdamW(net.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-4)
    # 动混合精度训练
    scaler = torch.cuda.amp.GradScaler()
    lrScheduler = warmUpLearningRate(args.end_epochs, warm_up_epochs=10, scheduler='cosine')
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lrScheduler)
    # if args.warmup:
    #     # 学习率在训练过程中呈余弦退火的方式变化
    #     scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.end_epochs - args.warmup_epochs,
    #                                                             eta_min=args.eta_min)
    #     scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=args.warmup_epochs,
    #                                        after_scheduler=scheduler_cosine)
    #     scheduler.step()
    # else:
    #     # 每经过 step_size（这里是50）个epoch，学习率乘以 gamma=0.5
    #     step = 50
    #     print("Using StepLR,step={}!".format(step))
    #     scheduler = StepLR(optimizer, step_size=step, gamma=0.5)
    #     scheduler.step()


    if args.resume:
        for i in range(0, (args.start_epochs + 1)):
            scheduler.step()
        args.start_epochs = args.resume_epoch
        load_path = args.save_model_name + "_" + "epoch" + str(args.resume_epoch) + ".pkl"

        pretrain_model = torch.load(load_path, map_location='cpu')
        # Extract K,V from the existing model
        model_dict = net.state_dict()
        # Create a new weight dictionary and update it
        state_dict = {k: v for k, v in pretrain_model.items() if k in model_dict.keys()}
        # Update the weight dictionary of the existing model
        model_dict.update(state_dict)
        # Load the updated weight dictionary
        net.load_state_dict(model_dict)
        print("-----resume train, load model state dict success------")

        # Load optimizer state
        if 'optimizer' in pretrain_model:
            optimizer.load_state_dict(pretrain_model['optimizer'])
            print("-----resume train, load optimizer state dict success------")

        # Load scheduler state
        if 'scheduler' in pretrain_model:
            scheduler.load_state_dict(pretrain_model['scheduler'])
            print("-----resume train, load scheduler state dict success------")

        # Ensure the learning rate scheduler and optimizer are in sync with the resume epoch
        for epoch in range(args.resume_epoch + 1):
            scheduler.step()

    # start train
    train(net, args.save_model_name, criterion, optimizer, scheduler, args.start_epochs, args.end_epochs)









