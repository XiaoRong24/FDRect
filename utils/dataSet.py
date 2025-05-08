import os
import numpy as np
import cv2
import torch
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import utils.constant as constant
import glob
from collections import OrderedDict
import random
import torchvision.transforms.functional as F
from random import random

grid_w = constant.GRID_W
grid_h = constant.GRID_H

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)


class SPRectanglingTestDataSet(Dataset):
    def __init__(self,input_path,mask_path,gt_path,resize_h,resize_w):
        super(SPRectanglingTestDataSet, self).__init__()
        self.index_all = list(sorted([x.split('.')[0] for x in  os.listdir(input_path)]))
        self.input_path = input_path
        self.mask_path = mask_path
        self.gt_path = gt_path
        self.resize_h = resize_h
        self.resize_w = resize_w
        self._origin_transform = transforms.Compose([
            transforms.Resize([self.resize_h, self.resize_w]),
            transforms.ToTensor(),
        ])
        self._origin_transform2 = transforms.Compose([
            transforms.Resize([384, 512]),
            transforms.ToTensor(),
        ])
        setup_seed(2023)


    def __len__(self):
        return len(self.index_all)

    def __getitem__(self, idx):
        idx = self.index_all[idx]
        input_img = cv2.imread(os.path.join(self.input_path, idx + '.jpg'))
        mask_img = cv2.imread(os.path.join(self.mask_path, idx + '.jpg'))
        gt_img = cv2.imread(os.path.join(self.gt_path, idx + '.jpg'))

        input_img = Image.fromarray(input_img)
        mask_img = Image.fromarray(mask_img)
        gt_img = Image.fromarray(gt_img)
        ###
        gt_img = self._origin_transform(gt_img)
        input_img = self._origin_transform(input_img)
        mask_img = self._origin_transform(mask_img)
        return input_img,mask_img,gt_img


class SPRectanglingTrainDataSet2TeachWeight(Dataset):
    def __init__(self,input_path,mask_path,gt_path,mesh_path1,mesh_path2,mesh_weight_path1,mesh_weight_path2,resize_h,resize_w):
        super(SPRectanglingTrainDataSet2TeachWeight, self).__init__()
        self.index_all = list(sorted([x.split('.')[0] for x in  os.listdir(input_path)]))
        self.input_path = input_path
        self.mask_path = mask_path
        self.mesh_path1 = mesh_path1
        self.mesh_path2 = mesh_path2
        self.mesh_weight_path1 = mesh_weight_path1
        self.mesh_weight_path2 = mesh_weight_path2
        self.gt_path = gt_path
        self.resize_h = resize_h
        self.resize_w = resize_w
        self._origin_transform = transforms.Compose([
            transforms.Resize([self.resize_h, self.resize_w]),
            transforms.ToTensor(),
        ])
        self._origin_transform2 = transforms.Compose([
            transforms.Resize([384, 512]),
            transforms.ToTensor(),
        ])
        setup_seed(2023)


    def __len__(self):
        return len(self.index_all)

    def __getitem__(self, idx):
        idx = self.index_all[idx]

        input_img = cv2.imread(os.path.join(self.input_path, idx + '.jpg'))
        mask_img = cv2.imread(os.path.join(self.mask_path, idx + '.jpg'))
        gt_img = cv2.imread(os.path.join(self.gt_path, idx + '.jpg'))
        ds_mesh1 = np.load(os.path.join(self.mesh_path1, idx + '.npy'), allow_pickle=True)#[0,:,:,:]
        ds_mesh2 = np.load(os.path.join(self.mesh_path2, idx + '.npy'), allow_pickle=True)  # [0,:,:,:]
        ds_weight_mesh1 = np.load(os.path.join(self.mesh_weight_path1, idx + '.npy'), allow_pickle=True)[1]
        ds_weight_mesh2 = np.load(os.path.join(self.mesh_weight_path2, idx + '.npy'), allow_pickle=True)[1]
        # print("ds_mesh:",ds_weight_mesh1.shape)

        input_img = Image.fromarray(input_img)
        mask_img = Image.fromarray(mask_img)
        gt_img = Image.fromarray(gt_img)

        gt_img = self._origin_transform(gt_img)
        input_img = self._origin_transform(input_img)
        mask_img = self._origin_transform(mask_img)

        return input_img,mask_img,gt_img,ds_mesh1,ds_mesh2,ds_weight_mesh1,ds_weight_mesh2


class FlowTrainDataSetTeachWeight(Dataset):
    def __init__(self,input_path,mask_path,gt_path, flow_path3, resize_h,resize_w):
        super(FlowTrainDataSetTeachWeight, self).__init__()
        self.index_all = list(sorted([x.split('.')[0] for x in os.listdir(input_path)]))
        self.input_path = input_path
        self.mask_path = mask_path
        self.flow_path3 = flow_path3


        self.gt_path = gt_path
        self.resize_h = resize_h
        self.resize_w = resize_w
        self._origin_transform = transforms.Compose([
            transforms.Resize([self.resize_h, self.resize_w]),
            transforms.ToTensor(),
        ])
        self._origin_transform2 = transforms.Compose([
            transforms.Resize([384, 512]),
            transforms.ToTensor(),
        ])
        setup_seed(2023)


    def __len__(self):
        return len(self.index_all)

    def __getitem__(self, idx):
        idx = self.index_all[idx]

        input_img = cv2.imread(os.path.join(self.input_path, idx + '.jpg'))
        mask_img = cv2.imread(os.path.join(self.mask_path, idx + '.jpg'))
        gt_img = cv2.imread(os.path.join(self.gt_path, idx + '.jpg'))

        ds_flow3 = np.load(os.path.join(self.flow_path3, idx + '.npy'), allow_pickle=True)

        # print("ds_mesh:",ds_weight_mesh1.shape)

        input_img = Image.fromarray(input_img)
        mask_img = Image.fromarray(mask_img)
        gt_img = Image.fromarray(gt_img)

        gt_img = self._origin_transform(gt_img)
        input_img = self._origin_transform(input_img)
        mask_img = self._origin_transform(mask_img)

        return input_img,mask_img,gt_img,ds_flow3



class GeneralTrainDataSet(Dataset):
    def __init__(self,datapath, resize_h,resize_w):
        self.width = resize_w
        self.height = resize_h
        self.prob = 0.5
        self.input_images = []
        self.gt_images = []
        self.masks = []
        self.flows = []
        self.task_id = []
        for index, path in enumerate(datapath):
            inputs = glob.glob(os.path.join(path, 'input/', '*.*'))
            gts = glob.glob(os.path.join(path, 'gt/', '*.*'))
            masks = glob.glob(os.path.join(path, 'mask/', '*.*'))
            flows = glob.glob(os.path.join(path, 'distill_flow/', '*.*'))
            inputs.sort()
            gts.sort()
            masks.sort()
            flows.sort()

            lens = len(inputs)
            index_array = [index] * lens
            self.task_id.extend(index_array)
            self.input_images.extend(inputs)
            self.gt_images.extend(gts)
            self.masks.extend(masks)
            self.flows.extend(flows)

        print("total dataset num: ", len(self.input_images))



    def __getitem__(self, index):
        # load image1
        input = cv2.imread(self.input_images[index])
        input = cv2.resize(input, (self.width, self.height))
        input = input.astype(dtype=np.float32)
        input = (input / 127.5) - 1.0
        input = np.transpose(input, [2, 0, 1])

        # load image1
        mask = cv2.imread(self.masks[index], cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (self.width, self.height))
        mask = np.expand_dims(mask, axis=-1)
        mask = mask.astype(dtype=np.float32)
        mask = mask / 255.0
        mask = np.transpose(mask, [2, 0, 1])

        # load image2
        gt = cv2.imread(self.gt_images[index])
        gt = cv2.resize(gt, (self.width, self.height))
        gt = gt.astype(dtype=np.float32)
        gt = (gt / 127.5) - 1.0
        gt = np.transpose(gt, [2, 0, 1])

        # convert to tensor
        input_img = torch.tensor(input)
        mask_img = torch.tensor(mask)
        gt_img = torch.tensor(gt)

        '''load flow'''
        flow = np.load(self.flows[index])
        flow = flow.astype(dtype=np.float32)
        ds_flow3 = torch.tensor(flow)

        return input_img, mask_img,gt_img,ds_flow3

    def __len__(self):
        return len(self.input_images)



class GeneralTestDataSet(Dataset):
    def __init__(self,datapath, resize_h,resize_w):
        self.width = resize_w
        self.height = resize_h
        self.prob = 0.5
        self.input_images = []
        self.gt_images = []
        self.masks = []
        self.flows = []
        self.task_id = []
        for index, path in enumerate(datapath):
            inputs = glob.glob(os.path.join(path, 'input/', '*.*'))
            gts = glob.glob(os.path.join(path, 'gt/', '*.*'))
            masks = glob.glob(os.path.join(path, 'mask/', '*.*'))
            inputs.sort()
            gts.sort()
            masks.sort()

            self.input_images.extend(inputs)
            self.gt_images.extend(gts)
            self.masks.extend(masks)

        print("total dataset num: ", len(self.input_images))



    def __getitem__(self, index):
        # load image1
        input = cv2.imread(self.input_images[index])
        input = cv2.resize(input, (self.width, self.height))
        input = input.astype(dtype=np.float32)
        input = (input / 127.5) - 1.0
        input = np.transpose(input, [2, 0, 1])

        # load image1
        mask = cv2.imread(self.masks[index], cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (self.width, self.height))
        mask = np.expand_dims(mask, axis=-1)
        mask = mask.astype(dtype=np.float32)
        mask = mask / 255.0
        mask = np.transpose(mask, [2, 0, 1])

        # load image2
        gt = cv2.imread(self.gt_images[index])
        gt = cv2.resize(gt, (self.width, self.height))
        gt = gt.astype(dtype=np.float32)
        gt = (gt / 127.5) - 1.0
        gt = np.transpose(gt, [2, 0, 1])

        # convert to tensor
        input_img = torch.tensor(input)
        mask_img = torch.tensor(mask)
        gt_img = torch.tensor(gt)

        return input_img, mask_img,gt_img

    def __len__(self):
        return len(self.input_images)

