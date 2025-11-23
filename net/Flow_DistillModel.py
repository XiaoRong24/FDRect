import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import utils.tf_spatial_transform_local as tf_spatial_transform_local
import utils.torch_tps_transform as torch_tps_transform
import utils.tf_mesh2flow as tf_mesh2flow
import utils.constant as constant
import utils.torch_tps2flow as torch_tps2flow
import time
from timm.layers.helpers import to_2tuple
from timm.models.layers import DropPath, trunc_normal_
from net.mambablock import MambaBlock, Conv2d_BN, RepDW

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

def H2Mesh(H, rigid_mesh):

    H_inv = torch.inverse(H)
    ori_pt = rigid_mesh.reshape(rigid_mesh.size()[0], -1, 2)
    ones = torch.ones(rigid_mesh.size()[0], (grid_h+1)*(grid_w+1),1)
    ori_pt = ori_pt.to(gpu_device)
    ones = ones.to(gpu_device)

    ori_pt = torch.cat((ori_pt, ones), 2) # bs*(grid_h+1)*(grid_w+1)*3
    tar_pt = torch.matmul(H_inv, ori_pt.permute(0,2,1)) # bs*3*(grid_h+1)*(grid_w+1)

    mesh_x = torch.unsqueeze(tar_pt[:,0,:]/tar_pt[:,2,:], 2)
    mesh_y = torch.unsqueeze(tar_pt[:,1,:]/tar_pt[:,2,:], 2)
    mesh = torch.cat((mesh_x, mesh_y), 2).reshape([rigid_mesh.size()[0], grid_h+1, grid_w+1, 2])

    return mesh


def autopad2(k, p=None, d=1):  # kernel, padding, dilation
    # Pad to 'same' shape outputs
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class Conv(nn.Module):
    # Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad2(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

def autopad(k, p=None):
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p

class RepConv(nn.Module):
    # Represented convolution
    # https://arxiv.org/abs/2101.03697
    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, act=nn.SiLU(), deploy=False):
        super(RepConv, self).__init__()
        self.deploy = deploy
        self.groups = g
        self.in_channels = c1
        self.out_channels = c2

        assert k == 3
        assert autopad(k, p) == 1

        padding_11 = autopad(k, p) - k // 2
        self.act = nn.LeakyReLU(0.1, inplace=True) if act is True else (
            act if isinstance(act, nn.Module) else nn.Identity())

        if deploy:
            self.rbr_reparam = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=True)
        else:
            self.rbr_identity = (
                nn.BatchNorm2d(num_features=c1, eps=0.001, momentum=0.03) if c2 == c1 and s == 1 else None)
            self.rbr_dense = nn.Sequential(
                nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False),
                nn.BatchNorm2d(num_features=c2, eps=0.001, momentum=0.03),
            )
            self.rbr_1x1 = nn.Sequential(
                nn.Conv2d(c1, c2, 1, s, padding_11, groups=g, bias=False),
                nn.BatchNorm2d(num_features=c2, eps=0.001, momentum=0.03),
            )

    def forward(self, inputs):
        if hasattr(self, "rbr_reparam"):
            return self.act(self.rbr_reparam(inputs))
        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)
        return self.act(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out)

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
        return (
            kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid,
            bias3x3 + bias1x1 + biasid,
        )

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, nn.Sequential):
            kernel = branch[0].weight
            running_mean = branch[1].running_mean
            running_var = branch[1].running_var
            gamma = branch[1].weight
            beta = branch[1].bias
            eps = branch[1].eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, "id_tensor"):
                input_dim = self.in_channels // self.groups
                kernel_value = np.zeros(
                    (self.in_channels, input_dim, 3, 3), dtype=np.float32
                )
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def repvgg_convert(self):
        kernel, bias = self.get_equivalent_kernel_bias()
        return (
            kernel.detach().cpu().numpy(),
            bias.detach().cpu().numpy(),
        )

    def fuse_conv_bn(self, conv, bn):
        std = (bn.running_var + bn.eps).sqrt()
        bias = bn.bias - bn.running_mean * bn.weight / std

        t = (bn.weight / std).reshape(-1, 1, 1, 1)
        weights = conv.weight * t

        bn = nn.Identity()
        conv = nn.Conv2d(in_channels=conv.in_channels,
                         out_channels=conv.out_channels,
                         kernel_size=conv.kernel_size,
                         stride=conv.stride,
                         padding=conv.padding,
                         dilation=conv.dilation,
                         groups=conv.groups,
                         bias=True,
                         padding_mode=conv.padding_mode)

        conv.weight = torch.nn.Parameter(weights)
        conv.bias = torch.nn.Parameter(bias)
        return conv

    def fuse_repvgg_block(self):
        if self.deploy:
            return
        print(f"RepConv.fuse_repvgg_block")
        self.rbr_dense = self.fuse_conv_bn(self.rbr_dense[0], self.rbr_dense[1])

        self.rbr_1x1 = self.fuse_conv_bn(self.rbr_1x1[0], self.rbr_1x1[1])
        rbr_1x1_bias = self.rbr_1x1.bias
        weight_1x1_expanded = torch.nn.functional.pad(self.rbr_1x1.weight, [1, 1, 1, 1])

        # Fuse self.rbr_identity
        if (isinstance(self.rbr_identity, nn.BatchNorm2d) or isinstance(self.rbr_identity,
                                                                        nn.modules.batchnorm.SyncBatchNorm)):
            identity_conv_1x1 = nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                groups=self.groups,
                bias=False)
            identity_conv_1x1.weight.data = identity_conv_1x1.weight.data.to(self.rbr_1x1.weight.data.device)
            identity_conv_1x1.weight.data = identity_conv_1x1.weight.data.squeeze().squeeze()
            identity_conv_1x1.weight.data.fill_(0.0)
            identity_conv_1x1.weight.data.fill_diagonal_(1.0)
            identity_conv_1x1.weight.data = identity_conv_1x1.weight.data.unsqueeze(2).unsqueeze(3)

            identity_conv_1x1 = self.fuse_conv_bn(identity_conv_1x1, self.rbr_identity)
            bias_identity_expanded = identity_conv_1x1.bias
            weight_identity_expanded = torch.nn.functional.pad(identity_conv_1x1.weight, [1, 1, 1, 1])
        else:
            bias_identity_expanded = torch.nn.Parameter(torch.zeros_like(rbr_1x1_bias))
            weight_identity_expanded = torch.nn.Parameter(torch.zeros_like(weight_1x1_expanded))

        self.rbr_dense.weight = torch.nn.Parameter(
            self.rbr_dense.weight + weight_1x1_expanded + weight_identity_expanded)
        self.rbr_dense.bias = torch.nn.Parameter(self.rbr_dense.bias + rbr_1x1_bias + bias_identity_expanded)

        self.rbr_reparam = self.rbr_dense
        self.deploy = True

        if self.rbr_identity is not None:
            del self.rbr_identity
            self.rbr_identity = None

        if self.rbr_1x1 is not None:
            del self.rbr_1x1
            self.rbr_1x1 = None

        if self.rbr_dense is not None:
            del self.rbr_dense
            self.rbr_dense = None


class C2f(nn.Module):
    # CSP Bottleneck with 2 convolutions
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):  # ch_in, ch_out, shortcut, groups, kernels, expand
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return self.cv2(self.cv1(x))



class ConvBlock(nn.Module):

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))




def window_partition(x, window_size):
    """
    Args:
        x: (B, C, H, W)
        window_size: window size
        h_w: Height of window
        w_w: Width of window
    Returns:
        local window features (num_windows*B, window_size*window_size, C)
    """
    B, C, H, W = x.shape
    x = x.view(B, C, H // window_size, window_size, W // window_size, window_size)
    windows = x.permute(0, 2, 4, 3, 5, 1).reshape(-1, window_size*window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: local window features (num_windows*B, window_size, window_size, C)
        window_size: Window size
        H: Height of image
        W: Width of image
    Returns:
        x: (B, C, H, W)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.reshape(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 5, 1, 3, 2, 4).reshape(B,windows.shape[2], H, W)
    return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x



class Attention(nn.Module):

    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            qk_norm=False,
            attn_drop=0.,
            proj_drop=0.,
            norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = True

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)

        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        attn = (q @ k.transpose(-2, -1)) * self.scale

        # For rare cases, the attention weights are inf due to the mix-precision training.
        # We clamp the tensor to the max values of the current data type
        # This is different from MAE training as we don't observe such cases on image-only MAE.
        if torch.isinf(attn).any():
            clamp_value = torch.finfo(attn.dtype).max - 1000
            attn = torch.clamp(attn, min=-clamp_value, max=clamp_value)

        attn = attn.softmax(dim=-1)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x




class Block(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=False,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 Mlp_block=Mlp,
                 layer_scale=None,
                 task_num=9,
                 noisy_gating=True,
                 att_w_topk_loss=0.0, att_limit_k=0,
                 cvloss=0, switchloss=0.01 * 1, zloss=0.001 * 1, w_topk_loss=0.0, limit_k=0,
                 w_MI=0., num_attn_experts=24, sample_topk=0, moe_type='normal'
                 ):
        super().__init__()
        self.norm1 = norm_layer(dim)


        self.mixer = Attention(
        dim,
        num_heads=num_heads,
        qkv_bias=qkv_bias,
        qk_norm=qk_scale,
        attn_drop=attn_drop,
        proj_drop=drop,
        norm_layer=norm_layer,
        )


        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp_block(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        use_layer_scale = layer_scale is not None and type(layer_scale) in [int, float]
        self.gamma_1 = nn.Parameter(layer_scale * torch.ones(dim))  if use_layer_scale else 1
        self.gamma_2 = nn.Parameter(layer_scale * torch.ones(dim))  if use_layer_scale else 1

    def forward(self, x):
        x = x + self.drop_path(self.gamma_1 * self.mixer(self.norm1(x)))
        x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x



class SelfAttentionLayer(nn.Module):
    """
    MambaVision layer"
    """

    def __init__(self,
                 dim,
                 num_heads,
                 window_size,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 layer_scale=None,
                 transformer_blocks=1,
    ):


        super().__init__()

        self.blocks = nn.ModuleList([Block(dim=dim,
                                           num_heads=num_heads,
                                           mlp_ratio=mlp_ratio,
                                           qkv_bias=qkv_bias,
                                           qk_scale=qk_scale,
                                           drop=drop,
                                           attn_drop=attn_drop,
                                           drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                           layer_scale=layer_scale)
                                     for i in range(transformer_blocks)])

        self.do_gt = False
        self.window_size = window_size

    def forward(self, x):
        _, _, H, W = x.shape


        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        if pad_r > 0 or pad_b > 0:

            x = torch.nn.functional.pad(x, (0,pad_r,0,pad_b))

            _, _, Hp, Wp = x.shape
        else:
            Hp, Wp = H, W

        x = window_partition(x, self.window_size)

        for _, blk in enumerate(self.blocks):
            x = blk(x)

        x = window_reverse(x, self.window_size, Hp, Wp)

        if pad_r > 0 or pad_b > 0:
            x = x[:, :, :H, :W].contiguous()

        return x

class Downsample(nn.Module):
    """
    Patch Embedding that is implemented by a layer of conv.
    Input: tensor in shape [B, C, H, W]
    Output: tensor in shape [B, C, H/stride, W/stride]
    """

    def __init__(self,in_chans=3, embed_dim=48, patch_size=16, stride=2, padding=0):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        stride = to_2tuple(stride)
        padding = to_2tuple(padding)
        self.proj = Conv2d_BN(in_chans, embed_dim, patch_size, stride, padding)

    def forward(self, x):
        x = self.proj(x)
        return x




class FeatureExtractor(nn.Module):
    def __init__(self,inchannels=3,mlp_ratio=4):
        super(FeatureExtractor, self).__init__()
        self.mlp_ratio = mlp_ratio
        # Conv
        self.featureExtractor = nn.Sequential(
            Downsample(inchannels, 64, 3, 2, 1),  # 512 -> 256
            Downsample(64, 64, 3, 2, 1),  # 256 -> 128
            ConvBlock(64, 64),
            Downsample(64, 64, 3, 2, 1),  # 128 -> 64
            ConvBlock(64,64),
            Downsample(64, 64, 3, 2, 1),  # 64 -> 32
            MambaBlock(64, ssm_d_state=8, ssm_ratio=1.0, ssm_conv_bias=False, window_size=8),
            SelfAttentionLayer(dim=64, num_heads=2, window_size=8, mlp_ratio=mlp_ratio, qkv_bias=True,
                               drop=0, attn_drop=0, drop_path=0, layer_scale=None),
            Downsample(64, 64, 3, 2, 1),  # 32 -> 16
            ConvBlock(64, 64),
            Downsample(64, 64, 3, 2, 1),  # 16 -> 8
            MambaBlock(64, ssm_d_state=8, ssm_ratio=1.0, ssm_conv_bias=False, window_size=2),
            # SelfAttentionLayer(dim=64, num_heads=2, window_size=8, mlp_ratio=mlp_ratio, qkv_bias=True,
            #                    drop=0, attn_drop=0, drop_path=0, layer_scale=None),
            Downsample(64, 64, 3, 2, 1),  # 8 -> 4
        )

    def forward(self,x):
        x = self.featureExtractor(x)
        return x


class RegressionNetwork(nn.Module):
    def __init__(self):
        super(RegressionNetwork, self).__init__()
        self.patch_height = (grid_h + 1)
        self.patch_width = (grid_w + 1)


        self.head = nn.Sequential(
            nn.Conv2d(64,1024,(3,4),2),
            nn.Flatten(),
            # nn.Linear(768, 1024),
            nn.SiLU(inplace=True),
            #nn.GELU(),
            nn.Linear(1024, self.patch_height * self.patch_width * 2)
        )

    def forward(self,x):
        x = self.head(x)
        return x.view(-1,self.patch_height, self.patch_width, 2)




class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        # 定义各个卷积块（卷积层 + 激活层 + 池化层）

        # Conv Block 1
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU()
        )

        # Conv Block 2
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU()
        )

        # Conv Block 3
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU()
        )

        # Conv Block 4
        self.conv_block4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU()
        )

        # Conv Block 5
        self.conv_block5 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU()
        )

    def forward(self, x):
        # 前向传播
        feature = []

        # Conv Block 1
        x = self.conv_block1(x)
        feature.append(x)
        x = F.max_pool2d(x, 2, 2)  # H/2

        # Conv Block 2
        x = self.conv_block2(x)
        feature.append(x)
        x = F.max_pool2d(x, 2, 2)  # H/4

        # Conv Block 3
        x = self.conv_block3(x)
        feature.append(x)
        x = F.max_pool2d(x, 2, 2)  # H/8

        # Conv Block 4
        x = self.conv_block4(x)
        feature.append(x)
        x = F.max_pool2d(x, 2, 2)  # H/16

        # Conv Block 5
        x = self.conv_block5(x)
        feature.append(x)

        return feature




class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.deconv1 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2)
        self.conv1_1 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)

        self.deconv2 = nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=2, stride=2)
        self.conv2_1 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)

        self.deconv3 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2)
        self.conv3_1 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)

        self.deconv4 = nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=2, stride=2)
        self.conv4_1 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)

        self.flow = nn.Conv2d(in_channels=64, out_channels=2, kernel_size=1, padding=0)

    def forward(self, feature):
        # Apply the first block of deconvolution and convolutions
        h_deconv1 = self.deconv1(feature[-1])
        h_deconv_concat1 = torch.cat([feature[-2], h_deconv1], dim=1)
        conv1 = self.conv1_1(h_deconv_concat1)
        conv1 = self.conv1_2(conv1)

        # Apply the second block of deconvolution and convolutions
        h_deconv2 = self.deconv2(conv1)
        h_deconv_concat2 = torch.cat([feature[-3], h_deconv2], dim=1)
        conv2 = self.conv2_1(h_deconv_concat2)
        conv2 = self.conv2_2(conv2)

        # Apply the third block of deconvolution and convolutions
        h_deconv3 = self.deconv3(conv2)
        h_deconv_concat3 = torch.cat([feature[-4], h_deconv3], dim=1)
        conv3 = self.conv3_1(h_deconv_concat3)
        conv3 = self.conv3_2(conv3)

        # Apply the fourth block of deconvolution and convolutions
        h_deconv4 = self.deconv4(conv3)
        h_deconv_concat4 = torch.cat([feature[-5], h_deconv4], dim=1)
        conv4 = self.conv4_1(h_deconv_concat4)
        conv4 = self.conv4_2(conv4)

        # Generate flow with a 1x1 convolution
        flow = self.flow(conv4)

        return flow



class FDRect(nn.Module):
    def __init__(self):
        super(FDRect, self).__init__()
        self.FeatureEncoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16, eps=0.001, momentum=0.03),
            nn.SiLU(inplace=True),

            nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32, eps=0.001, momentum=0.03),
            nn.SiLU(inplace=True),

            nn.Conv2d(32, 3, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(3, eps=0.001, momentum=0.03),
            nn.SiLU(inplace=True),
        )

        self.feature_extractor = FeatureExtractor(3)
        self.regression = RegressionNetwork()
        self.encoder = Encoder()
        self.decoder = Decoder()


    def forward(self, input_img, mask_img):
        batch_size, _, height, width = input_img.shape

        f_input_img = self.FeatureEncoder(input_img)

        feature = torch.mul(f_input_img, mask_img)
        feature = self.feature_extractor(feature)
        mesh_motion = self.regression(feature)

        rigid_mesh = get_rigid_mesh(batch_size, height, width)
        H_one = torch.eye(3)
        H = torch.tile(H_one.unsqueeze(0), [batch_size, 1, 1]).to(gpu_device)
        ini_mesh = H2Mesh(H, rigid_mesh)
        mesh_final = ini_mesh + mesh_motion
        norm_rigid_mesh = get_norm_mesh(rigid_mesh, height, width)
        norm_mesh = get_norm_mesh(mesh_final, height, width)

        warp_output = torch_tps_transform.transformer(torch.cat((input_img, mask_img), 1), norm_rigid_mesh, norm_mesh,(height, width))
        # output_flow = torch_tps2flow.transformer(torch.cat((input_img, mask_img), 1), norm_rigid_mesh, norm_mesh,
        #                                          (height, width))
        # warp_output = warp_with_flow(torch.cat((input_img, mask_img), 1), output_flow)
        warp_image_final = warp_output[:, 0:3, ...]
        warp_mask_final = warp_output[:, 3:6, ...]
        warp_image_final = torch.mul(warp_image_final, warp_mask_final)

        '''convert TPS deformation to optical flows (image resolution: 384*512)'''
        rigid_mesh = get_rigid_mesh(batch_size, height, width)
        norm_rigid_mesh = get_norm_mesh(rigid_mesh, height, width)
        pre_mesh = rigid_mesh + mesh_motion
        norm_pre_mesh = get_norm_mesh(pre_mesh, height, width)
        delta_flow = torch_tps2flow.transformer(input_img, norm_rigid_mesh, norm_pre_mesh, (height, width))

        ''' residue flow '''
        residue_feature = self.encoder(warp_image_final)
        # for i in range(len(residue_feature)):
        #     print(f'residue_feature[{i}]: {residue_feature[i].shape}')
        residual_flow = self.decoder(residue_feature)


        warp_flow1 = warp_with_flow(delta_flow,delta_flow)
        flow1 = delta_flow + warp_flow1
        warp_flow2 = warp_with_flow(flow1,residual_flow)
        flow = residual_flow + warp_flow2
        # flow = delta_flow + residual_flow
        final_image = warp_with_flow(input_img, flow)
        final_mask = warp_with_flow(mask_img, flow)

        return flow, final_mask, final_image#,delta_flow,residual_flow,warp_image_final




def tensor_DLT(src_p, dst_p):
    bs, _, _ = src_p.shape

    ones = torch.ones(bs, 4, 1)
    if torch.cuda.is_available():
        ones = ones.to(gpu_device)
    xy1 = torch.cat((src_p, ones), 2)
    zeros = torch.zeros_like(xy1)
    if torch.cuda.is_available():
        zeros = zeros.to(gpu_device)

    xyu, xyd = torch.cat((xy1, zeros), 2), torch.cat((zeros, xy1), 2)
    M1 = torch.cat((xyu, xyd), 2).reshape(bs, -1, 6)
    M2 = torch.matmul(
        dst_p.reshape(-1, 2, 1),
        src_p.reshape(-1, 1, 2),
    ).reshape(bs, -1, 2)

    # Ah = b
    A = torch.cat((M1, -M2), 2)
    b = dst_p.reshape(bs, -1, 1)

    # h = A^{-1}b
    Ainv = torch.inverse(A)
    h8 = torch.matmul(Ainv, b).reshape(bs, 8)

    H = torch.cat((h8, ones[:, 0, :]), 1).reshape(bs, 3, 3)
    return H



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




def test_mamba_vision_L():
    # 创建模型，传递一些自定义的参数，或者使用默认参数
    model = RectanglingNetwork().cuda()

    # 打印模型结构，验证是否成功初始化
    print("Model structure:")
    print(model)

    # 测试模型是否能正常前向传播
    try:
        # 假设输入是一个分辨率为 224x224 的图像，批次大小为 1，通道数为 3
        dummy_input = torch.randn(8, 3, 512, 384).cuda() # 批次大小为 1，RGB 图像 224x224
        output = model(dummy_input)  # 执行一次前向传播

        for i in range(len(output)):
            print(f"Output shape{i}:", output[i].shape)  # 打印输出形状

    except Exception as e:
        print("Error during forward pass:", e)


# 调用测试函数
if __name__ == "__main__":
    test_mamba_vision_L()