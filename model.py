import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from modules import *
from NONLOCAL.non_local_embedded_gaussian import NONLocalBlock2D
import scipy.io as sio
from basicsr.archs.arch_util import to_2tuple, trunc_normal_

device = 'cuda'



class InSSSRBlock(nn.Module):
    def __init__(self, in_channels, out_channels, ratio):
        super(InSSSRBlock, self).__init__()

        self.alpha_hidden = nn.Parameter(torch.rand(1, requires_grad=True, device=device))
        self.beta = nn.Parameter(torch.randn(1, requires_grad=True, device=device))
        self.gamma = nn.Parameter(torch.randn(1, requires_grad=True, device=device))

        self.upsample1 = Upsample(in_channels, ratio)
        self.upsample2 = Upsample(in_channels, ratio)
        self.upsample3 = Upsample(out_channels, ratio)

        self.downsample1 = Downsample(in_channels, ratio)
        self.downsample2 = Downsample(out_channels, ratio)

        self.nl = NONLocalBlock2D(in_channels=in_channels)

        self.correct = nn.Sequential(
            resblock(out_channels, 3),
            resblock(out_channels, 3),
            resblock(out_channels, 3),
        )

        self.channels_inc = nn.Parameter(torch.randn(out_channels, in_channels, 1, 1, requires_grad=True, device=device))
        self.channels_dec = nn.Parameter(torch.randn(in_channels, out_channels, 1, 1, requires_grad=True, device=device))
        self.conv1x1_1 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=1)
        self.relu = nn.ReLU()

    def forward(self, w):
        alpha = self.alpha_hidden
        ltw = self.upsample1(w)
        d_ltw = self.downsample1(2*alpha*ltw)
        nl_d_wt = self.nl(d_ltw)
        res_y = self.upsample2(nl_d_wt)
        y = self.relu((2*alpha*ltw-(2*alpha/self.beta)*res_y)/self.beta)
        g = F.conv2d(-y, self.channels_inc, stride=1)
        wrt = F.conv2d(w, self.channels_inc, stride=1)
        z = self.relu(self.conv1x1_1(2*(1-alpha)*wrt))
        h = self.upsample3(-z)
        x = self.relu(self.correct(self.beta * g + self.gamma * h))
        xr = F.conv2d(x, self.channels_dec, stride=1)
        u = xr - y
        lx = self.downsample2(x)
        v = lx - z

        return x, y, z, u, v


class MidSSSRBlock(nn.Module):
    def __init__(self, in_channels, out_channels, ratio):
        super(MidSSSRBlock, self).__init__()

        self.alpha_hidden = nn.Parameter(torch.rand(1, requires_grad=True, device=device))
        self.beta = nn.Parameter(torch.randn(1, requires_grad=True, device=device))
        self.gamma = nn.Parameter(torch.randn(1, requires_grad=True, device=device))
        self.eta = nn.Parameter(torch.randn(1, requires_grad=True, device=device))

        self.upsample1 = Upsample(in_channels, ratio)
        self.upsample2 = Upsample(in_channels, ratio)
        self.upsample3 = Upsample(out_channels, ratio)
        self.upsample4 = Upsample(out_channels, ratio)

        self.downsample1 = Downsample(in_channels, ratio)
        self.downsample2 = Downsample(out_channels, ratio)
        self.downsample3 = Downsample(out_channels, ratio)
        self.downsample4 = Downsample(out_channels, ratio)

        self.nl = NONLocalBlock2D(in_channels=in_channels)

        self.correct = nn.Sequential(
            resblock(out_channels, 3),
            resblock(out_channels, 3),
            resblock(out_channels, 3),
        )

        self.channels_inc = nn.Parameter(torch.randn(out_channels, in_channels, 1, 1, requires_grad=True, device=device))
        self.channels_dec = nn.Parameter(torch.randn(in_channels, out_channels, 1, 1, requires_grad=True, device=device))
        self.conv1x1_1 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=1)
        self.relu = nn.ReLU()

    def forward(self, w, x, u, v):
        alpha = self.alpha_hidden
        eta = self.relu(self.eta)
        ltw = self.upsample1(w)

        xkr = F.conv2d(x, self.channels_dec, stride=1)

        y = 2*alpha*ltw + self.beta* (u + xkr)
        d_y = self.downsample1(y)
        nl_d_y = self.nl(d_y)
        res_y = self.upsample2(nl_d_y)

        y = self.relu((y-(2*alpha/self.beta)*res_y)/self.beta)
        g = F.conv2d(u - y, self.channels_inc, stride=1)



        lx = self.downsample2(x)
        ltlx = self.upsample3(lx)

        xkrrt = F.conv2d(xkr, self.channels_inc, stride=1)

        lxk = self.downsample3(x)

        wrt = F.conv2d(w, self.channels_inc, stride=1)
        z = self.gamma * (lxk+v) + 2*(1-alpha)*wrt
        z = self.relu(self.conv1x1_1(z))
        h = self.upsample4(v - z)

        xk1 = self.relu(self.correct(x - eta * ((self.beta * g + self.gamma * ltlx) + (self.beta * xkrrt + self.gamma * h))))

        xk1r = F.conv2d(xk1, self.channels_dec, stride=1)
        uk1 = u + xk1r - y
        lxk1 = self.downsample4(xk1)
        vk1 = v + lxk1 - z

        return xk1, y, z, uk1, vk1


class OutSSSRBlock(nn.Module):
    def __init__(self, in_channels, out_channels, ratio):
        super(OutSSSRBlock, self).__init__()
        self.alpha_hidden = nn.Parameter(torch.rand(1, requires_grad=True, device=device))
        self.beta = nn.Parameter(torch.randn(1, requires_grad=True, device=device))
        self.gamma = nn.Parameter(torch.randn(1, requires_grad=True, device=device))
        self.eta = nn.Parameter(torch.randn(1, requires_grad=True, device=device))

        self.upsample1 = Upsample(in_channels, ratio)
        self.upsample2 = Upsample(in_channels, ratio)
        self.upsample3 = Upsample(out_channels, ratio)
        self.upsample4 = Upsample(out_channels, ratio)

        self.downsample1 = Downsample(in_channels, ratio)
        self.downsample2 = Downsample(out_channels, ratio)
        self.downsample3 = Downsample(out_channels, ratio)

        self.nl = NONLocalBlock2D(in_channels=in_channels)

        self.correct = nn.Sequential(
            resblock(out_channels, 3),
            resblock(out_channels, 3),
            resblock(out_channels, 3),
        )

        self.channels_inc = nn.Parameter(torch.randn(out_channels, in_channels, 1, 1, requires_grad=True, device=device))
        self.channels_dec = nn.Parameter(torch.randn(in_channels, out_channels, 1, 1, requires_grad=True, device=device))
        self.conv1x1_1 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=1)

        self.relu = nn.ReLU()

    def forward(self, w, x, u, v):
        alpha = self.alpha_hidden
        eta = self.relu(self.eta)
        ltw = self.upsample1(w)
        xkr = F.conv2d(x, self.channels_dec, stride=1)
        y = 2*alpha*ltw + self.beta * (u + xkr)
        d_y = self.downsample1(y)
        nl_d_y = self.nl(d_y)
        res_y = self.upsample2(nl_d_y)
        y = self.relu((y-(2*alpha/self.beta)*res_y)/self.beta)
        g = F.conv2d(u - y, self.channels_inc, stride=1)

        lx = self.downsample2(x)
        ltlx = self.upsample3(lx)

        xkrrt = F.conv2d(xkr, self.channels_inc, stride=1)

        lxk = self.downsample3(x)

        wrt = F.conv2d(w, self.channels_inc, stride=1)
        z = self.gamma * (lxk+v) + 2*(1-alpha)*wrt
        z = self.relu(self.conv1x1_1(z))
        h = self.upsample4(v - z)

        xk1 = self.relu(self.correct(x - eta * ((self.beta * g + self.gamma * ltlx) + (self.beta * xkrrt + self.gamma * h))))


        return xk1, y, z


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    From: https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/drop.py
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0], ) + (1, ) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).

    From: https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/drop.py
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class ChannelAttention(nn.Module):
    """Channel attention used in RCAN.
    Args:
        num_feat (int): Channel number of intermediate features.
        squeeze_factor (int): Channel squeeze factor. Default: 16.
    """

    def __init__(self, num_feat, squeeze_factor=16):
        super(ChannelAttention, self).__init__()
        
        
        
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_feat, num_feat // squeeze_factor, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feat // squeeze_factor, num_feat, 1, padding=0),
            nn.Sigmoid())

    def forward(self, x):
        y = self.attention(x)
        return x * y


class ECA(nn.Module):

    def __init__(self, num_feat, compress_ratio=3, squeeze_factor=30):
        super(ECA, self).__init__()
        depth_conv = nn.Conv2d(in_channels=num_feat, out_channels=num_feat, kernel_size=3, padding=1 , groups=num_feat)
        point_conv = nn.Conv2d(in_channels=num_feat, out_channels=num_feat // compress_ratio, kernel_size=1)
        self.depthwise_separable_conv = torch.nn.Sequential(depth_conv, point_conv)
        
        depth_conv = nn.Conv2d(in_channels=num_feat// compress_ratio, out_channels=num_feat// compress_ratio, kernel_size=3, padding=1 , groups=num_feat//compress_ratio)
        point_conv = nn.Conv2d(in_channels=num_feat // compress_ratio, out_channels=num_feat, kernel_size=1)
        self.depthwise_separable_conv1 = torch.nn.Sequential(depth_conv, point_conv)
        
        self.ECA = nn.Sequential(
            
            self.depthwise_separable_conv,
            #depthwise_separable_conv(num_feat, num_feat // compress_ratio, kernel_size = 3, padding = 1),
            #nn.Conv2d(num_feat, num_feat // compress_ratio, 3, 1, 1),
            nn.GELU(),
            #depthwise_separable_conv( num_feat // compress_ratio,num_feat, kernel_size = 3, padding=1),
            #nn.Conv2d(num_feat // compress_ratio, num_feat, 3, 1, 1),
            self.depthwise_separable_conv1,
            ChannelAttention(num_feat, squeeze_factor)
            )

    def forward(self, x):
        #print(x.shape)
        return self.ECA(x)


def window_partition(x, window_size):
    """
    Args:
        x: (b, h, w, c)
        window_size (int): window size

    Returns:
        windows: (num_windows*b, window_size, window_size, c)
    """
    b, h, w, c = x.shape
    x = x.view(b, h // window_size, window_size, w // window_size, window_size, c)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, c)
    return windows


def window_reverse(windows, window_size, h, w):
    """
    Args:
        windows: (num_windows*b, window_size, window_size, c)
        window_size (int): Window size
        h (int): Height of image
        w (int): Width of image

    Returns:
        x: (b, h, w, c)
    """
    b = int(windows.shape[0] / (h * w / window_size / window_size))
    x = windows.view(b, h // window_size, w // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, h, w, -1)
    return x


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)

        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, rpi, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*b, n, c)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        b_, n, c = x.shape
        qkv = self.qkv(x).reshape(b_, n, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[rpi.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nw = mask.shape[0]
            attn = attn.view(b_ // nw, nw, self.num_heads, n, n) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, n, n)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(b_, n, c)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self,embed_dim=96, norm_layer=None):
        super().__init__()
        self.embed_dim = embed_dim

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)  # b Ph*Pw c
        if self.norm is not None:
            x = self.norm(x)
        return x


class PatchUnEmbed(nn.Module):
    r""" Image to Patch Unembedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self,embed_dim=96):
        super().__init__()

        self.embed_dim = embed_dim

    def forward(self, x, x_size):
        x = x.transpose(1, 2).contiguous().view(x.shape[0], self.embed_dim, x_size[0], x_size[1])  # b Ph*Pw c
        return x

class CAFL(nn.Module):
    def __init__(self, dim):
        super(CAFL, self).__init__()
        self.fusion = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Linear(dim, dim)
        )
        self.gate = nn.Sigmoid()

    def forward(self, attn_x, conv_x):
        # attn_x, conv_x: (B, N, C)
        fusion_input = torch.cat([attn_x, conv_x], dim=-1)
        fused = self.fusion(fusion_input)
        gate = self.gate(fused)
        return attn_x * gate + conv_x * (1 - gate)

class MLP(nn.Module):
    def __init__(self, dim, mlp_ratio=4., drop=0.):
        super(MLP, self).__init__()
        hidden_dim = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class MAT(nn.Module):
    r""" Hybrid Attention Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self,
                 dim=124,
                 input_resolution=(64,64),
                 num_heads=4,
                 window_size=16,
                 shift_size=16 // 2,
                 compress_ratio=3,
                 squeeze_factor=30,
                 conv_scale=0.01,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 patch_norm=True,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, 'shift_size must in 0-window_size'

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim,
            window_size=to_2tuple(self.window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop)

        self.conv_scale = conv_scale
        self.conv_block = ECA(num_feat=dim, compress_ratio=compress_ratio, squeeze_factor=squeeze_factor)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
    
        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(embed_dim=124)
        
        ########## Cross Attention Fusion layer ##########
        self.cafl = CAFL(dim)
        self.mlp = MLP(dim, mlp_ratio=mlp_ratio, drop=drop)
        self.norm2 = norm_layer(dim)
        ##################################################

        # merge non-overlapping patches into image
        self.patch_unembed = PatchUnEmbed(
            embed_dim=124)

        self.pos_drop = nn.Dropout(p=0.0)

    def calculate_mask(self, x_size):
              # calculate attention mask for SW-MSA
              h, w = x_size
              img_mask = torch.zeros((1, h, w, 1))  # 1 h w 1
              h_slices = (slice(0, -self.window_size), slice(-self.window_size,
                                                            -self.shift_size), slice(-self.shift_size, None))
              w_slices = (slice(0, -self.window_size), slice(-self.window_size,
                                                            -self.shift_size), slice(-self.shift_size, None))
              cnt = 0
              for h in h_slices:
                  for w in w_slices:
                      img_mask[:, h, w, :] = cnt
                      cnt += 1

              mask_windows = window_partition(img_mask, self.window_size)  # nw, window_size, window_size, 1
              mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
              attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
              attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

              return attn_mask
    
    def calculate_rpi_sa(self):
        # calculate relative position index for SA
        coords_h = torch.arange(self.window_size)
        coords_w = torch.arange(self.window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size - 1
        relative_coords[:, :, 0] *= 2 * self.window_size - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        return relative_position_index

    def forward(self, x):
        x_size = (x.shape[2], x.shape[3])
        attn_mask = self.calculate_mask(x_size).to(x.device)
        h, w = x_size
        x = self.patch_embed(x)
        x = self.pos_drop(x)
        b,_,c = x.shape

        # assert seq_len == h * w, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(b, h, w, c)

        # Conv_X
        #print(x.shape)
        conv_x = self.conv_block(x.permute(0, 3, 1, 2))
        conv_x = conv_x.permute(0, 2, 3, 1).contiguous().view(b, h * w, c)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            attn_mask = attn_mask
        else:
            shifted_x = x
            attn_mask = None
        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nw*b, window_size, window_size, c
        x_windows = x_windows.view(-1, self.window_size * self.window_size, c)  # nw*b, window_size*window_size, c

        # W-MSA/SW-MSA (to be compatible for testing on images whose shapes are the multiple of window size
        rpi_sa= self.calculate_rpi_sa()
        attn_windows = self.attn(x_windows, rpi=rpi_sa, mask=attn_mask)

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, c)
        shifted_x = window_reverse(attn_windows, self.window_size, h, w)  # b h' w' c

        # reverse cyclic shift
        if self.shift_size > 0:
            attn_x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            attn_x = shifted_x
        attn_x = attn_x.view(b, h * w, c)

        # Cross-Attention Fusion Layer
        fused_x = self.cafl(attn_x, conv_x)

        # Residual Connection with DropPath
        x = shortcut + self.drop_path(fused_x)

        # MLP + Second Norm + Residual
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        x=self.patch_unembed(x, x_size)
        
        return x


class HOTNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=31, ratio=4):
        super(HOTNet, self).__init__()
        self.body1 = InSSSRBlock(in_channels, out_channels, ratio)
        self.body2 = MidSSSRBlock(in_channels, out_channels, ratio)
        self.body3 = OutSSSRBlock(in_channels, out_channels, ratio)
        self.body4 = MAT()
        self.res_x = nn.Sequential(
            Upsample(in_channels, ratio),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
        )

        self.convx = nn.Sequential(
            nn.Conv2d(in_channels=4 * out_channels, out_channels= out_channels, kernel_size=3, stride=1, padding=1, bias=True)
        )

        self.convxx = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=True)
    

    def forward(self, w):
        listX = []
        listY = []
        listZ = []
        res_x = self.res_x(w)
        x1, y1, z1, u, v = self.body1(w)
        listX.append(x1)
        listY.append(y1)
        listZ.append(z1)
        x2, y2, z2, u, v = self.body2(w, x1, u, v)
        listX.append(x2)
        listY.append(y2)
        listZ.append(z2)
        x3, y3, z3, u, v = self.body2(w, x2, u, v)
        listX.append(x3)
        listY.append(y3)
        listZ.append(z3)
        x4, y4, z4 = self.body3(w, x3, u, v)
        listX.append(x4)
        listY.append(y4)
        listZ.append(z4)

        xx = torch.cat([x1, x2, x3, x4], dim=1)
        xx1 = self.body4(xx)+xx
        xx2 = self.body4(xx1)+xx1
        xx3 = self.convx(xx2)
        x = xx3 + res_x
        x = self.convxx(x)

        return x, y4, z4, listX, listY, listZ


if __name__ == '__main__':
    x = torch.randn(2, 3, 6, 6)
    model = HOTNet(3, 6, 2)
    x, listx = model(x)
    print(x.size())
