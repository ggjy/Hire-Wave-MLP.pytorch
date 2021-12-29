import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch import Tensor
from functools import partial
from torch.nn.modules.utils import _pair
from torch.nn import init

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model

from mmseg.utils import get_root_logger
from ..builder import BACKBONES
from mmcv.runner import load_checkpoint


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class HireMLP(nn.Module):
    def __init__(self, dim, attn_drop=0., proj_drop=0., pixel=2, step=1,
                 step_pad_mode='c', pixel_pad_mode='c', norm_style=nn.BatchNorm2d):
        super().__init__()
        self.pixel = pixel
        self.step = step
        self.step_pad_mode = step_pad_mode
        self.pixel_pad_mode = pixel_pad_mode
        print('pixel: {} pad mode: {} step: {} pad mode: {}'.format(
              pixel, pixel_pad_mode, step, step_pad_mode))
        
        self.mlp_h1 = nn.Conv2d(dim*pixel, dim//2, 1, bias=False)
        self.mlp_h1_norm = norm_style(dim//2)
        self.mlp_h2 = nn.Conv2d(dim//2, dim*pixel, 1, bias=True)
        self.mlp_w1 = nn.Conv2d(dim*pixel, dim//2, 1, bias=False)
        self.mlp_w1_norm = norm_style(dim//2)
        self.mlp_w2 = nn.Conv2d(dim//2, dim*pixel, 1, bias=True)
        self.mlp_c = nn.Conv2d(dim, dim, 1, bias=True)
        
        self.act = nn.ReLU()

        self.reweight = Mlp(dim, dim // 4, dim * 3)

        self.proj = nn.Conv2d(dim, dim, 1)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        """
        h: H x W x C -> H/pixel x W x C*pixel
        w: H x W x C -> H x W/pixel x C*pixel
        Setting of F.pad: (left, right, top, bottom)
        """
        B, C, H, W = x.shape

        pad_h, pad_w = (self.pixel - H % self.pixel) % self.pixel, (self.pixel - W % self.pixel) % self.pixel
        h, w = x.clone(), x.clone()
        
        if self.step:
            if self.step_pad_mode == '0':
                h = F.pad(h, (0, 0, self.step, 0), "constant", 0)
                w = F.pad(w, (self.step, 0, 0, 0), "constant", 0)
            elif self.step_pad_mode == 'c':
                h = F.pad(h, (0, 0, self.step, 0), mode='circular')
                w = F.pad(w, (self.step, 0, 0, 0), mode='circular')
            else:
                raise NotImplementedError("Invalid pad mode.")
            h = torch.narrow(h, 2, 0, H)
            w = torch.narrow(w, 3, 0, W)
        
        if self.pixel_pad_mode == '0':
            h = F.pad(h, (0, 0, 0, pad_h), "constant", 0)
            w = F.pad(w, (0, pad_w, 0, 0), "constant", 0)
        elif self.pixel_pad_mode == 'c':
            h = F.pad(h, (0, 0, 0, pad_h), mode='circular')
            w = F.pad(w, (0, pad_w, 0, 0), mode='circular')
        elif self.pixel_pad_mode == 'replicate':
            h = F.pad(h, (0, 0, 0, pad_h), mode='replicate')
            w = F.pad(w, (0, pad_w, 0, 0), mode='replicate')
        else:
            raise NotImplementedError("Invalid pad mode.")
        
        h = h.reshape(B, C, (H + pad_h) // self.pixel, self.pixel, W).permute(0, 1, 3, 2, 4).reshape(B, C*self.pixel, (H + pad_h) // self.pixel, W)
        w = w.reshape(B, C, H, (W + pad_w) // self.pixel, self.pixel).permute(0, 1, 4, 2, 3).reshape(B, C*self.pixel, H, (W + pad_w) // self.pixel)
          
        h = self.mlp_h1(h)
        h = self.mlp_h1_norm(h)
        h = self.act(h)
        h = self.mlp_h2(h)
        
        w = self.mlp_w1(w)
        w = self.mlp_w1_norm(w)
        w = self.act(w)
        w = self.mlp_w2(w)
        
        h = h.reshape(B, C, self.pixel, (H + pad_h) // self.pixel, W).permute(0, 1, 3, 2, 4).reshape(B, C, H + pad_h, W)
        w = w.reshape(B, C, self.pixel, H, (W + pad_w) // self.pixel).permute(0, 1, 3, 4, 2).reshape(B, C, H, W + pad_w)
        
        h = torch.narrow(h, 2, 0, H)
        w = torch.narrow(w, 3, 0, W)
        
        # back to original order
        if self.step and self.step_pad_mode == 'c':
            h = torch.roll(h, -self.step, -2)
            w = torch.roll(w, -self.step, -1)
            
        c = self.mlp_c(x)

        a = (h + w + c).flatten(2).mean(2).unsqueeze(2).unsqueeze(2)
        a = self.reweight(a).reshape(B, C, 3).permute(2, 0, 1).softmax(dim=0).unsqueeze(3).unsqueeze(3)

        x = h * a[0] + w * a[1] + c * a[2]
        
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class HireBlock(nn.Module):
    def __init__(self, dim, mlp_ratio=4., drop=0., attn_drop=0., drop_path=0.,
                 pixel=2, step=1, step_pad_mode='c', pixel_pad_mode='c',
                 norm_style=nn.BatchNorm2d):
        super().__init__()
        self.norm1 = norm_style(dim)
        self.attn = HireMLP(
            dim, attn_drop=attn_drop, pixel=pixel, step=step,
            step_pad_mode=step_pad_mode, pixel_pad_mode=pixel_pad_mode,
            norm_style=norm_style)

        self.norm2 = norm_style(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbedOverlapping(nn.Module):
    """ 2D Image to Patch Embedding with overlapping
    """
    def __init__(self, patch_size=16, stride=16, padding=0, in_chans=3, embed_dim=768, groups=1, norm_style=nn.BatchNorm2d):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        stride = to_2tuple(stride)
        padding = to_2tuple(padding)
        self.patch_size = patch_size

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride, padding=padding, groups=groups)
        self.norm = norm_style(embed_dim)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class Downsample(nn.Module):
    """ Downsample stage
    """
    def __init__(self, in_embed_dim, out_embed_dim, patch_size, norm_style=nn.BatchNorm2d):
        super().__init__()
        assert patch_size == 2, patch_size
        self.proj = nn.Conv2d(in_embed_dim, out_embed_dim, kernel_size=(3, 3), stride=(2, 2), padding=1)
        self.norm = norm_style(out_embed_dim)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)
        x = self.act(x)
        return x


def basic_blocks(dim, index, layers, mlp_ratio=4., attn_drop=0., drop_path_rate=0.,
                 pixel=2, step_stride=2, step_dilation=1,
                 step_pad_mode='c', pixel_pad_mode='c',
                 norm_style=nn.BatchNorm2d, **kwargs):
    blocks = []
    for block_idx in range(layers[index]):
        block_dpr = drop_path_rate * (block_idx + sum(layers[:index])) / (sum(layers) - 1)
        blocks.append(
            HireBlock(
                dim, mlp_ratio=mlp_ratio, attn_drop=attn_drop, drop_path=block_dpr, pixel=pixel,
                step=(block_idx % step_stride) * step_dilation,
                step_pad_mode=step_pad_mode, pixel_pad_mode=pixel_pad_mode,
                norm_style=norm_style))
    blocks = nn.Sequential(*blocks)

    return blocks


class HireMLPNet(nn.Module):
    def __init__(self, layers=[2,2,4,2], img_size=224, patch_size=4, in_chans=3,
        embed_dims=None, mlp_ratios=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
        norm_layer=nn.BatchNorm2d, pixel=[2,2,2,2], out_indices=[0,2,4,6],
        step_stride=[2,2,2,2], step_dilation=[1,1,1,1],
        step_pad_mode='c', pixel_pad_mode='c', load_from_ema=False,
        norm_style=nn.BatchNorm2d, extra_norm=True):
        super().__init__()
        print('drop path rate: {}'.format(drop_path_rate))
        self.load_from_ema = load_from_ema
        self.extra_norm = extra_norm
        self.patch_embed = PatchEmbedOverlapping(
            patch_size=7, stride=4, padding=2, in_chans=3, embed_dim=embed_dims[0], norm_style=norm_style)

        network = []
        for i in range(len(layers)):
            stage = basic_blocks(
                embed_dims[i], i, layers, mlp_ratio=mlp_ratios[i],
                attn_drop=attn_drop_rate, drop_path_rate=drop_path_rate, pixel=pixel[i],
                step_stride=step_stride[i], step_dilation=step_dilation[i],
                step_pad_mode=step_pad_mode, pixel_pad_mode=pixel_pad_mode,
                norm_style=norm_style)
            network.append(stage)
            if i >= len(layers) - 1:
                break
            network.append(Downsample(embed_dims[i], embed_dims[i+1], 2, norm_style=norm_style))

        self.network = nn.ModuleList(network)
        self.out_indices = out_indices
        
        if self.extra_norm:
            for idx, num in enumerate(self.out_indices):
                layer = norm_style(embed_dims[idx])
                layer_name = f'norm{num}'
                self.add_module(layer_name, layer)
                
        self.apply(self.cls_init_weights)

    def cls_init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.SyncBatchNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            state_dict = torch.load(pretrained, map_location='cpu')
            if self.load_from_ema:
                self.load_state_dict(state_dict['model_ema'], strict=False)
                print('Successfully load backbone ema ckpt.')
            else:
                self.load_state_dict(state_dict['model'], strict=False)
                print('Successfully load backbone ckpt.')

    def forward_embeddings(self, x):
        x = self.patch_embed(x)
        return x

    def forward_tokens(self, x):
        outs = []
        for idx, block in enumerate(self.network):
            x = block(x)
            # added by jianyuan.guo
            if idx in self.out_indices:
                if self.extra_norm:
                    norm_layer = getattr(self, f'norm{idx}')
                    outs.append(norm_layer(x))
                else:
                    outs.append(x)
        return outs

    def forward(self, x):
        x = self.forward_embeddings(x)
        x = self.forward_tokens(x)

        return x


@BACKBONES.register_module()
class seg_hire_mlp_tiny(HireMLPNet):
    def __init__(self, **kwargs):
        pixel = kwargs.get('pixel', [4, 3, 3, 2])
        step_stride = kwargs.get('step_stride', [2, 2, 3, 2])
        step_dilation = kwargs.get('step_dilation', [2, 2, 1, 1])
        step_pad_mode = kwargs.get('step_pad_mode', 'c')
        pixel_pad_mode = kwargs.get('pixel_pad_mode', 'c')
        drop_path_rate = kwargs.get('drop_path_rate', 0.1)
        load_from_ema = kwargs.get('load_from_ema', False)
        extra_norm = kwargs.get('extra_norm', True)
        norm = kwargs.get('norm_style', 'bn')
        norm_style = nn.SyncBatchNorm if norm == 'syncbn' else nn.BatchNorm2d
        super(seg_hire_mlp_tiny, self).__init__(
            layers = [2, 2, 4, 2],
            embed_dims = [64, 128, 320, 512],
            patch_size = 7,
            mlp_ratios = [4, 4, 4, 4],
            pixel = pixel,
            step_stride = step_stride,
            step_dilation = step_dilation,
            step_pad_mode = step_pad_mode,
            pixel_pad_mode = pixel_pad_mode,
            drop_path_rate = drop_path_rate,
            load_from_ema = load_from_ema,
            norm_style = norm_style,
            extra_norm = extra_norm)


@BACKBONES.register_module()
class seg_hire_mlp_small(HireMLPNet):
    def __init__(self, **kwargs):
        pixel = kwargs.get('pixel', [4, 3, 3, 2])
        step_stride = kwargs.get('step_stride', [2, 2, 3, 2])
        step_dilation = kwargs.get('step_dilation', [2, 2, 1, 1])
        step_pad_mode = kwargs.get('step_pad_mode', 'c')
        pixel_pad_mode = kwargs.get('pixel_pad_mode', 'c')
        drop_path_rate = kwargs.get('drop_path_rate', 0.1)
        load_from_ema = kwargs.get('load_from_ema', False)
        extra_norm = kwargs.get('extra_norm', True)
        norm = kwargs.get('norm_style', 'bn')
        norm_style = nn.SyncBatchNorm if norm == 'syncbn' else nn.BatchNorm2d
        super(seg_hire_mlp_small, self).__init__(
            layers = [3, 4, 10, 3],
            embed_dims = [64, 128, 320, 512],
            patch_size = 7,
            mlp_ratios = [4, 4, 4, 4],
            pixel = pixel,
            step_stride = step_stride,
            step_dilation = step_dilation,
            step_pad_mode = step_pad_mode,
            pixel_pad_mode = pixel_pad_mode,
            drop_path_rate = drop_path_rate,
            load_from_ema = load_from_ema,
            norm_style = norm_style,
            extra_norm = extra_norm)


@BACKBONES.register_module()
class seg_hire_mlp_base(HireMLPNet):
    def __init__(self, **kwargs):
        pixel = kwargs.get('pixel', [4, 3, 3, 2])
        step_stride = kwargs.get('step_stride', [2, 2, 3, 2])
        step_dilation = kwargs.get('step_dilation', [2, 2, 1, 1])
        step_pad_mode = kwargs.get('step_pad_mode', 'c')
        pixel_pad_mode = kwargs.get('pixel_pad_mode', 'c')
        drop_path_rate = kwargs.get('drop_path_rate', 0.2)
        load_from_ema = kwargs.get('load_from_ema', False)
        extra_norm = kwargs.get('extra_norm', True)
        norm = kwargs.get('norm_style', 'bn')
        norm_style = nn.SyncBatchNorm if norm == 'syncbn' else nn.BatchNorm2d
        super(seg_hire_mlp_base, self).__init__(
            layers = [4, 6, 24, 3],
            embed_dims = [64, 128, 320, 512],
            patch_size = 7,
            mlp_ratios = [4, 4, 4, 4],
            pixel = pixel,
            step_stride = step_stride,
            step_dilation = step_dilation,
            step_pad_mode = step_pad_mode,
            pixel_pad_mode = pixel_pad_mode,
            drop_path_rate = drop_path_rate,
            load_from_ema = load_from_ema,
            norm_style = norm_style,
            extra_norm = extra_norm)


@BACKBONES.register_module()
class seg_hire_mlp_large(HireMLPNet):
    def __init__(self, **kwargs):
        pixel = kwargs.get('pixel', [4, 3, 3, 2])
        step_stride = kwargs.get('step_stride', [2, 2, 3, 2])
        step_dilation = kwargs.get('step_dilation', [2, 2, 1, 1])
        step_pad_mode = kwargs.get('step_pad_mode', 'c')
        pixel_pad_mode = kwargs.get('pixel_pad_mode', 'c')
        drop_path_rate = kwargs.get('drop_path_rate', 0.3)
        load_from_ema = kwargs.get('load_from_ema', False)
        extra_norm = kwargs.get('extra_norm', True)
        norm = kwargs.get('norm_style', 'bn')
        norm_style = nn.SyncBatchNorm if norm == 'syncbn' else nn.BatchNorm2d
        super(seg_hire_mlp_large, self).__init__(
            layers = [4, 6, 24, 3],
            embed_dims = [96, 192, 384, 768],
            patch_size = 7,
            mlp_ratios = [4, 4, 4, 4],
            pixel = pixel,
            step_stride = step_stride,
            step_dilation = step_dilation,
            step_pad_mode = step_pad_mode,
            pixel_pad_mode = pixel_pad_mode,
            drop_path_rate = drop_path_rate,
            load_from_ema = load_from_ema,
            norm_style = norm_style,
            extra_norm = extra_norm)
