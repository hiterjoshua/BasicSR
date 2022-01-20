import torch
from torch import nn as nn

from basicsr.archs.arch_util import Upsample, make_layer
from basicsr.archs.block_utils import RBX2SR
from basicsr.utils.registry import ARCH_REGISTRY


@ARCH_REGISTRY.register()
class X2SR(nn.Module):
    """X2SR network structure.

    Args:
        num_in_ch (int): Channel number of inputs.
        num_out_ch (int): Channel number of outputs.
        num_feat (int): Channel number of intermediate features.
            Default: 64.
        num_block (int): Block number in the trunk network. Default: 16.
        upscale (int): Upsampling factor. Support 2^n and 3.
            Default: 4.
        res_scale (float): Used to scale the residual in residual block.
            Default: 1.
        img_range (float): Image range. Default: 255.
        rgb_mean (tuple[float]): Image mean in RGB orders.
            Default: (0.4488, 0.4371, 0.4040), calculated from DIV2K dataset.
    """

    def __init__(self,
                 num_in_ch,
                 num_out_ch,
                 num_feat=16,
                 inside_feat = 64, 
                 num_block=4,
                 upscale=2,
                 res_scale=1,
                 img_range=255.,
                 rgb_mean=(0.4488, 0.4371, 0.4040),
                 deconv_flag=False,
                 deploy_flag=False):
        super(X2SR, self).__init__()

        self.img_range = img_range
        self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)

        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 1, 1)
        self.relu = nn.ReLU(inplace=True)
        
        self.body = make_layer(RBX2SR, num_block, inp_planes=num_feat, out_planes=num_feat, \
                                depth_multiplier=2, deploy_flag = False)

        self.conv_after_body = nn.Conv2d(num_feat, upscale**2, 3, 1, 1)
        if deconv_flag:
            self.upsample = nn.ConvTranspose2d(4, 1, 3, 2, 1, output_padding=1, bias=True)
        else:
            self.upsample = nn.PixelShuffle(upscale)

    def forward(self, x):
        x = self.relu(self.conv_first(x))
        
        res = self.body(x)
        res = res+x
        
        x = self.conv_after_body(res)
        x = self.upsample(x)
        return x

# class X2SR(nn.Module):
#     """X2SR network structure.

#     Args:
#         num_in_ch (int): Channel number of inputs.
#         num_out_ch (int): Channel number of outputs.
#         num_feat (int): Channel number of intermediate features.
#             Default: 64.
#         num_block (int): Block number in the trunk network. Default: 16.
#         upscale (int): Upsampling factor. Support 2^n and 3.
#             Default: 4.
#         res_scale (float): Used to scale the residual in residual block.
#             Default: 1.
#         img_range (float): Image range. Default: 255.
#         rgb_mean (tuple[float]): Image mean in RGB orders.
#             Default: (0.4488, 0.4371, 0.4040), calculated from DIV2K dataset.
#     """

#     def __init__(self,
#                  num_in_ch,
#                  num_out_ch,
#                  num_feat=16,
#                  inside_feat = 64, 
#                  num_block=4,
#                  upscale=2,
#                  res_scale=1,
#                  img_range=255.,
#                  rgb_mean=(0.4488, 0.4371, 0.4040),
#                  deconv_flag=False,
#                  deploy_flag=False):
#         super(X2SR, self).__init__()

#         self.img_range = img_range
#         self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)

#         self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 2, 1)
#         self.relu = nn.ReLU(inplace=True)
        
#         self.body = make_layer(RBX2SR, num_block, inp_planes=num_feat, out_planes=num_feat, \
#                                 depth_multiplier=2, deploy_flag = False)

#         self.conv_after_body = nn.Conv2d(num_feat, 16, 3, 1, 1)
#         if deconv_flag:
#             self.upsample = nn.ConvTranspose2d(4, 1, 3, 2, 1, output_padding=1, bias=True)
#         else:
#             self.upsample = nn.PixelShuffle(4)

#     def forward(self, x):
#         x = self.relu(self.conv_first(x))
        
#         res = self.body(x)
#         res = res+x
        
#         x = self.conv_after_body(res)
#         x = self.upsample(x)
#         return x

