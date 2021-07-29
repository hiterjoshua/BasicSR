import torch
from torch import nn as nn

from basicsr.archs.arch_util import RBRepSR, RBRepSR_xt, Upsample, make_layer
from basicsr.utils.registry import ARCH_REGISTRY


@ARCH_REGISTRY.register()
class RepSR(nn.Module):
    """EDSR network structure.

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
                 num_feat=64,
                 inside_feat = 256, 
                 num_block=16,
                 upscale=4,
                 res_scale=1,
                 img_range=255.,
                 rgb_mean=(0.4488, 0.4371, 0.4040),
                 xt_flag=True):
        super(RepSR, self).__init__()

        self.img_range = img_range
        self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)

        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.relu = nn.ReLU(inplace=True)
        
        if xt_flag:
            self.body = make_layer(RBRepSR_xt, num_block, num_feat=num_feat, inside_feat=inside_feat, res_scale=res_scale, pytorch_init=True)
        else:
            self.body = make_layer(RBRepSR, num_block, num_feat=num_feat, res_scale=res_scale, pytorch_init=True)

        self.conv_after_body = nn.Conv2d(num_feat, upscale**2, 3, 1, 1)
        self.upsample = nn.PixelShuffle(4)


    def forward(self, x):
        x = self.relu(self.conv_first(x))
        
        res = self.body(x)
        res = res+x
        
        x = self.conv_after_body(res)
        x = self.upsample(x)
        return x
