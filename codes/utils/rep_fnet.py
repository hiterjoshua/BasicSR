import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .block_utils import reparameter_13, reparameter_31
from ptflops import get_model_complexity_info

class RBFNet(nn.Module):
    def __init__(self,  inp_planes, out_planes, depth_multiplier=1, deploy_flag = False):
        super(RBFNet, self).__init__()
        self.depth_multiplier = depth_multiplier
        self.inp_planes = inp_planes
        self.out_planes = out_planes
        self.mid_planes = int(inp_planes * depth_multiplier)
        self.deploy = deploy_flag
        
        if self.deploy:
            self.rbr_reparam = nn.Conv2d(self.inp_planes, self.out_planes, kernel_size=3, padding=1, bias=True)
        else:
            self.padding = nn.ZeroPad2d(1)
            self.conv3x3 = torch.nn.Conv2d(self.inp_planes, self.mid_planes, kernel_size=3, padding=0)
            self.conv1x1 = torch.nn.Conv2d(self.mid_planes, self.out_planes, kernel_size=1, padding=0)

    def forward(self, x):
        if hasattr(self, 'rbr_reparam'):
            y = self.rbr_reparam(x)
        else:
            y = self.conv1x1(self.conv3x3(self.padding(x)))
        return y

    def rep_params(self):
        # re-param conv kernel
        device = self.conv3x3.weight.get_device()
        if device < 0:
            device = None
        fused = reparameter_31(self.conv3x3, self.conv1x1)
        return fused.weight.to(device), fused.bias.to(device)

    def switch_to_deploy(self):
        if hasattr(self, 'rbr_reparam'):
            return
        self.rbr_reparam = torch.nn.Conv2d(self.inp_planes, self.out_planes, kernel_size=3, padding=1)
        RK, RB = self.rep_params()
        self.rbr_reparam.weight.data = RK
        self.rbr_reparam.bias.data = RB
        for para in self.parameters():
            para.detach_()
        self.__delattr__('conv3x3')
        self.__delattr__('conv1x1')


class REP_FNet(nn.Module):
    """ Optical flow estimation network
    """

    def __init__(self, in_nc):
        super(REP_FNet, self).__init__()

        self.encoder = nn.Sequential(RBFNet(2*in_nc, 16, 8), RBFNet(16, 32, 2), RBFNet(32, 64, 2))
        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.pooling = nn.MaxPool2d(2, 2)
        self.decoder = nn.Sequential(RBFNet(64, 128, 2), RBFNet(128, 64, 0.5), RBFNet(64, 32, 0.5))
        self.flowRB = RBFNet(32, 2, 0.5)
    
    def forward(self, x1, x2):
        """ Compute optical flow from x1 to x2
        """
        out = torch.cat([x1, x2], dim=1)
        for block in self.encoder:
            out = block(out)
            out = self.act(out)
            out = self.pooling(out)
        for block in self.decoder:
            out = block(out)
            out = F.interpolate(out, scale_factor=2, mode='bilinear', align_corners=False)
        out = torch.tanh(self.flowRB(out)) * 24  # 24 is the max velocity
        return out

if __name__ == '__main__':

    # # test seq-conv
    x = torch.randn(1, 2, 128, 128)* 200
    conv = RBFNet(2, 16, 8)
    y0 = conv(x)
    RK, RB = conv.rep_params()
    y1 = F.conv2d(input=x, weight=RK, bias=RB, stride=1, padding=1)
    print('RBFNet: ', torch.mean(torch.abs(y0-y1)))


# class REP_FNet(nn.Module):
#     """ Optical flow estimation network
#     """

#     def __init__(self, in_nc):
#         super(REP_FNet, self).__init__()

#         self.encoder1 = nn.Sequential(
#             nn.Conv2d(2*in_nc, 16, 3, 1, 1, bias=True),
#             nn.Conv2d(16, 16, 1, 1, 1, bias=True),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.MaxPool2d(2, 2))
#             #nn.AvgPool2d(2,2))

#         self.encoder2 = nn.Sequential(
#             nn.Conv2d(16, 32, 3, 1, 1, bias=True),
#             nn.Conv2d(32, 32, 1, 1, 1, bias=True),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.MaxPool2d(2, 2))
#             #nn.AvgPool2d(2,2))

#         self.encoder3 = nn.Sequential(
#             nn.Conv2d(32, 64, 3, 1, 1, bias=True),
#             nn.Conv2d(64, 64, 1, 1, 1, bias=True),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.MaxPool2d(2, 2))
#             #nn.AvgPool2d(2,2))

#         self.decoder1 = nn.Sequential(
#             nn.Conv2d(64, 128, 3, 1, 1, bias=True),
#             nn.Conv2d(128, 128, 1, 1, 1, bias=True),
#             nn.LeakyReLU(0.2, inplace=True))

#         self.decoder2 = nn.Sequential(
#             nn.Conv2d(128, 64, 3, 1, 1, bias=True),
#             nn.Conv2d(64, 64, 1, 1, 1, bias=True),
#             nn.LeakyReLU(0.2, inplace=True))

#         self.decoder3 = nn.Sequential(
#             nn.Conv2d(64, 32, 3, 1, 1, bias=True),
#             nn.Conv2d(32, 32, 1, 1, 1, bias=True),
#             nn.LeakyReLU(0.2, inplace=True))

#         self.flow = nn.Sequential(
#             nn.Conv2d(32, 16, 3, 1, 1, bias=True),
#             nn.Conv2d(16, 2, 1, 1, 1, bias=True))
    
    
#     def forward(self, x1, x2):
#         """ Compute optical flow from x1 to x2
#         """

#         out = self.encoder1(torch.cat([x1, x2], dim=1))
#         out = self.encoder2(out)
#         out = self.encoder3(out)
#         out = F.interpolate(
#             self.decoder1(out), scale_factor=2, mode='bilinear', align_corners=False)
#         out = F.interpolate(
#             self.decoder2(out), scale_factor=2, mode='bilinear', align_corners=False)
#         out = F.interpolate(
#             self.decoder3(out), scale_factor=2, mode='bilinear', align_corners=False)
#         out = torch.tanh(self.flow(out)) * 24  # 24 is the max velocity

#         return out