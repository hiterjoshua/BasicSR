import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_nets import BaseSequenceGenerator, BaseSequenceDiscriminator
from utils.net_utils import space_to_depth, backward_warp, get_upsampling_func
from utils.net_utils import initialize_weights
from utils.data_utils import float32_to_uint8
from utils.block_utils import RepVSRRB, RepVSRRB_nores
from utils.rep_fnet import REP_FNet, FNet

import flow_vis
from ptflops import get_model_complexity_info

class ResidualBlock(nn.Module):
    """ Residual block without batch normalization
    """

    def __init__(self, nf=64):
        super(ResidualBlock, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(nf, nf, 3, 1, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf, nf, 3, 1, 1, bias=True))

    def forward(self, x):
        out = self.conv(x) + x

        return out


class SRNet(nn.Module):
    """ Reconstruction & Upsampling network
    """

    def __init__(self, in_nc=3, out_nc=3, nf=64, nb=6, upsample_func=None,
                 scale=4):
        super(SRNet, self).__init__()

        self.in_nc = in_nc
        if self.in_nc == 3:
            nf = 16
        # input conv.
        self.conv_in = nn.Sequential(
            nn.Conv2d((scale**2 + 1) * in_nc, nf, 3, 1, 1, bias=True),
            nn.ReLU(inplace=True))

        # residual blocks
        self.resblocks = nn.Sequential(*[RepVSRRB(nf, nf, 4) for _ in range(nb)])
        self.act = nn.ReLU(inplace=True)

        # upsampling function
        self.conv_up_pixelshuffle = nn.Sequential(
            nn.PixelShuffle(4),
            nn.ReLU(inplace=True))

        # output conv.
        self.conv_out = nn.Conv2d(16, 48, 3, 1, 1, bias=True)

    def forward(self, lr_curr, hr_prev_tran):
        """ lr_curr: the current lr data in shape nchw
            hr_prev_tran: the previous transformed hr_data in shape n(4*4*c)hw
        """
        out = self.conv_in(torch.cat([lr_curr, hr_prev_tran], dim=1))
        for block in self.resblocks:
            out = block(out)
            out = self.act(out)
        if self.in_nc == 3:
            out = self.conv_out(out)
            out = self.act(out)
        out = self.conv_up_pixelshuffle(out)
        return out

class SRNet_120648(nn.Module):
    """ Reconstruction & Upsampling network
    """

    def __init__(self, in_nc=3, out_nc=3, nf=64, nb=6, upsample_func=None,
                 scale=4):
        super(SRNet_120648, self).__init__()

        self.in_nc = in_nc
        if self.in_nc == 3:
            nf = 48
        # input conv.
        self.conv_in = nn.Sequential(
            nn.Conv2d((scale**2 + 1) * in_nc, nf, 3, 1, 1, bias=True),
            nn.ReLU(inplace=True))

        # residual blocks
        self.resblocks = nn.Sequential(*[RepVSRRB(nf, nf, 4) for _ in range(nb)])
        self.act = nn.ReLU(inplace=True)

        # upsampling function
        self.conv_up_pixelshuffle = nn.Sequential(
            nn.PixelShuffle(4),
            nn.ReLU(inplace=True))


    def forward(self, lr_curr, hr_prev_tran):
        """ lr_curr: the current lr data in shape nchw
            hr_prev_tran: the previous transformed hr_data in shape n(4*4*c)hw
        """
        out = self.conv_in(torch.cat([lr_curr, hr_prev_tran], dim=1))
        for block in self.resblocks:
            out = block(out)
            out = self.act(out)
        out = self.conv_up_pixelshuffle(out)
        return out


class SRNet_1206cascade(nn.Module):
    """ Reconstruction & Upsampling network
    """

    def __init__(self, in_nc=3, out_nc=3, nf=64, nb=6, upsample_func=None,
                 scale=4):
        super(SRNet_1206cascade, self).__init__()

        self.in_nc = in_nc
        if self.in_nc == 3:
            nf = 16
        # input conv.
        self.conv_in = nn.Sequential(
            nn.Conv2d((scale**2 + 1) * in_nc, nf, 3, 1, 1, bias=True),
            nn.ReLU(inplace=True))

        # residual blocks
        self.resblocks = nn.Sequential(*[RepVSRRB(nf, nf, 4) for _ in range(nb)] \
                            ,RepVSRRB_nores(nf, 32, 4),RepVSRRB_nores(32, 48, 4))
        self.act = nn.ReLU(inplace=True)

        # upsampling function
        self.conv_up_pixelshuffle = nn.Sequential(
            nn.PixelShuffle(4),
            nn.ReLU(inplace=True))
        
    def forward(self, lr_curr, hr_prev_tran):
        """ lr_curr: the current lr data in shape nchw
            hr_prev_tran: the previous transformed hr_data in shape n(4*4*c)hw
        """
        out = self.conv_in(torch.cat([lr_curr, hr_prev_tran], dim=1))
        for block in self.resblocks:
            out = block(out)
            out = self.act(out)
        out = self.conv_up_pixelshuffle(out)
        return out

class SRNet_1206trans(nn.Module):
    """ Reconstruction & Upsampling network
    """

    def __init__(self, in_nc=3, out_nc=3, nf=64, nb=6, upsample_func=None,
                 scale=4):
        super(SRNet_1206trans, self).__init__()

        self.in_nc = in_nc
        if self.in_nc == 3:
            nf = 16
        # input conv.
        self.conv_in = nn.Sequential(
            nn.Conv2d((scale**2 + 1) * in_nc, nf, 3, 1, 1, bias=True),
            nn.ReLU(inplace=True))

        # residual blocks
        self.resblocks = nn.Sequential(*[RepVSRRB(nf, nf, 4) for _ in range(nb)])
        self.act = nn.ReLU(inplace=True)

        self.conv_up = nn.Sequential(
            nn.ConvTranspose2d(nf, nf, 3, 2, 1, output_padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(nf, nf, 3, 2, 1, output_padding=1, bias=True),
            nn.ReLU(inplace=True))

        # output conv.
        self.conv_out = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)


    def forward(self, lr_curr, hr_prev_tran):
        """ lr_curr: the current lr data in shape nchw
            hr_prev_tran: the previous transformed hr_data in shape n(4*4*c)hw
        """
        out = self.conv_in(torch.cat([lr_curr, hr_prev_tran], dim=1))
        for block in self.resblocks:
            out = block(out)
            out = self.act(out)
        out = self.conv_up(out)
        out = self.conv_out(out)
        return out


class FRNet(BaseSequenceGenerator):
    """ Frame-recurrent network proposed in https://arxiv.org/abs/1801.04590
    """

    def __init__(self, in_nc=3, out_nc=3, nf=64, nb=16, degradation='BI',
                 scale=4):
        super(FRNet, self).__init__()

        self.scale = scale

        # get upsampling function according to the degradation mode
        self.upsample_func = get_upsampling_func(self.scale, degradation)

        # define fnet & srnet
        # self.fnet = FNet(in_nc)
        self.fnet = REP_FNet(in_nc)
        # self.srnet = SRNet_120648(in_nc, out_nc, nf, nb, self.upsample_func)
        # self.srnet = SRNet_1206cascade(in_nc, out_nc, nf, nb, self.upsample_func)
        self.srnet = SRNet_1206trans(in_nc, out_nc, nf, nb, self.upsample_func)
        # self.srnet = SRNet(in_nc, out_nc, nf, nb, self.upsample_func)
        self.print_network_fnet(self.fnet)
        self.print_network_srnet(self.srnet)

    def generate_dummy_input(self, lr_size):
        c, lr_h, lr_w = lr_size
        s = self.scale

        lr_curr = torch.rand(1, c, lr_h, lr_w, dtype=torch.float32)
        lr_prev = torch.rand(1, c, lr_h, lr_w, dtype=torch.float32)
        hr_prev = torch.rand(1, c, s * lr_h, s * lr_w, dtype=torch.float32)

        data_dict = {
            'lr_curr': lr_curr,
            'lr_prev': lr_prev,
            'hr_prev': hr_prev
        }

        return data_dict

    def forward(self, lr_curr, lr_prev, hr_prev):
        """
            Parameters:
                :param lr_curr: the current lr data in shape nchw
                :param lr_prev: the previous lr data in shape nchw
                :param hr_prev: the previous hr data in shape nc(4h)(4w)
        """

        # estimate lr flow (lr_curr -> lr_prev)
        lr_flow = self.fnet(lr_curr, lr_prev)

        # pad if size is not a multiple of 8
        pad_h = lr_curr.size(2) - lr_curr.size(2) // 8 * 8
        pad_w = lr_curr.size(3) - lr_curr.size(3) // 8 * 8
        lr_flow_pad = F.pad(lr_flow, (0, pad_w, 0, pad_h), 'reflect')

        # # change the encoder-decoder down/up scale from 8 to 4, so the 8 ought to change to 4 either
        # pad_h = lr_curr.size(2) - lr_curr.size(2) // 4 * 4
        # pad_w = lr_curr.size(3) - lr_curr.size(3) // 4 * 4
        # lr_flow_pad = F.pad(lr_flow, (0, pad_w, 0, pad_h), 'reflect')

        # upsample lr flow
        hr_flow = self.scale * self.upsample_func(lr_flow_pad)

        # warp hr_prev
        hr_prev_warp = backward_warp(hr_prev, hr_flow)

        # compute hr_curr
        hr_curr = self.srnet(lr_curr, space_to_depth(hr_prev_warp, self.scale))

        return hr_curr, hr_flow, hr_prev_warp

    def forward_sequence(self, lr_data):
        """
            Parameters:
                :param lr_data: lr data in shape ntchw
        """

        n, t, c, lr_h, lr_w = lr_data.size()
        hr_h, hr_w = lr_h * self.scale, lr_w * self.scale

        # calculate optical flows
        lr_prev = lr_data[:, :-1, ...].reshape(n * (t - 1), c, lr_h, lr_w)
        lr_curr = lr_data[:, 1:, ...].reshape(n * (t - 1), c, lr_h, lr_w)
        lr_flow = self.fnet(lr_curr, lr_prev)  # n*(t-1),2,h,w

        # upsample lr flows
        hr_flow = self.scale * self.upsample_func(lr_flow)
        hr_flow = hr_flow.view(n, (t - 1), 2, hr_h, hr_w)

        # compute the first hr data
        hr_data = []
        hr_prev = self.srnet(
            lr_data[:, 0, ...],
            torch.zeros(n, (self.scale**2)*c, lr_h, lr_w, dtype=torch.float32,
                        device=lr_data.device))
        hr_data.append(hr_prev)

        # compute the remaining hr data
        for i in range(1, t):
            # warp hr_prev
            hr_prev_warp = backward_warp(hr_prev, hr_flow[:, i - 1, ...])

            # compute hr_curr
            hr_curr = self.srnet(
                lr_data[:, i, ...],
                space_to_depth(hr_prev_warp, self.scale))

            # save and update
            hr_data.append(hr_curr)
            hr_prev = hr_curr

        hr_data = torch.stack(hr_data, dim=1)  # n,t,c,hr_h,hr_w

        # construct output dict
        ret_dict = {
            'hr_data': hr_data,  # n,t,c,hr_h,hr_w
            'hr_flow': hr_flow,  # n,t,2,hr_h,hr_w
            'lr_prev': lr_prev,  # n(t-1),c,lr_h,lr_w
            'lr_curr': lr_curr,  # n(t-1),c,lr_h,lr_w
            'lr_flow': lr_flow,  # n(t-1),2,lr_h,lr_w
        }

        return ret_dict

    def infer_sequence(self, lr_data, device):
        """
            Parameters:
                :param lr_data: torch.FloatTensor in shape tchw
                :param device: torch.device

                :return hr_seq: uint8 np.ndarray in shape tchw
        """

        # setup params
        tot_frm, c, h, w = lr_data.size()
        s = self.scale

        # forward
        hr_seq = []
        lr_prev = torch.zeros(1, c, h, w, dtype=torch.float32).to(device)
        hr_prev = torch.zeros(
            1, c, s * h, s * w, dtype=torch.float32).to(device)

        for i in range(tot_frm):
            with torch.no_grad():
                self.eval()

                lr_curr = lr_data[i: i + 1, ...].to(device)
                hr_curr, hr_flow, hr_warp = self.forward(lr_curr, lr_prev, hr_prev)
                lr_prev, hr_prev = lr_curr, hr_curr

                hr_warp = hr_warp.squeeze(0).cpu().numpy()  # chw|rgb|uint8
                hr_frm = hr_warp.transpose(1, 2, 0)  # hwc
                flow_frm = hr_flow.squeeze(0).cpu().numpy()  # chw|rgb|uint8
                flow_uv = flow_frm.transpose(1, 2, 0)  # hwc
                flow_color = flow_vis.flow_to_color(flow_uv, convert_to_bgr=False)

            hr_seq.append(float32_to_uint8(hr_frm))
            # hr_seq.append(float32_to_uint8(flow_color))

        return np.stack(hr_seq)

    def print_network_fnet(self, net):

        net_cls_str = f'{net.__class__.__name__}'
        net_str = str(net)
        net_params = sum(map(lambda x: x.numel(), net.parameters()))
        # a = map(lambda x: x.numel(), net.parameters())
        # for key in a:
        #     print(key)
        #print(net_str)
        print(f'FNet Network: {net_cls_str}, with parameters: {net_params:,d}')


    def print_network_srnet(self, net):

        net_cls_str = f'{net.__class__.__name__}'
        net_str = str(net)
        net_params = sum(map(lambda x: x.numel(), net.parameters()))

        #print(net_str)
        print(f'SRNet Network: {net_cls_str}, with parameters: {net_params:,d}')


