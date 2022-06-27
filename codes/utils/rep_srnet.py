import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.block_utils import RepVSRRB, RepVSRRB_nores


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

class SRNet_fnet1124three(nn.Module):
    """ Reconstruction & Upsampling network
    """

    def __init__(self, in_nc=3, out_nc=3, nf=64, nb=6, upsample_func=None,
                 scale=4):
        super(SRNet_fnet1124three, self).__init__()

        self.in_nc = in_nc
        if self.in_nc == 3:
            nf = 64
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
        self.conv_out = nn.Conv2d(4, out_nc, 3, 1, 1, bias=True)

    def forward(self, lr_curr, hr_prev_tran):
        """ lr_curr: the current lr data in shape nchw
            hr_prev_tran: the previous transformed hr_data in shape n(4*4*c)hw
        """
        out = self.conv_in(torch.cat([lr_curr, hr_prev_tran], dim=1))
        for block in self.resblocks:
            out = block(out)
            out = self.act(out)
        out = self.conv_up_pixelshuffle(out)
        if self.in_nc == 3:
            out = self.conv_out(out)

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


class SRNet1210_48(nn.Module):
    """ Reconstruction & Upsampling network
    """

    def __init__(self, in_nc=3, out_nc=3, nf=64, nb=6, upsample_func=None,
                 scale=4):
        super(SRNet1210_48, self).__init__()

        self.in_nc = in_nc
        if self.in_nc == 3:
            nf = 48
        # input conv.
        self.conv_in = nn.Sequential(
            nn.Conv2d((scale**2 + 1) * in_nc, nf, 3, 1, 1, bias=True),
            nn.ReLU(inplace=True))

        # residual blocks
        self.resblocks = nn.Sequential(RepVSRRB_nores(nf, 32, 1), RepVSRRB(32, 32, 2), \
                    RepVSRRB_nores(32, 16, 1), RepVSRRB(16, 16, 2),RepVSRRB_nores(16, 32, 2), \
                    RepVSRRB(32, 32, 2), RepVSRRB_nores(32, 48, 2),)
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


class SRNet1210_64(nn.Module):
    """ Reconstruction & Upsampling network
    """

    def __init__(self, in_nc=3, out_nc=3, nf=64, nb=6, upsample_func=None,
                 scale=4):
        super(SRNet1210_64, self).__init__()

        self.in_nc = in_nc
        if self.in_nc == 3:
            nf = 64
        # input conv.
        self.conv_in = nn.Sequential(
            nn.Conv2d((scale**2 + 1) * in_nc, nf, 3, 1, 1, bias=True),
            nn.ReLU(inplace=True))

        # residual blocks
        self.resblocks = nn.Sequential(*[RepVSRRB(nf, nf, 2) for _ in range(nb)])
        self.act = nn.ReLU(inplace=True)

        # upsampling function
        self.conv_up_pixelshuffle = nn.Sequential(
            nn.PixelShuffle(4),
            nn.ReLU(inplace=True))

        self.conv_out = nn.Conv2d(4, 3, 3, 1, 1, bias=True)


    def forward(self, lr_curr, hr_prev_tran):
        """ lr_curr: the current lr data in shape nchw
            hr_prev_tran: the previous transformed hr_data in shape n(4*4*c)hw
        """
        out = self.conv_in(torch.cat([lr_curr, hr_prev_tran], dim=1))
        for block in self.resblocks:
            out = block(out)
            out = self.act(out)
        out = self.conv_up_pixelshuffle(out)
        out = self.conv_out(out)
        return out


class SRNet1210_cascade(nn.Module):
    """ Reconstruction & Upsampling network
    """

    def __init__(self, in_nc=3, out_nc=3, nf=64, nb=6, upsample_func=None,
                 scale=4):
        super(SRNet1210_cascade, self).__init__()

        self.in_nc = in_nc
        if self.in_nc == 3:
            nf = 16
        # input conv.
        self.conv_in = nn.Sequential(
            nn.Conv2d((scale**2 + 1) * in_nc, nf, 3, 1, 1, bias=True),
            nn.ReLU(inplace=True))

        # residual blocks
        self.resblocks = nn.Sequential(*[RepVSRRB(nf, nf, 2) for _ in range(3)], \
                    RepVSRRB_nores(nf, 32, 1), RepVSRRB(32, 32, 2), RepVSRRB_nores(32, 48, 2))
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


class SRNet1210_trans(nn.Module):
    """ Reconstruction & Upsampling network
    """

    def __init__(self, in_nc=3, out_nc=3, nf=64, nb=6, upsample_func=None,
                 scale=4):
        super(SRNet1210_trans, self).__init__()

        self.in_nc = in_nc
        if self.in_nc == 3:
            nf = 16
        # input conv.
        self.conv_in = nn.Sequential(
            nn.Conv2d((scale**2 + 1) * in_nc, nf, 3, 1, 1, bias=True),
            nn.ReLU(inplace=True))

        # residual blocks
        self.resblocks = nn.Sequential(*[RepVSRRB(nf, nf, 2) for _ in range(8)])
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
