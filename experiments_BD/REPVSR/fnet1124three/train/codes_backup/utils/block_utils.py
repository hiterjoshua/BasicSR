import torch
import torch.nn as nn
import torch.nn.functional as F

def reparameter_13(s_1, s_2):   
    if isinstance(s_1, nn.Conv2d):
        w_s_1 = s_1.weight  # output# * input# * kernel * kernel
        b_s_1 = s_1.bias
    else:
        w_s_1 = s_1['weight']
        b_s_1 = s_1['bias']
    if isinstance(s_2, nn.Conv2d):
        w_s_2 = s_2.weight
        b_s_2 = s_2.bias
    else:
        w_s_2 = s_2['weight']
        b_s_2 = s_2['bias']
    
    device = w_s_1.get_device()
    if device < 0:
        device = None
    device = torch.device("cuda")

    fused = torch.nn.Conv2d(
        s_1.in_channels,
        s_2.out_channels,
        kernel_size=s_2.kernel_size,
        stride=s_2.stride,
        padding=s_2.padding,
        bias=True
    )
    w_s_1_ = w_s_1.view(w_s_1.size(0), w_s_1.size(1) * w_s_1.size(2) * w_s_1.size(3))
    w_s_2_tmp = w_s_2.view(w_s_2.size(0), w_s_2.size(1),  w_s_2.size(2) * w_s_2.size(3))

    new_weight = torch.Tensor(w_s_2.size(0), w_s_1.size(1), w_s_2.size(2)*w_s_2.size(3))
    for i in range(w_s_2.size(0)):
        tmp = w_s_2_tmp[i, :, :].view( w_s_2.size(1),  w_s_2.size(2) * w_s_2.size(3))
        new_weight[i, :, :] = torch.matmul(w_s_1_.t(), tmp)
    new_weight = new_weight.view(w_s_2.size(0), w_s_1.size(1),  w_s_2.size(2), w_s_2.size(3))

    if b_s_1 is not None and b_s_2 is not None:
        b_sum = torch.sum(w_s_2_tmp, dim=2)
        new_bias = torch.matmul(b_sum, b_s_1) + b_s_2  # with bias
    elif b_s_1 is None and b_s_2 is not None:
        new_bias = b_s_2  #without Bias
    else:
        new_bias = torch.zeros(s_2.weight.size(0))

    fused.weight.data = new_weight
    fused.bias.data = new_bias
    return fused

def reparameter_31(s_1, s_2):
    if isinstance(s_1, nn.Conv2d):
        w_s_1 = s_1.weight  # output# * input# * kernel * kernel
        b_s_1 = s_1.bias
    else:
        w_s_1 = s_1['weight']
        b_s_1 = s_1['bias']
    if isinstance(s_2, nn.Conv2d):
        w_s_2 = s_2.weight
        b_s_2 = s_2.bias
    else:
        w_s_2 = s_2['weight']
        b_s_2 = s_2['bias']

    device = w_s_1.get_device()
    if device < 0:
        device = None
    device = torch.device("cuda")

    fused = torch.nn.Conv2d(
        s_1.in_channels,
        s_2.out_channels,
        kernel_size=s_1.kernel_size,
        stride=s_1.stride,
        padding=s_1.padding,
        bias=True
    )

    w_s_1_ = w_s_1.view(w_s_1.size(0), w_s_1.size(1),  w_s_1.size(2) * w_s_1.size(3))
    w_s_2_ = w_s_2.view(w_s_2.size(0), w_s_2.size(1) * w_s_2.size(2) * w_s_2.size(3))
    new_weight_ = torch.Tensor(w_s_2.size(0), w_s_1.size(1), w_s_1.size(2)*w_s_1.size(3))
    for i in range(w_s_1.size(1)):
        tmp = w_s_1_[:, i, :].view(w_s_1.size(0),  w_s_1.size(2) * w_s_1.size(3))
        new_weight_[:, i, :] = torch.matmul(w_s_2_.to(device), tmp.to(device))
    new_weight = new_weight_.view(w_s_2.size(0), w_s_1.size(1),  w_s_1.size(2), w_s_1.size(3))

    if b_s_1 is not None and b_s_2 is not None:
        new_bias = torch.matmul(w_s_2_, b_s_1) + b_s_2  # with bias
    elif b_s_1 is None and b_s_2 is not None:
        new_bias = b_s_2  #without Bias
    else:
        new_bias = torch.zeros(s_2.weight.size(0))

    fused.weight.data = new_weight
    fused.bias.data = new_bias
    return fused

class SeqConv3x3(nn.Module):
    def __init__(self, seq_type, inp_planes, out_planes, depth_multiplier):
        super(SeqConv3x3, self).__init__()

        self.type = seq_type
        self.inp_planes = inp_planes
        self.out_planes = out_planes
        self.mid_planes = int(out_planes * depth_multiplier)

        if self.type == 'conv1x1-conv3x3':
            conv0 = torch.nn.Conv2d(self.inp_planes, self.mid_planes, kernel_size=1, padding=0)
            self.k0 = conv0.weight
            self.b0 = conv0.bias

            conv1 = torch.nn.Conv2d(self.mid_planes, self.out_planes, kernel_size=3)
            self.k1 = conv1.weight
            self.b1 = conv1.bias

        elif self.type == 'conv1x1-conv3x3-conv1x1':
            self.padding = nn.ZeroPad2d(1)

            self.conv0 = torch.nn.Conv2d(self.inp_planes, self.mid_planes, kernel_size=1, padding=0)
            self.conv1 = torch.nn.Conv2d(self.mid_planes, self.mid_planes, kernel_size=3, padding=0)
            self.conv2 = torch.nn.Conv2d(self.mid_planes, self.out_planes, kernel_size=1, padding=0)
            self.k0 = self.conv0.weight
        
        elif self.type == 'conv3x3-conv1x1':
            self.padding = nn.ZeroPad2d(1)

            self.conv0 = torch.nn.Conv2d(self.inp_planes, self.mid_planes, kernel_size=3, padding=0)
            self.conv1 = torch.nn.Conv2d(self.mid_planes, self.out_planes, kernel_size=1, padding=0)
            self.k0 = self.conv0.weight

        elif self.type == 'conv1x1-laplacian':
            conv0 = torch.nn.Conv2d(self.inp_planes, self.out_planes, kernel_size=1, padding=0)
            self.k0 = conv0.weight
            self.b0 = conv0.bias

            # init scale & bias
            scale = torch.randn(size=(self.out_planes, 1, 1, 1)) * 1e-3
            self.scale = nn.Parameter(torch.FloatTensor(scale))
            # bias = 0.0
            # bias = [bias for c in range(self.out_planes)]
            # bias = torch.FloatTensor(bias)
            bias = torch.randn(self.out_planes) * 1e-3
            bias = torch.reshape(bias, (self.out_planes,))
            self.bias = nn.Parameter(torch.FloatTensor(bias))
            # init mask
            self.mask = torch.zeros((self.out_planes, 1, 3, 3), dtype=torch.float32)
            for i in range(self.out_planes):
                self.mask[i, 0, 0, 1] = 1.0
                self.mask[i, 0, 1, 0] = 1.0
                self.mask[i, 0, 1, 2] = 1.0
                self.mask[i, 0, 2, 1] = 1.0
                self.mask[i, 0, 1, 1] = -4.0
            self.mask = nn.Parameter(data=self.mask, requires_grad=False)
        else:
            raise ValueError('the type of seqconv is not supported!')

    def forward(self, x):
        if self.type == 'conv1x1-conv3x3':
            # conv-1x1
            y0 = F.conv2d(input=x, weight=self.k0, bias=self.b0, stride=1)
            # explicitly padding with bias
            y0 = F.pad(y0, (1, 1, 1, 1), 'constant', 0)
            b0_pad = self.b0.view(1, -1, 1, 1)
            y0[:, :, 0:1, :] = b0_pad
            y0[:, :, -1:, :] = b0_pad
            y0[:, :, :, 0:1] = b0_pad
            y0[:, :, :, -1:] = b0_pad
            # conv-3x3
            y1 = F.conv2d(input=y0, weight=self.k1, bias=self.b1, stride=1)
        
        elif self.type == 'conv1x1-conv3x3-conv1x1':
            y1 = self.conv2(self.conv1(self.conv0(self.padding(x))))

        elif self.type == 'conv3x3-conv1x1':
            y1 = self.conv1(self.conv0(self.padding(x)))
            
        else:
            y0 = F.conv2d(input=x, weight=self.k0, bias=self.b0, stride=1)
            # explicitly padding with bias
            y0 = F.pad(y0, (1, 1, 1, 1), 'constant', 0)
            b0_pad = self.b0.view(1, -1, 1, 1)
            y0[:, :, 0:1, :] = b0_pad
            y0[:, :, -1:, :] = b0_pad
            y0[:, :, :, 0:1] = b0_pad
            y0[:, :, :, -1:] = b0_pad
            # conv-3x3
            y1 = F.conv2d(input=y0, weight=self.scale * self.mask, bias=self.bias, stride=1, groups=self.out_planes)
        return y1
    
    def rep_params(self):
        device = self.k0.get_device()
        if device < 0:
            device = None

        if self.type == 'conv1x1-conv3x3':
            # re-param conv kernel
            RK = F.conv2d(input=self.k1, weight=self.k0.permute(1, 0, 2, 3))
            # re-param conv bias
            RB = torch.ones(1, self.mid_planes, 3, 3, device=device) * self.b0.view(1, -1, 1, 1)
            RB = F.conv2d(input=RB, weight=self.k1).view(-1,) + self.b1
        
        elif self.type == 'conv1x1-conv3x3-conv1x1':
            # re-param conv kernel
            temp = reparameter_13(self.conv0, self.conv1)
            fused = reparameter_31(temp, self.conv2)
            return fused.weight.to(device), fused.bias.to(device)

        elif self.type == 'conv3x3-conv1x1':
            # re-param conv kernel
            fused = reparameter_31(self.conv0, self.conv1)
            return fused.weight.to(device), fused.bias.to(device)

        else:
            tmp = self.scale * self.mask
            k1 = torch.zeros((self.out_planes, self.out_planes, 3, 3), device=device)
            for i in range(self.out_planes):
                k1[i, i, :, :] = tmp[i, 0, :, :]
            b1 = self.bias
            # re-param conv kernel
            RK = F.conv2d(input=k1, weight=self.k0.permute(1, 0, 2, 3))
            # re-param conv bias
            RB = torch.ones(1, self.out_planes, 3, 3, device=device) * self.b0.view(1, -1, 1, 1)
            RB = F.conv2d(input=RB, weight=k1).view(-1,) + b1
        return RK, RB



class RepVSRRB(nn.Module):
    def __init__(self, inp_planes, out_planes, depth_multiplier, deploy_flag = False):
        super(RepVSRRB, self).__init__()

        self.depth_multiplier = depth_multiplier
        self.inp_planes = inp_planes
        self.out_planes = out_planes
        self.deploy = deploy_flag
        
        if self.deploy:
            self.rbr_reparam = nn.Conv2d(self.inp_planes, self.out_planes, kernel_size=3, padding=1, bias=True)
        else:
            self.conv3x3 = torch.nn.Conv2d(self.inp_planes, self.out_planes, kernel_size=3, padding=1)
            self.conv1x1_3x3 = SeqConv3x3('conv1x1-conv3x3', self.inp_planes, self.out_planes, self.depth_multiplier)
            self.conv1x1_3x3_1x1 = SeqConv3x3('conv1x1-conv3x3-conv1x1', self.inp_planes, self.out_planes, self.depth_multiplier)   
            self.conv3x3_1x1 = SeqConv3x3('conv3x3-conv1x1', self.inp_planes, self.out_planes, self.depth_multiplier)       
            self.conv1x1_lpl = SeqConv3x3('conv1x1-laplacian', self.inp_planes, self.out_planes, -1)

    def forward(self, x):
        if hasattr(self, 'rbr_reparam'):
            y = self.rbr_reparam(x)
        else:
            y = self.conv3x3(x)     + \
                self.conv1x1_3x3(x) + \
                self.conv1x1_3x3_1x1(x) + \
                self.conv3x3_1x1(x) + \
                self.conv1x1_lpl(x)
            y += x
        return y

    def rep_params(self):
        K0, B0 = self.conv3x3.weight, self.conv3x3.bias
        K1, B1 = self.conv1x1_3x3.rep_params()
        K2, B2 = self.conv1x1_3x3_1x1.rep_params()
        K3, B3 = self.conv3x3_1x1.rep_params()
        K4, B4 = self.conv1x1_lpl.rep_params()
        RK, RB = (K0+K1+K2+K3+K4), (B0+B1+B2+B3+B4)

        device = RK.get_device()
        if device < 0:
            device = None
        K_idt = torch.zeros(self.out_planes, self.out_planes, 3, 3, device=device)
        for i in range(self.out_planes):
            K_idt[i, i, 1, 1] = 1.0
        B_idt = 0.0
        RK, RB = RK + K_idt, RB + B_idt
        return RK, RB
    
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
        self.__delattr__('conv1x1_3x3')
        self.__delattr__('conv1x1_3x3_1x1')
        self.__delattr__('conv3x3_1x1')
        self.__delattr__('conv1x1_lpl')


if __name__ == '__main__':

    # # test seq-conv
    x = torch.randn(1, 3, 5, 5)* 200
    conv = SeqConv3x3('conv1x1-conv3x3-conv1x1', 3, 3, 2)
    y0 = conv(x)
    RK, RB = conv.rep_params()
    y1 = F.conv2d(input=x, weight=RK, bias=RB, stride=1, padding=1)
    print('conv1x1-conv3x3-conv1x1: ', torch.mean(torch.abs(y0-y1)))

    # test seq-conv
    x = torch.randn(1, 3, 128, 128)* 200
    conv = SeqConv3x3('conv1x1-laplacian', 3, 3, 2)
    y0 = conv(x)
    RK, RB = conv.rep_params()
    y1 = F.conv2d(input=x, weight=RK, bias=RB, stride=1, padding=1)
    print('conv1x1-laplacian: ', torch.mean(torch.abs(y0-y1)))

    # test seq-conv
    x = torch.randn(1, 3, 128, 128)* 200
    conv = SeqConv3x3('conv1x1-conv3x3', 3, 3, 2)
    y0 = conv(x)
    RK, RB = conv.rep_params()
    y1 = F.conv2d(input=x, weight=RK, bias=RB, stride=1, padding=1)
    print('conv1x1-conv3x3: ', torch.mean(torch.abs(y0-y1)))

        # test seq-conv
    x = torch.randn(1, 3, 128, 128)* 200
    conv = SeqConv3x3('conv3x3-conv1x1', 3, 3, 2)
    y0 = conv(x)
    RK, RB = conv.rep_params()
    y1 = F.conv2d(input=x, weight=RK, bias=RB, stride=1, padding=1)
    print('conv3x3-conv1x1: ', torch.mean(torch.abs(y0-y1)))

    # test repvsrrb
    x = torch.randn(1, 16, 5, 5) * 200
    repvsrrb = RepVSRRB(16, 16, 4)
    y0 = repvsrrb(x)

    RK, RB = repvsrrb.rep_params()
    y1 = F.conv2d(input=x, weight=RK, bias=RB, stride=1, padding=1)
    print('RepVSRRB after reparameterization: ', torch.mean(torch.abs(y0-y1)))