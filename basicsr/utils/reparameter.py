import torch
import torch.nn as nn
import torch.nn.functional as F


def reparameter_13(s_1, s_2):
    """
    Compute weights from s_1 and s_2
    :param s_1: 1*1 conv layer
    :param s_2: 3*3 conv layer
    :return: new weight and bias
    """
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

    fused.weight.data = new_weight.cuda()
    fused.bias.data = new_bias.cuda()
    return fused


def reparameter_31(s_1, s_2):
    """
    compute weights from former computation and last 1*1 conv layer
    :param s_1: 3*3 conv layer
    :param s_2: 1*1 conv layer
    :return: new weight and bias
    """
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
        new_weight_[:, i, :] = torch.matmul(w_s_2_, tmp)
    new_weight = new_weight_.view(w_s_2.size(0), w_s_1.size(1),  w_s_1.size(2), w_s_1.size(3))

    if b_s_1 is not None and b_s_2 is not None:
        new_bias = torch.matmul(w_s_2_, b_s_1) + b_s_2  # with bias
    elif b_s_1 is None and b_s_2 is not None:
        new_bias = b_s_2  #without Bias
    else:
        new_bias = torch.zeros(s_2.weight.size(0))

    fused.weight.data = new_weight.cuda()
    fused.bias.data = new_bias.cuda()
    return fused


def reparameter_33(s_1, s_2):
    """
    Compute weight from 2 conv layers, whose kernel size larger than 3*3
    After derivation, F.conv_transpose2d can be used to compute weight of original conv layer
    :param s_1: 3*3 or larger conv layer
    :param s_2: 3*3 or larger conv layer
    :return: new weight and bias
    """
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
    
    fused = torch.nn.Conv2d(
        s_1.in_channels,
        s_2.out_channels,
        kernel_size=s_2.kernel_size,
        stride=s_2.stride,
        padding=1,
        bias=True
    )

    w_s_2_tmp = w_s_2.view(w_s_2.size(0), w_s_2.size(1), w_s_2.size(2) * w_s_2.size(3))

    if b_s_1 is not None and b_s_2 is not None:
        b_sum = torch.sum(w_s_2_tmp, dim=2)
        new_bias = torch.matmul(b_sum, b_s_1) + b_s_2
    elif b_s_1 is None and b_s_2 is not None:
        new_bias = b_s_2  #without Bias
    else:
        new_bias = torch.zeros(s_2.weight.size(0))

    new_weight = F.conv_transpose2d(w_s_2, w_s_1)

    fused.weight.data = new_weight
    fused.bias.data = new_bias
    return fused


def reparameter_fc(s_1, s_2):
    """
    Compute weight from 2 fc layers
    :param s_1: p * n
    :param s_2: m * p
    :return: new weight m*n and bias
    """
    if isinstance(s_1, nn.Linear):
        w_s_1 = s_1.weight
        b_s_1 = s_1.bias
    else:
        w_s_1 = s_1['weight']
        b_s_1 = s_1['bias']
    if isinstance(s_2, nn.Linear):
        w_s_2 = s_2.weight
        b_s_2 = s_2.bias
    else:
        w_s_2 = s_2['weight']
        b_s_2 = s_2['bias']

    fused = torch.nn.Conv2d(
        s_2.in_channels,
        s_1.out_channels,
        kernel_size=s_2.kernel_size,
        stride=s_2.stride,
        padding=s_2.padding,
        bias=True
    )

    if b_s_1 is not None and b_s_2 is not None:
        new_bias = torch.matmul(w_s_2, b_s_1) + b_s_2
    elif b_s_1 is None and b_s_2 is not None:
        new_bias = b_s_2  #without Bias
    else:
        new_bias = torch.zeros(s_2.weight.size(0))

    new_weight = torch.matmul(w_s_2, w_s_1)

    fused.weight.data = new_weight.cuda()
    fused.bias.data = new_bias.cuda()
    return fused