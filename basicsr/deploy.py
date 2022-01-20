import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from os import path as osp

from basicsr.data import build_dataloader, build_dataset
from basicsr.models import build_model
from basicsr.utils import get_env_info, get_root_logger, get_time_str, make_exp_dirs
from basicsr.utils.options import dict2str, parse_options

def compute_ck(s_1, s_2):
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

    w_s_2_tmp = w_s_2.view(w_s_2.size(0), w_s_2.size(1), w_s_2.size(2) * w_s_2.size(3))

    if b_s_1 is not None and b_s_2 is not None:
        b_sum = torch.sum(w_s_2_tmp, dim=2)
        new_bias = torch.matmul(b_sum, b_s_1) + b_s_2
    elif b_s_1 is None and b_s_2 is not None:
        new_bias = b_s_2  #without Bias
    else:
        new_bias = torch.zeros(s_2.weight.size(0))

    new_weight = F.conv_transpose2d(w_s_2, w_s_1)

    return {'weight': new_weight, 'bias': new_bias}

def onnx_trans(model):
    import torch
    import torch.onnx
    import os
    # create random input
    input_data = torch.randn(1,1,2040,1530).cuda()

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    model.eval()

    # Forward pass
    output = model(input_data)

    # 设置输入张量名，多个输入就是多个名
    input_names = ["input"]
    # 设置输出张量名
    output_names = ["output"]

    # Export model to onnx
    onnx_path = "./onnx/"
    if not os.path.exists(onnx_path):
        os.mkdir(onnx_path)
    if not os.path.exists(tf_path):
        os.mkdir(tf_path)
    filename_onnx = onnx_path + "net_g_300000_reparam-2040x1530.onnx"

    torch.onnx.export(model, input_data, filename_onnx, input_names=input_names, 
    output_names=output_names, opset_version=11, do_constant_folding=True)


def test_pipeline(root_path):
    # parse options, set distributed setting, set ramdom seed
    opt = parse_options(root_path, is_train=False)

    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    # mkdir and initialize loggers
    make_exp_dirs(opt)
    log_file = osp.join(opt['path']['log'], f"test_{opt['name']}_{get_time_str()}.log")
    logger = get_root_logger(logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
    logger.info(get_env_info())
    logger.info(dict2str(opt))

    # create test dataset and dataloader
    test_loaders = []
    for phase, dataset_opt in sorted(opt['datasets'].items()):
        test_set = build_dataset(dataset_opt)
        test_loader = build_dataloader(
            test_set, dataset_opt, num_gpu=opt['num_gpu'], dist=opt['dist'], sampler=None, seed=opt['manual_seed'])
        logger.info(f"Number of test images in {dataset_opt['name']}: {len(test_set)}")
        test_loaders.append(test_loader)

        # create model
        model = build_model(opt)
        if opt['convert_flag']:
            save_path = opt['path'].get('pretrain_network_g', None)
            reparam_name = save_path.split('/')[-1].split('.')[0].split('_')[-1] + '_reparam'
            if reparam_name is not None:
                model.save_reparam(reparam_name)

        # onnx transfomation to do the time test for single image, add by hukunlei 20220118
        # onnx_trans(model.net_g)

        for test_loader in test_loaders:
            test_set_name = test_loader.dataset.opt['name']
            logger.info(f'Testing {test_set_name}...')
            model.validation(test_loader, current_iter=opt['name'], tb_logger=None, save_img=opt['val']['save_img'])


        # validation
        if not opt['convert_flag']:
            model_linear = torch.nn.Sequential(
                nn.ZeroPad2d(3),
                model.net_g.body[0].conv3x3_1,
                model.net_g.body[0].conv3x3_3,
                model.net_g.body[0].conv3x3_3
            )
            input = torch.randn(1, 64, 256, 256).cuda()
            f1 = model_linear.forward(input) + input

            from basicsr.utils.reparameter import reparameter_13, reparameter_31, reparameter_33
            fused = torch.nn.Conv2d(
                model.net_g.body[0].conv3x3_1.in_channels,
                model.net_g.body[0].conv3x3_2.out_channels,
                kernel_size=7,
                stride=1,
                padding=7//2,
                bias=True
            )
            res = compute_ck(model_linear[1], model_linear[2])
            res = compute_ck(res, model_linear[3])
            kernel_identity = torch.zeros((64, 64, 7, 7))
            for i in range(64):
                kernel_identity[i, i, 3, 3] = 1
            fused.weight.data = res['weight'] + kernel_identity.cuda()
            fused.bias.data = res['bias']
            f2 = fused.forward(input)
            d = torch.mean(torch.abs(f1 - f2))
            print("error:",d)

            m =  nn.ZeroPad2d(3)
            m1 = model.net_g.body[0].conv3x3_1
            m2 = model.net_g.body[0].conv3x3_3
            m3 = model.net_g.body[0].conv3x3_3
            f3 = m3(m2(m1(m(input))))+input
            d = torch.mean(torch.abs(f3 - f2))
            print("error no sequential:",d)

if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    test_pipeline(root_path)