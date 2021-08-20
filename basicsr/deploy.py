import logging
import torch
from os import path as osp

from basicsr.data import build_dataloader, build_dataset
from basicsr.models import build_model
from basicsr.utils import get_env_info, get_root_logger, get_time_str, make_exp_dirs
from basicsr.utils.options import dict2str, parse_options


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

    from basicsr.utils.reparameter import reparameter_13, reparameter_31, reparameter_33
    model_linear = torch.nn.Sequential(
    model.net_g.body[0].conv1x1_1,
    model.net_g.body[0].conv3x3_1,
    model.net_g.body[0].conv1x1_2
    )
    input = torch.randn(1, 64, 256, 256).cuda()
    fused = reparameter_13(model_linear[0], model_linear[1])
    fused_ = reparameter_31(fused, model_linear[2])
    f2 = fused_.forward(input)

    f1 = model_linear.forward(input)

    d = (f1 - f2).sum().item()
    print("error ddd:",d)










    if opt['convert_flag']:
        save_path = opt['path'].get('pretrain_network_g', None)
        reparam_name = save_path.split('/')[-1].split('.')[0].split('_')[-1] + '_reparam'
        if reparam_name is not None:
            model.save_reparam(reparam_name)

    for test_loader in test_loaders:
        test_set_name = test_loader.dataset.opt['name']
        logger.info(f'Testing {test_set_name}...')
        model.validation(test_loader, current_iter=opt['name'], tb_logger=None, save_img=opt['val']['save_img'])


if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    test_pipeline(root_path)
