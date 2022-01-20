#CUDA_VISIBLE_DEVICES=7 python basicsr/train.py -opt options/train/RepSR/train_RepSR_x4.yml
# CUDA_VISIBLE_DEVICES=0 python basicsr/train.py -opt options/train/RepSR/train_RepSR_x4_xt.yml
#CUDA_VISIBLE_DEVICES=5 python basicsr/train.py -opt options/train/RepSR/train_RepSR_x4_nossim.yml
#CUDA_VISIBLE_DEVICES=6 python basicsr/train.py -opt options/train/RepSR/train_RepSR_x4_deep15.yml
#CUDA_VISIBLE_DEVICES=2 python basicsr/train.py -opt options/train/RepSR/train_RepSR_x4_three.yml


CUDA_VISIBLE_DEVICES=4 python basicsr/train.py -opt options/train/x2SR/train_X2SR_x2_scale.yml
#train_X2SR_x2_finetune train_X2SR_x2 train_X2SR_x2_deconv train_X2SR_x2_scale
