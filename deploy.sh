#CUDA_VISIBLE_DEVICES=0,4,7 python basicsr/deploy.py -opt options/deploy/RepSR/deploy_RepSR_x4_xt.yml
#CUDA_VISIBLE_DEVICES=4,7 python basicsr/deploy.py -opt options/deploy/RepSR/deploy_RepSR_x4_three.yml
#CUDA_VISIBLE_DEVICES=0,4,7 python basicsr/deploy.py -opt options/deploy/RepSR/deploy_RepSR_x4_xt_big.yml

CUDA_VISIBLE_DEVICES=4 python basicsr/deploy.py -opt options/deploy/x2SR/deploy_X2SR_x2_finetune.yml
# deploy_X2SR_x2 deploy_X2SR_x2_finetune