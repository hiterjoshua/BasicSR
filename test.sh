#CUDA_VISIBLE_DEVICES=7 python basicsr/test.py -opt options/test/RepSR/test_RepSR_x4.yml
#CUDA_VISIBLE_DEVICES=5 python basicsr/test.py -opt options/test/RepSR/test_RepSR_x4_xt.yml
CUDA_VISIBLE_DEVICES=5 python basicsr/test.py -opt options/test/RepSR/test_RepSR_x4_nossim.yml
#CUDA_VISIBLE_DEVICES=6 python basicsr/test.py -opt options/test/RepSR/test_RepSR_x4_deep15.yml