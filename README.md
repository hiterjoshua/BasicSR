# :rocket: BasicSR-RepSR

2021.07: add RepSR network

2021.08: add collapse function and deploy operation, all the code finished and tested

## :Deployment: Step by Step
set the convert_flag: true and deploy_flag: false in the deploy yaml, and run the deploy.sh, you would find the converted model in reparam folder.

Once you finished the convert process, you could set convert_flag: false and deploy_flag: true, and load the converted model by changing deploy yaml parameters, now the deploy file could be treated as a test function file, and you could run a test.

### A starting steps
bash train.sh
bash test.sh
bash deploy.sh

### B code skills
1. real3w data python 文件内部 有关于训练文件名同名或者不同名，有samename_flag来控制不同文件夹的读取。