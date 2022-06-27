# EGVSR-PyTorch
[GitHub](https://github.com/Thmen/EGVSR) **|** [Gitee码云](https://gitee.com/Digital-Design/EGVSR)

<p align = "center">
    <img src="results/city.gif" width="480" /><br>VSR x4: EGVSR; Upscale x4: Bicubic Interpolation
</p>


## Contents
- [EGVSR-PyTorch](#egvsr-pytorch)
  - [Contents](#contents)
  - [Introduction](#introduction)
  - [Features](#features)
  - [Dependencies](#dependencies)
  - [Datasets](#datasets)
    - [A. Training Dataset](#a-training-dataset)
    - [B. Testing Datasets](#b-testing-datasets)
  - [Benchmarks](#benchmarks)
    - [A. Test on Vid4 Dataset](#a-test-on-vid4-dataset)
    - [B. Test on Tos3 Dataset](#b-test-on-tos3-dataset)
    - [C. Test on Gvt72 Dataset](#c-test-on-gvt72-dataset)
    - [D. Optical-Flow based Motion Compensation](#d-optical-flow-based-motion-compensation)
    - [E. Comprehensive Performance](#e-comprehensive-performance)
  - [License & Citations](#license--citations)
  - [Acknowledgements](#acknowledgements)


## Introduction
This is a PyTorch implementation of **EGVSR**: **E**fficcient & **G**eneric Video Super-Resolution (**VSR**), using subpixel convolution to optimize the inference speed of TecoGAN VSR model. Please refer to the official implementation [ESPCN](https://github.com/leftthomas/ESPCN) and [TecoGAN](https://github.com/thunil/TecoGAN) for more information.


## Features
- **Unified Framework**: This repo provides a unified framework for various state-of-the-art DL-based VSR methods, such as VESPCN, SOFVSR, FRVSR, TecoGAN and our EGVSR.
- **Multiple Test Datasets**: This repo offers three types of video datasets for testing, i.e., standard test dataset -- Vid4, Tos3 used in TecoGAN and our new dataset -- Gvt72 (selected from [Vimeo](https://vimeo.com) site and including more scenes).
- **Better Performance**: This repo provides model with faster inferencing speed and better overall performance than prior methods. See more details in [Benchmarks](#benchmarks) section.


## Dependencies
- Ubuntu >= 16.04
- NVIDIA GPU + CUDA & CUDNN
- Python 3
- PyTorch >= 1.0.0
- Python packages: numpy, matplotlib, opencv-python, pyyaml, lmdb ([requirements.txt](requirements.txt) & [req.txt](req.txt))
- (Optional) Matlab >= R2016b


## Datasets
### A. Training Dataset
Download the official training dataset based on the instructions in [TecoGAN-TensorFlow](https://github.com/thunil/TecoGAN), rename to `VimeoTecoGAN` and then place under `./data`.

### B. Testing Datasets
* Vid4 -- Four video sequences: city, calendar, foliage and walk;
* Tos3 -- Three video sequences: bridge, face and room;
* Gvt72 -- Generic VSR Test Dataset: 72 video sequences (including natural scenery, culture scenery, streetscape scene, life record, sports photography, etc, as shown below)

<p align = "center">
    <img src="results/gvt72_preview.gif" width="640" />
</p>

You can get them at :arrow_double_down: [百度网盘](https://pan.baidu.com/s/1lKyLJ5u6lrrXejyljao0Mw) (提取码:8tqc) and put them into :file_folder: [Datasets](data).
The following shows the structure of the above three datasets.
```tex
data
  ├─ Vid4
    ├─ GT                # Ground-Truth (GT) video sequences
      └─ calendar
        ├─ 0001.png
        └─ ...
    ├─ Gaussian4xLR      # Low Resolution (LR) video sequences in gaussian degradation and x4 down-sampling
      └─ calendar
        ├─ 0001.png
        └─ ...
  └─ ToS3
    ├─ GT
    └─ Gaussian4xLR
  └─ Gvt72
    ├─ GT
    └─ Gaussian4xLR
```


## Benchmarks
**Experimental Environment**
|        | Version            | Info.      |
| ------ |:------------------ | :--------- |
| System | Ubuntu 18.04.5 LTS | X86_64     |
| CPU    | Intel i9-9900      | 3.10GHz    |
| GPU    | Nvidia RTX 2080Ti  | 11GB GDDR6 |
| Memory | DDR4 2666          | 32GB×2    |

### A. Test on Vid4 Dataset
<p align = "center">
    <img src="results/cmp_Vid4_calendar_20.png"/>
    <img src="results/cmp_Vid4_city_20.png"/>
    <img src="results/cmp_Vid4_foliage_20.png"/>
    <img src="results/cmp_Vid4_walk_10.png"/>
    <br> 1.LR 2.VESPCN 3.SOFVSR 4.DUF 5.Ours:EGVSR 6.GT
    <img src="results/vid4_benchmarks.png" width='640'/>
    <br>Objective metrics for visual quality evaluation[1]
</p>

### B. Test on Tos3 Dataset
<p align = "center">
    <img src="results/input_img_ToS3.png"/>
    <img src="results/cmp_ToS3_bridge_13945.png"/>
    <img src="results/cmp_ToS3_face_9945.png"/>
    <img src="results/cmp_ToS3_room_4400.png"/>
    <br> 1.VESPCN 2.SOFVSR 3. FRVSR 4.TecoGAN 5.Ours:EGVSR 6.GT
    
</p>
    
### C. Test on Gvt72 Dataset
<p align = "center">
    <img src="results/cmp_Gvt72.png"/>
    <br> 1.LR 2.VESPCN 3.SOFVSR 4.DUF 5.Ours:EGVSR 6.GT    
    <img src="results/benchmarks.png" width='800'/>
    <br> Objective metrics for visual quality and temporal coherence evaluation[1]
</p>

### D. Optical-Flow based Motion Compensation

Please refer to [FLOW_walk](results/seq_flow_walk.png), [FLOW_foliage](results/seq_flow_foliage.png) and [FLOW_city](results/seq_flow_city.png).

### E. Comprehensive Performance

<!-- <script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>

$$M_{nor}=(M-M_{min})/(M_{max}-M_{min})) \\
Score=1-(\lambda_{1}LPIPS_{nor}+\lambda_{2}tOF{nor}+\lambda_{3}tLP100_{nor})[2]$$ -->


<p align = "center">
  <img src="resources/score_formula.png" width="540" />
  <img src="results/performance.png" width="640" />
  <br> Comparison of various SOTA VSR model on video quality score and speed performance[3]
</p>


> <sup>[1]</sup> :arrow_down::smaller value for better performance, :arrow_up:: on the contrary; Red: stands for Top1, Blue: Top2.
> <sup>[2]</sup> The calculation formula of video quality score considering both spatial and temporal domain, using lambda1=lambda2=lambda3=1/3.
> <sup>[3]</sup> FLOPs & speed are computed on RGB with resolution 960x540 to 3840x2160 (4K) on NVIDIA GeForce GTX 2080Ti GPU.


## License & Citations
This EGVSR project is released under the MIT license. See more details in [LICENSE](LICENSE).
The provided implementation is strictly for academic purposes only. If EGVSR helps your research or work, please consider citing EGVSR. The following is a BibTeX reference:

``` latex
@misc{thmen2021egvsr,
  author =       {Yanpeng Cao, Chengcheng Wang, Changjun Song, Yongming Tang and He Li},
  title =        {EGVSR},
  howpublished = {\url{https://github.com/Thmen/EGVSR}},
  year =         {2021}
}
```

> Yanpeng Cao, Chengcheng Wang, Changjun Song, Yongming Tang and He Li. EGVSR. https://github.com/Thmen/EGVSR, 2021.


## Acknowledgements
This code is built on the following projects. We thank the authors for sharing their codes.
1. [ESPCN](https://github.com/leftthomas/ESPCN)
2. [BasicSR](https://github.com/xinntao/BasicSR)
3. [VideoSuperResolution](https://github.com/LoSealL/VideoSuperResolution)
4. [TecoGAN-PyTorch](https://github.com/skycrapers/TecoGAN-PyTorch)


## Code usage
1. DATA: When download the frames from Vimeo and Youtube datasets online, you would get the separated images as groundtruth data, and this file dataloader asked user to use lmdb format data. 
scirpts/create_lmdb.py resize_BD_BI.py could be used for lmdb generation and lr transformation, but noticed that only hr lmdb demanded in this repo.
Also pay attention to vemioGAN dataset, there are image rename python files in this repo.
2. Doing: change SRNet to smaller network, namned RepFrvsr network.
3. TODO: FNet reparameterazation and lmdb change, input channel and output channel changed from 3 to 1, follow the more advanced style--YUV style.

## Project launch
bash profile.sh BD REPVSR 1x144x180
bash test.sh BD REPVSR
bash train.sh BD REPVSR
Git push: git push -u origin master:EGVSR

### A. 20211102 update:
SRNet changed to smaller network and renamed RepVSR network, lmdb load style and input channels ought to be modified in recent time, take YUV problem as urgent affair.

### B. 20211105 update:
YUV single channel mode has been finished, include data processing, vsr_model infer stage code and gauss blur function rewrite.

### C. 20211112 update:
Finished Repameterization part, could change the origin architecture to the rep arch
Finished Fnet simplification, and start Fnet training, the number of parameters is decreased rapidly, which proves the powerful rep trick in deep learning. 

### D. 20211119 update:
finished the FNet part, now the network change a lot, the number of parameter is about 24k respectively, which include SRNet 17k and FNet 7k. 

### E. 20211128 update:
实验细节记录部分：在REPVSR目录下，进行的是VSR部分效果与效率实验的部分
001：重头训练达到网络pretrain的效果
singleway：对SRNet进行简化，将三通道修改为单通道
newfnet/fnet1117：对SRNet进行简化，外加FNet进行第一版本的简化处理，大概是1-3-1的重参数化结构
fnet1119：仿照SRNet，对FNet简化，上下采样数由8简化到4；效果不佳

fnet1124：SRNet数目为6，FNet上下采样恢复为8，同时加大内部channel数目,RepVSR-S
fnet1124three：在上面的基础上恢复三通道策略,RepVSR-L
fnet1124srnet：在上面的基础上，SRNet数目为6，FNet直接恢复为初始形式；

fnet1203three：在fnet1124three的基础上将通道数更改为16，16转48，48直接做pixel shuffle的操作

SRNet_120648: 在fnet1124three的基础上将通道数更改为48，减少不必要的参数量
SRNet_1206cascade：对6的block进行渐变的block变化，16-32-48
SRNet_1206trans：一直是16，最后使用反卷积；避免pixel-shuffle对通道数的要求

SRNet1210_64:原来64通道的模型，block内部升维改为2，通道有改进，大概是48-32-16-32-48的递进改变
SRNet1210_48：48重新实验，升维改为2
SRNet1210_cascade：cascade做改进，升维改为2，因为是16进，所以保持16-32-48
SRNet1210_trans：一直是16，最后使用反卷积；block数目定为10；升维度改为2

fnet0125three_long: fnet1124three基础上将sr block数目由6升到9，panr有0.1db提升
fnet0529three_enhance ：fnet1124three基础上将sr block数目由6升到10，且光流网络使用的是Fnet_enhance

### F. 20220601 update:
add tensorboaed part 