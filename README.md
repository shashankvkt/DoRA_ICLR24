# DoRA_ICLR24
This repo contains the official implementation of ICLR 2024 paper "Is ImageNet worth 1 video? Learning strong image encoders from 1 long unlabelled video""  

[[`arXiv`](https://arxiv.org/abs/2310.08584)], [[`paper`](https://openreview.net/forum?id=Yen1lGns2o)], [[`dataset`](https://uvaauas.figshare.com/articles/dataset/Dora_WalkingTours_Dataset_ICLR_2024_/25189275)], [[`Project Page`](https://shashankvkt.github.io/dora)]



## Overview
### Motivation
Our goal is to build robust representations by leveraging the rich information in video frames. Standard SSL frameworks such as SimCLR, DINo etc. often assume correspondences between different views. This is true whether using dense or global representations by pooling e.g. IBoT. While it is relatively straightforward to establish correspondences in images, it becomes more challenging when dealing with temporal deformations, requiring some form of object tracking. In videos with a large field of view or ego-motion, obtaining correspondences becomes even more difficult.  


<div align="center">
  <img width="100%" alt="DoRA illustration" src="images/dora.jpg">
</div>


## High-level idea 

We introduce DoRA, based on multi-object Discovery and tRAcking. It leverages the attention from the [CLS] token of distinct heads in a vision transformer to identify and consistently track multiple objects within a given frame across temporal sequences. On these, a teacher-student distillation loss is then applied. 
Importantly, we do not use any off-the-shelf object tracker or optical flow network. This keeps our pipeline simple and does not require any additional data or training. It also ensures that the learned representation is robust.

## Dataset Preparation

<div align="center">
  <img width="100%" alt="DoRA illustration" src="images/Wt_img.jpeg">
</div>