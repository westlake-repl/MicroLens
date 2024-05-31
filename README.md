# [MicroLens: A Content-Driven Micro-Video Recommendation Dataset at Scale](https://arxiv.org/pdf/2309.15379.pdf)

<a href="https://arxiv.org/pdf/2309.15379.pdf" alt="paper"><img src="https://img.shields.io/badge/ArXiv-2309.06789-FAA41F.svg?style=flat" /></a>
<a href="https://github.com/westlake-repl/MicroLens/blob/master/MicroLens_DeepMind_Talk.pdf" alt="Talk"><img src="https://img.shields.io/badge/Talk-DeepMind-orange" /></a> 
<a href="https://medium.com/@lifengyi_6964/building-a-large-scale-short-video-recommendation-dataset-and-benchmark-06e744746555" alt="blog"><img src="https://img.shields.io/badge/Blog-Medium-purple" /></a> 
<a href="https://zhuanlan.zhihu.com/p/675213913" alt="zhihu"><img src="https://img.shields.io/badge/Zhihu-知乎-blue" /></a> 
 
![Multi-Modal](https://img.shields.io/badge/Task-Multi--Modal-red) 
![Foundation-Model](https://img.shields.io/badge/Task-Foundation--Model-red) 
![Video-Understanding](https://img.shields.io/badge/Task-Video--Understanding-red) 
![Video-Generation](https://img.shields.io/badge/Task-Video--Generation-red) 
![Video-Recommendation](https://img.shields.io/badge/Task-Video--Recommendation-red) 

Quick Links: [🗃️Dataset](#Dataset) |
[📭Citation](#Citation) |
[🛠️Code](#Code) |
[🚀Baseline Evaluation](#Baseline_Evaluation) |
[🤗Video Understanding Meets Recommender Systems](#Video_Understanding_Meets_Recommender_Systems) |
[💡News](#News)

<p align="center" width="100%">
  <img src='https://camo.githubusercontent.com/ace7effc2b35cda2c66d5952869af563e851f89e5e1af029cfc9f69c7bebe78d/68747470733a2f2f692e696d6775722e636f6d2f77617856496d762e706e67' width="100%">
</p>

<!--## We provide support for a range of tasks, including **Short Video Generation** related to popular models like **Stable Diffusion** and **Sora**, **General Video Understanding** tasks, and **Video Recommendation**.-->

<!--# A Content-Driven Micro-Video Recommendation Dataset at Scale-->

# Talks & Slides: Invited Talk by Google DeepMind & YouTube & Alipay [(Slides)](https://github.com/westlake-repl/MicroLens/blob/master/MicroLens_DeepMind_Talk.pdf)

# Dataset

Download links: https://recsys.westlake.edu.cn/MicroLens-50k-Dataset/ and https://recsys.westlake.edu.cn/MicroLens-100k-Dataset/

**Email us if you find the link is not available.**

<div align=center><img src="https://github.com/westlake-repl/MicroLens/blob/master/Results/dataset.png"/></div>

<!-- Dataset downloader (for Windows): https://github.com/microlens2023/microlens-dataset/blob/master/Downloader/microlens_downloader.exe

Dataset downloader (for Linux): https://github.com/microlens2023/microlens-dataset/blob/master/Downloader/microlens_downloader

Dataset downloader (for Mac): https://github.com/microlens2023/microlens-dataset/blob/master/Downloader/microlens_downloader_mac

For review purposes, we are temporarily releasing a portion of our Microlens dataset.

We have uploaded a MicroLens-TOY folder, which contains 100 randomly sampled videos from the Microlens dataset. The folder includes cover images, audio files, video content, and textual captions for all 100 videos.

Additionally, we have provided a MicroLens-100K folder, which consists of the MicroLens-100K_pairs.tsv file containing interaction pairs (each row indicates a user and the videos they interacted with, sorted by interaction timestamp), along with audio files, textual captions, and corresponding watermarked cover files for all videos in the MicroLens-100K dataset. Please note that video content for MicroLens-100K is currently not available.

For various types of modal data and the interaction pairs of MicroLens-100K, MicroLens-1M, and MicroLens, we will release all of them once the paper is accepted. -->

## News

- **2024/05/31**: The "like" and "view" data for each video has been uploaded, please see [MicroLens-50k_likes_and_views.txt](https://recsys.westlake.edu.cn/MicroLens-50k-Dataset/MicroLens-50k_likes_and_views.txt) and [MicroLens-100k_likes_and_views.txt](https://recsys.westlake.edu.cn/MicroLens-100k-Dataset/MicroLens-100k_likes_and_views.txt).

- **2024/04/15**: Our dataset has been added to the MMRec framework, see https://github.com/enoche/MMRec/tree/master/data.

- **2024/04/04**: We have provided extracted multi-modal features (text/images/videos) of MicroLens-100k for multimodal recommendation tasks, see https://recsys.westlake.edu.cn/MicroLens-100k-Dataset/extracted_modality_features/. The preprocessed code is uploaded, see [video_feature_extraction_(from_lmdb).py](https://github.com/westlake-repl/MicroLens/blob/master/Data%20Processing/video_feature_extraction_(from_lmdb).py).

- **2024/03/01**: We have updated the command example for automatically downloading all videos, see https://github.com/westlake-repl/MicroLens/blob/master/Downloader/quick_download.txt.

- **2023/10/21**: We also release a subset of our MicroLens with extracted features for multimodal fairness recommendation, which can be downloaded from https://recsys.westlake.edu.cn/MicroLens-Fairness-Dataset/

- **2023/09/28**: We have temporarily released MicroLens-50K (50,000 users) and MicroLens-100K (100,000 users) along with their associated multimodal data, including raw text, images, audio, video, and video comments. You can access them through the provided link. To acquire the complete MicroLens dataset, kindly reach out to the corresponding author via email. If you have an innovative idea for building a foundational recommendation model but require a large dataset and computational resources, consider joining our lab as an intern. We can provide access to 100 NVIDIA 80G A100 GPUs and a billion-level dataset of user-video/image/text interactions.

# Citation
If you use our dataset, code or find MicroLens useful in your work, please cite our paper as:

```bib
@article{ni2023content,
  title={A Content-Driven Micro-Video Recommendation Dataset at Scale},
  author={Ni, Yongxin and Cheng, Yu and Liu, Xiangyan and Fu, Junchen and Li, Youhua and He, Xiangnan and Zhang, Yongfeng and Yuan, Fajie},
  journal={arXiv preprint arXiv:2309.15379},
  year={2023}
}
```

> :warning: **Caution**: It's prohibited to privately modify the dataset and then offer secondary downloads. If you've made alterations to the dataset in your work, you are encouraged to open-source the data processing code, so others can benefit from your methods. Or notify us of your new dataset so we can put it on this Github with your paper.


# Code

We have released the codes for all algorithms, including VideoRec (which implements all 15 video models in this project), IDRec, and VIDRec. For more details, please refer to the following paths: "Code/VideoRec", "Code/IDRec", and "Code/VIDRec". Each folder contains multiple subfolders, with each subfolder representing the code for a baseline.

## Special instructions on VideoRec

In VideoRec, if you wish to switch to a different training mode, please execute the following Python scripts: 'run_id.py', 'run_text.py', 'run_image.py', and 'run_video.py'. For testing, you can use 'run_id_test.py', 'run_text_test.py', 'run_image_test.py', and 'run_video_test.py', respectively. Please see the path "Code/VideoRec/SASRec" for more details.

Before running the training script, please make sure to modify the dataset path, item encoder, pretrained model path, GPU devices, GPU numbers, and hyperparameters. Additionally, remember to specify the best validation checkpoint (e.g., 'epoch-30.pt') before running the test script.

Note that you will need to prepare an LMDB file and specify it in the scripts before running image-based or video-based VideoRec. To assist with this, we have provided a Python script for LMDB generation. Please refer to 'Data Generation/generate_cover_frames_lmdb.py' for more details.

## Special instructions on IDRec and VIDRec

In IDRec, see `IDRec\process_data.ipynb` to process the interaction data.  Execute the following Python scripts: 'main.py'  under each folder to run the corresponding baselines. The data path, model parameters can be modified by changing the `yaml` file under each folder. 

## Environments
```
python==3.8.12
Pytorch==1.8.0
cudatoolkit==11.1
torchvision==0.9.0
transformers==4.23.1
```

# Baseline_Evaluation

<div align=center><img src="https://github.com/westlake-repl/MicroLens/blob/master/Results/baseline_evaluation.png"/></div>

# Video_Understanding_Meets_Recommender_Systems

<div align=center><img src="https://github.com/westlake-repl/MicroLens/blob/master/Results/video_meets_rs.png"/></div>

# Ad
#### The laboratory is hiring research assistants, interns, doctoral students, and postdoctoral researchers. Please contact the corresponding author for details.
#### 实验室招聘科研助理，实习生，博士生和博士后，请联系通讯作者。
