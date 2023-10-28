# A Content-Driven Micro-Video Recommendation Dataset at Scale

# Invited Talk by Google DeepMind [(Slides)](https://github.com/westlake-repl/MicroLens/blob/master/MicroLens_DeepMind_Talk.pdf)

# Dataset

Download links: https://recsys.westlake.edu.cn/MicroLens-50k-Dataset/ and https://recsys.westlake.edu.cn/MicroLens-100k-Dataset/.

<!-- Dataset downloader (for Windows): https://github.com/microlens2023/microlens-dataset/blob/master/Downloader/microlens_downloader.exe

Dataset downloader (for Linux): https://github.com/microlens2023/microlens-dataset/blob/master/Downloader/microlens_downloader

Dataset downloader (for Mac): https://github.com/microlens2023/microlens-dataset/blob/master/Downloader/microlens_downloader_mac

For review purposes, we are temporarily releasing a portion of our Microlens dataset.

We have uploaded a MicroLens-TOY folder, which contains 100 randomly sampled videos from the Microlens dataset. The folder includes cover images, audio files, video content, and textual captions for all 100 videos.

Additionally, we have provided a MicroLens-100K folder, which consists of the MicroLens-100K_pairs.tsv file containing interaction pairs (each row indicates a user and the videos they interacted with, sorted by interaction timestamp), along with audio files, textual captions, and corresponding watermarked cover files for all videos in the MicroLens-100K dataset. Please note that video content for MicroLens-100K is currently not available.

For various types of modal data and the interaction pairs of MicroLens-100K, MicroLens-1M, and MicroLens, we will release all of them once the paper is accepted. -->

## Videos uploaded (2023.10.28 updated).

We have temporarily released MicroLens-50K (50,000 users) and MicroLens-100K (100,000 users) along with their associated multimodal data, including raw text, images, audio, video, and video comments. You can access them through the provided link. To acquire the complete MicroLens dataset, kindly reach out to the corresponding author via email. If you have an innovative idea for building a foundational recommendation model but require a large dataset and computational resources, consider joining our lab as an intern. We can provide access to 100 NVIDIA 80G A100 GPUs and a billion-level dataset of user-video/image/text interactions.

We also release a subset of our MicroLens for multimodal fairness recommendation, which can be downloaded from https://recsys.westlake.edu.cn/MicroLens-Fairness-Dataset/

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

# Kind Reminder

Note that you should not perform secondary processing on the dataset provided in this paper and offer a new download. If you need to do so, we encourage you to open source only the code for data processing and cite our work for data downloading and local processing.

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

# Baseline Evaluation

<div align=center><img src="https://github.com/westlake-repl/MicroLens/blob/master/Results/baseline_evaluation.png"/></div>

# Video Understanding Meets Recommender Systems

<div align=center><img src="https://github.com/westlake-repl/MicroLens/blob/master/Results/video_meets_rs.png"/></div>

# News
#### The laboratory is hiring research assistants, interns, doctoral students, and postdoctoral researchers. Please contact the corresponding author for details.
#### 实验室招聘科研助理，实习生，博士生和博士后，请联系通讯作者。
