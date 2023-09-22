import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
root_data_dir = '/home/public/data/scale_datasets/'
root_model_dir = '/home/public/data/'

dataset = 'core_datasets'
tag = '10wu'
behaviors = tag + '_ks_pairs.tsv'
text_data = tag + '_ks_title.csv'
image_data = tag + '_ks_cover.lmdb'
frame_interval = 1
frame_no = 5
video_data = tag + '_ks_fi'+str(frame_interval)+'_fn'+str(frame_no)+'_frames.lmdb'
max_seq_len_list = [10]

logging_num = 10
testing_num = 1
save_step = 1

image_resize = 224
max_video_no = 34321 # 34321 for 10wu, 91717 for 100wu

text_model_load = 'bert-base-uncased' # 'bert-base-cn' 
image_model_load = 'vit-base-mae' # 'vit-b-32-clip'

# last 2 layer of trms
text_freeze_paras_before = 165
image_freeze_paras_before = 164


'''
model name: freeze layer/batch size
video-mae: 9999/45 152/45 92/18 0/10 (frozen, topT, halfT, FT, TFS)
r3d18: 9999/32 30/32 xx/xx 0/20
r3d50: 9999/70 168/70 xx/xx 0/10
c2d50: 9999/45 129/45 xx/xx 0/10
i3d50: 9999/45 72/45 xx/xx 0/15
csn101: 9999/45 282/45 xx/xx 0/10
slow50: 9999/45 129/45 xx/xx 0/10
efficient-x3d-s, efficient-x3d-xs: 9999/100 228/100 xx/xx 0/30
x3d-l: 9999/90 455/90 xx/xx 0/12
x3d-m, x3d-s, x3d-xs: 9999/110 226/110 xx/xx 0/20
mvit-base-16(0/5), mvit-base-16x4(0/5), mvit-base-32x3: 9999/30 326/30 192/30 0/10 0/10-scratch
slowfast-50: 9999/120 270/120 153/60 0/20 0/20-scratch
slowfast16x8-101: 9999/120 576/120 153/25 0/15 0/15-scratch
'''
video_model_load = 'x3d-s' # mvit-base-32x3 slowfast-50 slowfast16x8-101
video_freeze_paras_before = 0 # 326 270 576
batch_size_list = [20] # 30 120 120

mode = 'train' # train test
item_tower = 'video' # modal, text, image, video, id

epoch = 50
load_ckpt_name = 'None'
# load_ckpt_name = 'epoch-47.pt'

weight_decay = 0.1
drop_rate = 0.1

embedding_dim_list = [512]
lr_list = [1e-4]
text_fine_tune_lr_list = [1e-4]
image_fine_tune_lr_list = [1e-4]
video_fine_tune_lr_list = [1e-4]
index_list = [0]

scheduler = 'step_schedule_with_warmup'
scheduler_gap = 1
scheduler_alpha = 1
version = 'v1'

for batch_size in batch_size_list:
    for embedding_dim in embedding_dim_list:
        for max_seq_len in max_seq_len_list:
            for index in index_list:
                text_fine_tune_lr = text_fine_tune_lr_list[index]
                image_fine_tune_lr = image_fine_tune_lr_list[index]
                video_fine_tune_lr = video_fine_tune_lr_list[index]
                lr = lr_list[index]   # 两种fine_tune_lr要组合使用

                label_screen = '{}_bs{}_ed{}_lr{}_dp{}_L2{}_len{}'.format(
                        item_tower, batch_size, embedding_dim, lr,
                        drop_rate, weight_decay, max_seq_len)

                run_py = "CUDA_VISIBLE_DEVICES='0,1,2,3' \
                        /opt/anaconda3/envs/torch1.8/bin/python -m torch.distributed.launch \
                        --nproc_per_node 4 --master_port 128 main.py \
                        --root_data_dir {} --root_model_dir {} --dataset {} --behaviors {} --text_data {}  --image_data {} --video_data {}\
                        --mode {} --item_tower {} --load_ckpt_name {} --label_screen {} --logging_num {} --save_step {}\
                        --testing_num {} --weight_decay {} --drop_rate {} --batch_size {} --lr {} --embedding_dim {}\
                        --image_resize {} --image_model_load {} --text_model_load {} --video_model_load {} --epoch {} \
                        --text_freeze_paras_before {} --image_freeze_paras_before {} --video_freeze_paras_before {} --max_seq_len {} --frame_interval {} --frame_no {}\
                        --text_fine_tune_lr {} --image_fine_tune_lr {} --video_fine_tune_lr {}\
                        --scheduler {} --scheduler_gap {} --scheduler_alpha {} --max_video_no {}\
                        --version {}".format(
                        root_data_dir, root_model_dir, dataset, behaviors, text_data, image_data, video_data,
                        mode, item_tower, load_ckpt_name, label_screen, logging_num, save_step,
                        testing_num,weight_decay, drop_rate, batch_size, lr, embedding_dim,
                        image_resize, image_model_load, text_model_load, video_model_load, epoch,
                        text_freeze_paras_before, image_freeze_paras_before, video_freeze_paras_before, max_seq_len, frame_interval, frame_no,
                        text_fine_tune_lr, image_fine_tune_lr, video_fine_tune_lr, 
                        scheduler, scheduler_gap, scheduler_alpha, max_video_no,
                        version)
            
                os.system(run_py)