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
freeeze/bs
video-mae: 152/45 92/18 0/10
r3d18: 30/32
r3d50: 168/70
c2d50: 129/45
i3d50: 72/45
csn101: 282/45
slow50: 129/45
efficient-x3d-s, efficient-x3d-xs: 228/100
x3d-l: 455/90
x3d-m, x3d-s, x3d-xs: 226/110
mvit-base-16, mvit-base-16x4, mvit-base-32x3: 326/30
slowfast-50: 270/120
slowfast16x8-101: 576/120

done:
video-mae: 152/45 92/18 
r3d18: 30/32
r3d50: 168/70
c2d50: 129/45
i3d50: 72/45
csn101: 282/45
slow50: 129/45
efficient-x3d-s, efficient-x3d-xs: 228/100
x3d-l: 455/90
x3d-m, x3d-s, x3d-xs: 226/110
mvit-base-16, mvit-base-16x4, mvit-base-32x3: 326/30
slowfast-50: 270/120
slowfast16x8-101: 576/120 
'''
video_model_load = 'slowfast16x8-101' # video-mae
video_freeze_paras_before = 576
batch_size_list = [120]

mode = 'test' # train test
item_tower = 'video' # modal, text, image, video, id

epoch = 50
load_ckpt_name = 'epoch-19.pt'

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
                lr = lr_list[index]

                label_screen = '{}_bs{}_ed{}_lr{}_dp{}_L2{}_len{}'.format(
                        item_tower, batch_size, embedding_dim, lr,
                        drop_rate, weight_decay, max_seq_len)

                run_py = "CUDA_VISIBLE_DEVICES='2,3' \
                        /opt/anaconda3/envs/torch1.8/bin/python -m torch.distributed.launch \
                        --nproc_per_node 2 --master_port 102 main.py \
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