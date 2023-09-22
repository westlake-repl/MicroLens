import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
root_data_dir = '/home/public/data/scale_datasets/'
root_model_dir = '/home/public/data/'

dataset = 'core_datasets'
tag = '100wu'
behaviors = tag + '_ks_pairs.tsv'
text_data = tag + '_ks_title.csv'
image_data = tag + '_ks_cover.lmdb'
frame_no = 5
video_data = tag + '_ks_fi1_fn'+str(frame_no)+'_frames.lmdb'
max_seq_len_list = [10]

logging_num = 10
testing_num = 1
save_step = 1

image_resize = 224
max_video_no = 34321 # 34321 for 10wu

text_model_load = 'chinese-roberta-wwm-ext' # 'bert-base-cn' 
image_model_load = 'vit-base-mae' # 'vit-b-32-clip'
video_model_load = 'video-mae' # video-mae

'''
videomae: freeeze 152 bs 32
r3d18: freeeze 30 bs 32
'''

# last 2 layer of trms
text_freeze_paras_before = 165
image_freeze_paras_before = 164
video_freeze_paras_before = 152

mode = 'train' # train test
item_tower = 'text' # modal, text, image, video, id

epoch = 50
load_ckpt_name = 'None'
# load_ckpt_name = 'epoch-200.pt'

weight_decay = 0.1
drop_rate = 0.1
batch_size_list = [256]

embedding_dim_list = [512]
lr_list = [1e-4]
text_fine_tune_lr_list = [5e-5]
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

                run_py = "CUDA_VISIBLE_DEVICES='0,1,2,3' \
                        /opt/anaconda3/envs/torch1.8/bin/python -m torch.distributed.launch \
                        --nproc_per_node 4 --master_port 103 main.py \
                        --root_data_dir {} --root_model_dir {} --dataset {} --behaviors {} --text_data {}  --image_data {} --video_data {}\
                        --mode {} --item_tower {} --load_ckpt_name {} --label_screen {} --logging_num {} --save_step {}\
                        --testing_num {} --weight_decay {} --drop_rate {} --batch_size {} --lr {} --embedding_dim {}\
                        --image_resize {} --image_model_load {} --text_model_load {} --video_model_load {} --epoch {} \
                        --text_freeze_paras_before {} --image_freeze_paras_before {} --video_freeze_paras_before {} --max_seq_len {} --frame_no {}\
                        --text_fine_tune_lr {} --image_fine_tune_lr {} --video_fine_tune_lr {}\
                        --scheduler {} --scheduler_gap {} --scheduler_alpha {} --max_video_no {}\
                        --version {}".format(
                        root_data_dir, root_model_dir, dataset, behaviors, text_data, image_data, video_data,
                        mode, item_tower, load_ckpt_name, label_screen, logging_num, save_step,
                        testing_num,weight_decay, drop_rate, batch_size, lr, embedding_dim,
                        image_resize, image_model_load, text_model_load, video_model_load, epoch,
                        text_freeze_paras_before, image_freeze_paras_before, video_freeze_paras_before, max_seq_len, frame_no,
                        text_fine_tune_lr, image_fine_tune_lr, video_fine_tune_lr, 
                        scheduler, scheduler_gap, scheduler_alpha, max_video_no,
                        version)
            
                os.system(run_py)