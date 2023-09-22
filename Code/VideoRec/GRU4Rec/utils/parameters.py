import os
import argparse

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
root_data_dir = os.path.abspath(os.path.join(BASE_DIR, '..'))

def parse_args():
    parser = argparse.ArgumentParser()

    # ============== data_dir ==============
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--item_tower', type=str, default='modal', choices=['video', 'modal', 'text', 'image', 'id'])
    parser.add_argument('--root_data_dir', type=str, default=root_data_dir)
    parser.add_argument('--root_model_dir', type=str, default=root_data_dir)
    parser.add_argument('--dataset', type=str, default=None)
    parser.add_argument('--behaviors', type=str, default=None)
    parser.add_argument('--text_data', type=str, default=None)
    parser.add_argument('--image_data', type=str, default=None)
    parser.add_argument('--video_data', type=str, default=None) 

    # ============== train parameters==============
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--epoch', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--drop_rate', type=float, default=None)
    parser.add_argument('--weight_decay', type=float, default=None)
    parser.add_argument('--scheduler', type=str, default=None)
    parser.add_argument('--scheduler_gap', type=int, default=None)
    parser.add_argument('--scheduler_alpha', type=float, default=None)
    parser.add_argument('--tau', type=float, default=0.07)
    parser.add_argument('--save_step', type=int, default=100)
    parser.add_argument('--neg_num', type=int, default=100)
    parser.add_argument('--power', type=float, default=1.0)
    parser.add_argument('--version', type=str, default='v3')
    parser.add_argument('--model', type=str, default='gru4rec')
    parser.add_argument('--block_num', type=int, default=2)

    # ============== model parameters for text ==============
    parser.add_argument('--text_model_load', type=str, default=None)
    parser.add_argument('--word_embedding_dim', type=int, default=768)
    parser.add_argument('--text_freeze_paras_before', type=int, default=None)
    parser.add_argument('--text_fine_tune_lr', type=float, default=None)

    # ============== text information ==============
    parser.add_argument('--num_words_title', type=int, default=30)

    # ============== model parameters for image ==============
    parser.add_argument('--image_model_load', type=str, default=None)
    parser.add_argument('--image_freeze_paras_before', type=int, default=None)
    parser.add_argument('--image_resize', type=int, default=None)
    parser.add_argument('--image_fine_tune_lr', type=float, default=None)

    # ============== model parameters for video ==============
    parser.add_argument('--video_model_load', type=str, default=None)
    parser.add_argument('--video_freeze_paras_before', type=int, default=None)
    parser.add_argument('--video_fine_tune_lr', type=float, default=None)
    parser.add_argument("--min_video_no", type=int, default=1)
    parser.add_argument("--max_video_no", type=int, default=-1)

    # ============== model parameters ==============
    parser.add_argument('--embedding_dim', type=int, default=512)
    parser.add_argument('--max_seq_len', type=int, default=23)
    parser.add_argument('--min_seq_len', type=int, default=5)
    parser.add_argument('--frame_interval', type=int, default=-1)
    parser.add_argument('--frame_no', type=int, default=-1)

    # ============== SASRec parameters ==============
    parser.add_argument('--num_attention_heads', type=int, default=2)
    parser.add_argument('--transformer_block', type=int, default=2)
    parser.add_argument('--dnn_layers', type=int, default=0)

    # ============== switch and logging setting ==============
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--load_ckpt_name', type=str, default=None)
    parser.add_argument('--label_screen', type=str, default=None)
    parser.add_argument('--logging_num', type=int, default=None)
    parser.add_argument('--testing_num', type=int, default=None)
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--node_rank', default=0, type=int)

    # ============== fusion methods ==============
    parser.add_argument('--fusion_method', type=str, default='concat', choices=['none', 'sum', 'concat', 'film', 'gated'])

    args = parser.parse_args()

    return args

