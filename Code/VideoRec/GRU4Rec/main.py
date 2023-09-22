from warnings import simplefilter
from transformers import logging
logging.set_verbosity_warning()
simplefilter(action='ignore', category=UserWarning)
simplefilter(action='ignore', category=FutureWarning)
simplefilter(action='ignore', category=DeprecationWarning)

import os
import re
import time
import torch
import random
import subprocess

import numpy as np
import torch.optim as optim
import torch.distributed as dist
import torchvision.models as models

from torch import nn
from pathlib import Path
from statistics import mode
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast as autocast
from transformers import CLIPVisionModel, SwinForImageClassification, ViTMAEModel
from transformers import BertModel, BertTokenizer, BertConfig, RobertaTokenizer, RobertaModel, RobertaConfig
from transformers import VideoMAEFeatureExtractor, VideoMAEModel, VideoMAEConfig
from pytorchvideo_rs.pytorchvideo.models.hub import mvit_base_16, mvit_base_16x4, mvit_base_32x3, slowfast_16x8_r101_50_50

from model.model import Model
# from model.model_dnn import Model
from utils.lr_decay import *
from utils.parameters import parse_args
from utils.load_data import read_texts, read_behaviors_text, read_items, read_behaviors, get_doc_input_bert, read_videos
from utils.logging_utils import para_and_log, report_time_train, report_time_eval, save_model, setuplogger, get_time
from utils.dataset import IdDataset, ModalDataset, TextDataset, ImageDataset, VideoDataset, LMDB_Image, LMDB_VIDEO
from utils.metrics import get_item_text_score, get_item_id_score, get_item_image_score, get_item_video_score, eval_model, get_fusion_score
# from utils.metrics_dnn import get_item_text_score, get_item_id_score, get_item_image_score, get_item_video_score, eval_model, get_fusion_score

os.environ['TOKENIZERS_PARALLELISM'] = 'false'
scaler = torch.cuda.amp.GradScaler()

def setup_seed(seed):
    '''
    global seed config
    '''
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

def train(args, model_dir, Log_file, Log_screen, start_time, local_rank):
    # ========================================== Text and Image Encoders ===========================================
    if 'modal' == args.item_tower or 'text' == args.item_tower:
        if 'roberta-base-en' in args.text_model_load:
            Log_file.info('load roberta model...')
            text_model_load = os.path.abspath(os.path.join(args.root_model_dir, 'pretrained_models/bert', args.text_model_load))
            tokenizer = RobertaTokenizer.from_pretrained(text_model_load)
            config = RobertaConfig.from_pretrained(text_model_load, output_hidden_states=True)
            text_model = RobertaModel.from_pretrained(text_model_load, config=config)
            pooler_para = [197, 198]
            args.word_embedding_dim = 768
            
        elif 'chinese-roberta-wwm-ext' in args.text_model_load:
            Log_file.info('load Chinese Roberta model...')
            text_model_load = os.path.abspath(os.path.join(args.root_model_dir, 'pretrained_models/bert', args.text_model_load))
            tokenizer = BertTokenizer.from_pretrained(text_model_load)
            text_model = BertModel.from_pretrained(text_model_load)
            pooler_para = [197, 198]
            args.word_embedding_dim = 768

        else:
            Log_file.info('load bert model...')
            text_model_load = os.path.abspath(os.path.join(args.root_model_dir, 'pretrained_models/bert', args.text_model_load))
            print(text_model_load)
            tokenizer = BertTokenizer.from_pretrained(text_model_load)
            config = BertConfig.from_pretrained(text_model_load, output_hidden_states=True)
            text_model = BertModel.from_pretrained(text_model_load, config=config)

            if 'tiny' in args.text_model_load:
                pooler_para = [37, 38]
                args.word_embedding_dim = 128
            if 'mini' in args.text_model_load:
                pooler_para = [69, 70]
                args.word_embedding_dim = 256
            if 'small' in args.text_model_load:
                pooler_para = [69, 70]
                args.word_embedding_dim = 512
            if 'medium' in args.text_model_load:
                pooler_para = [133, 134]
                args.word_embedding_dim = 512
            if 'base' in args.text_model_load:
                pooler_para = [197, 198]
                args.word_embedding_dim = 768
            if 'large' in args.text_model_load:
                pooler_para = [389, 390]
                args.word_embedding_dim = 1024

        for index, (name, param) in enumerate(text_model.named_parameters()):
            if index < args.text_freeze_paras_before or index in pooler_para:
                param.requires_grad = False

        if 'text' == args.item_tower:
            image_model = None
            video_model = None

    if 'modal' == args.item_tower or 'image' == args.item_tower:
        if 'vit-b-32-clip' in args.image_model_load:
            Log_file.info('load Vit model...')
            image_model_load = os.path.abspath(os.path.join(args.root_model_dir, 'pretrained_models', args.image_model_load))
            # vit of clip
            image_model = CLIPVisionModel.from_pretrained(image_model_load)

        elif 'vit-base-mae' in args.image_model_load:
            Log_file.info('load MAE model...')
            image_model_load = os.path.abspath(os.path.join(args.root_model_dir, 'pretrained_models', args.image_model_load))
            # mae
            image_model = ViTMAEModel.from_pretrained(image_model_load)

        elif 'swin_base' in args.image_model_load:
            image_model_load = os.path.abspath(os.path.join(args.root_model_dir, 'pretrained_models', args.image_model_load))
            image_model = SwinForImageClassification.from_pretrained(image_model_load)

        elif 'swin_tiny' in args.image_model_load:
            image_model_load = os.path.abspath(os.path.join(args.root_model_dir, 'pretrained_models', args.image_model_load))
            image_model = SwinForImageClassification.from_pretrained(image_model_load)

        elif 'resnet' in args.image_model_load:
            Log_file.info('load resnet model...')
            image_model_load = os.path.abspath(os.path.join(args.root_model_dir, 'pretrained_models', 'resnet', args.image_model_load))
            if '18' in image_model_load:
                image_model = models.resnet18(pretrained=False)
            elif '34' in image_model_load:
                image_model = models.resnet34(pretrained=False)
            elif '50' in image_model_load:
                image_model = models.resnet50(pretrained=False)
            elif '101' in image_model_load:
                image_model = models.resnet101(pretrained=False)
            elif '152' in image_model_load:
                image_model = models.resnet152(pretrained=False)
            else:
                image_model = None
            image_model.load_state_dict(torch.load(image_model_load))

        for index, (name, param) in enumerate(image_model.named_parameters()):
            # print(index,name,param.size())
            if index < args.image_freeze_paras_before and index >= 4:
                param.requires_grad = False

        if 'image' == args.item_tower:
            text_model = None
            video_model = None
    
    if 'modal' == args.item_tower or 'video' == args.item_tower:
        if 'video-mae' in args.video_model_load:
            Log_file.info('load video mae model...')
            configuration = VideoMAEConfig(num_frames=args.frame_no)
            video_model_load = os.path.abspath(os.path.join(args.root_model_dir, 'pretrained_models', 'videomae_base'))
            video_model = VideoMAEModel.from_pretrained(video_model_load, config = configuration)
        elif 'r3d18' in args.video_model_load:
            Log_file.info('load r3d18 model...')
            video_model = models.video.r3d_18(pretrained=True)
        elif 'r3d50' in args.video_model_load:
            Log_file.info('load r3d50 model...')
            video_model = torch.hub.load('./pytorchvideo_rs', 'r2plus1d_r50', source='local', head_pool_kernel_size=(1, 7, 7), pretrained=True)
        elif 'c2d50' in args.video_model_load:
            Log_file.info('load c2d50 model...')
            video_model = torch.hub.load('./pytorchvideo_rs', 'c2d_r50', source='local', head_pool_kernel_size=(1, 7, 7), pretrained=True)
        elif 'i3d50' in args.video_model_load:
            Log_file.info('load i3d50 model...')
            video_model = torch.hub.load('./pytorchvideo_rs', 'i3d_r50', source='local', head_pool_kernel_size=(1, 7, 7), pretrained=True)
        elif 'csn101' in args.video_model_load:
            Log_file.info('load csn101 model...')
            video_model = torch.hub.load('./pytorchvideo_rs', 'csn_r101', source='local', pretrained=True)
        elif 'slow50' in args.video_model_load:
            Log_file.info('load slow50 model...')
            video_model = torch.hub.load('./pytorchvideo_rs', 'slow_r50', source='local', pretrained=True)
        elif 'efficient-x3d-s' in args.video_model_load:
            Log_file.info('load efficient-x3d-s model...')
            video_model = torch.hub.load('./pytorchvideo_rs', 'efficient_x3d_s', source='local', pretrained=True)
        elif 'efficient-x3d-xs' in args.video_model_load:
            Log_file.info('load efficient-x3d-xs model...')
            video_model = torch.hub.load('./pytorchvideo_rs', 'efficient_x3d_xs', source='local', pretrained=True)
        elif 'x3d-xs' in args.video_model_load:
            Log_file.info('load x3d-xs model...')
            video_model = torch.hub.load('./pytorchvideo_rs', 'x3d_xs', source='local', pretrained=True)
        elif 'x3d-s' in args.video_model_load:
            Log_file.info('load x3d-s model...')
            video_model = torch.hub.load('./pytorchvideo_rs', 'x3d_s', source='local', pretrained=True)
        elif 'x3d-m' in args.video_model_load:
            Log_file.info('load x3d-m model...')
            video_model = torch.hub.load('./pytorchvideo_rs', 'x3d_m', source='local', pretrained=True)
        elif 'x3d-l' in args.video_model_load:
            Log_file.info('load x3d-l model...')
            video_model = torch.hub.load('./pytorchvideo_rs', 'x3d_l', source='local', pretrained=True)
        elif 'mvit-base-16' in args.video_model_load:
            Log_file.info('load mvit-base-16 model...')
            video_model = mvit_base_16(pretrained=False)
            video_model.load_state_dict(torch.load('/root/.cache/torch/hub/checkpoints/MVIT_B_16_f292487636.pyth'), strict=False)
        elif 'mvit-base-16x4' in args.video_model_load:
            Log_file.info('load mvit-base-16x4 model...')
            video_model = mvit_base_16x4(pretrained=False)
            video_model.load_state_dict(torch.load('/root/.cache/torch/hub/checkpoints/MVIT_B_16x4.pyth'), strict=False)
        elif 'mvit-base-32x3' in args.video_model_load:
            Log_file.info('load mvit-base-32x3 model...')
            video_model = mvit_base_32x3(pretrained=False)
            video_model.load_state_dict(torch.load('./MVIT_B_32x3_f294077834.pyth'), strict=False)
        elif 'slowfast-50' in args.video_model_load:
            Log_file.info('load slowfast50 model...')
            video_model = torch.hub.load('./pytorchvideo_rs', model='slowfast_r50', source='local', head_pool_kernel_sizes=((1, 7, 7), (4, 7, 7)), pretrained=True)
            # video_model = torch.hub.load('./pytorchvideo-main', model='slowfast_r50', source='local', head_pool_kernel_sizes=((1, 7, 7), (4, 7, 7)), pretrained=False)
        elif 'slowfast16x8-101' in args.video_model_load:
            Log_file.info('load slowfast16x8-101 model...')
            video_model = slowfast_16x8_r101_50_50(pretrained=False)
            video_model.load_state_dict(torch.load('SLOWFAST_16x8_R101_50_50.pyth'), strict=False)
            # video_model = torch.hub.load('./pytorchvideo_rs', model='slowfast_16x8_r101_50_50', source='local', pretrained=True)
            # video_model = torch.hub.load('./pytorchvideo-main', model='slowfast_16x8_r101_50_50', source='local', pretrained=False)

        for index, (name, param) in enumerate(video_model.named_parameters()):
            # print(index, name)
            if index < args.video_freeze_paras_before:
                param.requires_grad = False

        # xxx
        if 'video' == args.item_tower:
            text_model = None
            image_model = None

    if 'id' == args.item_tower:
        text_model = None
        image_model = None
        video_model = None

    # ========================================== Loading Data ===========================================
    item_content = None
    item_id_to_keys = None

    if 'modal' == args.item_tower or 'text' == args.item_tower:
        Log_file.info('read texts ...')
        item_dic_titles_after_tokenizer, before_item_name_to_index, before_item_index_to_name = read_texts(tokenizer, args)

        Log_file.info('read behaviors ...')
        item_num, item_dic_titles_after_tokenizer, item_name_to_index, users_train, users_valid, users_history_for_valid, pop_prob_list = \
            read_behaviors_text(item_dic_titles_after_tokenizer, before_item_name_to_index, before_item_index_to_name, Log_file, args)

        Log_file.info('combine text information...')
        text_title, text_title_attmask = get_doc_input_bert(item_dic_titles_after_tokenizer, item_name_to_index, args)

        item_content = np.concatenate([text_title, text_title_attmask], axis=1)

    if 'modal' == args.item_tower or 'image' == args.item_tower or 'video' == args.item_tower or 'id' == args.item_tower:
        Log_file.info('read images/videos/id...')
        if 'video' == args.item_tower:
            before_item_id_to_keys, before_item_name_to_id = read_videos(args.min_video_no, args.max_video_no)
        else:
            before_item_id_to_keys, before_item_name_to_id = read_items(args)

        Log_file.info('read behaviors...')
        item_num, item_id_to_keys, users_train, users_valid, users_history_for_valid, pop_prob_list = \
            read_behaviors(before_item_id_to_keys, before_item_name_to_id, Log_file, args)

    # ========================================== Building Model ===========================================
    Log_file.info('build model...')
    model = Model(args, pop_prob_list, item_num, text_model, image_model, video_model, item_content).to(local_rank)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(local_rank)

    if 'epoch' in args.load_ckpt_name:
        Log_file.info('load ckpt if not None...')
        #############
        
        #model_dir = './checkpoint/checkpoint_100wu_ks_pairs_id/cpt_v1_gru4rec_blocknum_1_tau_0.07_bs_128_ed_2048_lr_0.0005_l2_0.1_maxLen_10'
        ckpt_path = os.path.abspath(os.path.join(model_dir, args.load_ckpt_name))
        
        start_epoch = int(re.split(r'[._-]', args.load_ckpt_name)[1])
        
        checkpoint = torch.load(ckpt_path, map_location=torch.device('cpu'))
        Log_file.info('load checkpoint...')
        model.load_state_dict(checkpoint['model_state_dict'])
        Log_file.info(f'Model loaded from {args.load_ckpt_name}')
        torch.set_rng_state(checkpoint['rng_state'])  # load torch的随机数生成器状态
        torch.cuda.set_rng_state(checkpoint['cuda_rng_state'])  # load torch.cuda的随机数生成器状态
        is_early_stop = False
    else:
        checkpoint = None  # new
        ckpt_path = None  # new
        start_epoch = 0
        is_early_stop = False

    # for index, (name, param) in enumerate(model.named_parameters()):
    #     print(index, name, param.shape)
    
    Log_file.info('model.cuda()...')
    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    Log_file.info(model)
    # ============================ Dataset and Dataloader ============================

    if 'modal' == args.item_tower:
        Log_file.info('build  text and image dataset...')
        train_dataset = ModalDataset(u2seq=users_train,
                                    item_content=item_content,
                                    max_seq_len=args.max_seq_len,
                                    item_num=item_num,
                                    text_size=args.num_words_title,
                                    image_db_path=os.path.join(args.root_data_dir, args.dataset, args.image_data),
                                    video_db_path=os.path.join(args.root_data_dir, args.dataset, args.video_data),
                                    item_id_to_keys=item_id_to_keys,
                                    resize=args.image_resize,
                                    args=args)

    elif 'image' == args.item_tower:
        train_dataset = ImageDataset(u2seq=users_train,
                                    item_num=item_num,
                                    max_seq_len=args.max_seq_len,
                                    db_path=os.path.join(args.root_data_dir, args.dataset, args.image_data),
                                    item_id_to_keys=item_id_to_keys, 
                                    resize=args.image_resize)

    elif 'text' == args.item_tower:
        train_dataset = TextDataset(userseq=users_train, 
                                   item_content=item_content, 
                                   max_seq_len=args.max_seq_len,
                                   item_num=item_num, 
                                   text_size=args.num_words_title)

    elif 'video' == args.item_tower:
        train_dataset = VideoDataset(u2seq=users_train,
                                    item_num=item_num,
                                    max_seq_len=args.max_seq_len,
                                    db_path=os.path.join(args.root_data_dir, args.dataset, args.video_data),
                                    item_id_to_keys=item_id_to_keys,
                                    frame_no=args.frame_no)

    elif 'id' == args.item_tower:
        train_dataset = IdDataset(u2seq=users_train, 
                                 item_num=item_num, 
                                 max_seq_len=args.max_seq_len,
                                 args=args)

    Log_file.info('build DDP sampler...')
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)

    def worker_init_reset_seed(worker_id):
        initial_seed = torch.initial_seed() % 2 ** 31
        worker_seed = initial_seed + worker_id
        random.seed(worker_seed)
        np.random.seed(worker_seed)

    Log_file.info('build dataloader...')
    train_dl = DataLoader(train_dataset, 
                        batch_size=args.batch_size, 
                        num_workers=args.num_workers,
                        multiprocessing_context='fork', 
                        worker_init_fn=worker_init_reset_seed, 
                        pin_memory=True, 
                        sampler=train_sampler)

    # ============================ Optimizer ============================
    image_net_params = []
    video_net_params = []
    text_net_params = []
    recsys_params = []

    for index, (name, param) in enumerate(model.module.named_parameters()):
        if param.requires_grad:
            if 'image_encoder' in name:
                # if 'image_proj' in name or 'fc' in name: # 在vit中也有叫fc的所以容易冲撞，
                if 'image_proj' in name or 'classifier' in name:  # image_proj: clip_vit, mae ; classifier: swin_tiny, swin_base ; fc: resnet
                    recsys_params.append(param)
                elif 'resnet' in name and 'fc' in name:
                    recsys_params.append(param)
                else:
                    image_net_params.append(param)
            elif 'text_encoder' in name:
                if 'pooler' in name:
                    recsys_params.append(param)
                else:
                    text_net_params.append(param)
            elif 'video_encoder' in name:
                video_net_params.append(param)
            else:
                recsys_params.append(param)

    optimizer = optim.AdamW([
        {'params': text_net_params,'lr': args.text_fine_tune_lr, 'weight_decay': 0,  'initial_lr': args.text_fine_tune_lr},
        {'params': image_net_params, 'lr': args.image_fine_tune_lr, 'weight_decay': 0, 'initial_lr': args.image_fine_tune_lr},
        {'params': video_net_params, 'lr': args.video_fine_tune_lr, 'weight_decay': 0, 'initial_lr': args.video_fine_tune_lr},
        {'params': recsys_params, 'lr': args.lr, 'weight_decay': args.weight_decay, 'initial_lr': args.lr}
        ])
    # optimizer = optim.Adam([{'params':model.module.parameters(), 'lr':args.lr, 'initial_lr': args.lr}])

    for children_model in optimizer.state_dict()['param_groups']:
        Log_file.info('***** {} parameters have learning rate {} *****'.format(
            len(children_model['params']), children_model['lr']))

    Log_file.info('***** {} fine-tuned parameters in text encoder *****'.format(
        len(list(text_net_params))))
    Log_file.info('***** {} fiue-tuned parameters in image encoder*****'.format(
        len(list(image_net_params))))
    Log_file.info('***** {} fiue-tuned parameters in video encoder*****'.format(
        len(list(video_net_params))))
    Log_file.info('***** {} parameters with grad in recsys *****'.format(
        len(list(recsys_params))))

    model_params_require_grad = []
    model_params_freeze = []
    for param_name, param_tensor in model.module.named_parameters():
        if param_tensor.requires_grad:
            model_params_require_grad.append(param_name)
        else:
            model_params_freeze.append(param_name)

    Log_file.info('***** model: {} parameters require grad, {} parameters freeze *****'.format(
        len(model_params_require_grad), len(model_params_freeze)))

    if 'None' not in args.load_ckpt_name:   # load 优化器状态
        optimizer.load_state_dict(checkpoint['optimizer'])
        Log_file.info(f'optimizer loaded from {ckpt_path}')
    
    # ============================  training  ============================

    total_num = sum(p.numel() for p in model.module.parameters())
    trainable_num = sum(p.numel() for p in model.module.parameters() if p.requires_grad)
    Log_file.info('##### total_num {} #####'.format(total_num))
    Log_file.info('##### trainable_num {} #####'.format(trainable_num))

    Log_file.info('\n')
    Log_file.info('Training...')
    next_set_start_time = time.time()
    max_epoch, early_stop_epoch = 0, args.epoch
    max_eval_value, early_stop_count = 0, 0

    steps_for_log, steps_for_eval = para_and_log(model, len(users_train), args.batch_size, Log_file,
                                                logging_num=args.logging_num, testing_num=args.testing_num)
    Log_screen.info('{} train start'.format(args.label_screen))

    warmup_steps = 0
    if args.scheduler == "cosine_schedule_with_warmup":
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=args.epoch,
            start_epoch=start_epoch-1)
        
    elif args.scheduler == "linear_schedule_with_warmup":
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=args.epoch,
            start_epoch=start_epoch-1)
        
    elif args.scheduler == "step_schedule_with_warmup":
        lr_scheduler = get_step_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            gap_steps = args.scheduler_gap,
            scheduler_alpha = args.scheduler_alpha,
            start_epoch=start_epoch-1)
    else:
        raise ValueError("{} is not a valid scheduler.".format(args.scheduler))

    epoch_left = args.epoch - start_epoch

    for ep in range(epoch_left):
        now_epoch = start_epoch + ep + 1
        train_dl.sampler.set_epoch(now_epoch)
        loss, batch_index, need_break = 0.0, 1, False
        align, uniform = 0.0, 0.0
        
        if not need_break and (now_epoch-1) % 1 == 0 and now_epoch > 1:
            max_eval_value, max_epoch, early_stop_epoch, early_stop_count, need_break = \
                eval(now_epoch, max_epoch, early_stop_epoch, max_eval_value, early_stop_count, model, users_history_for_valid, \
                    users_valid, 64, item_num, args.mode, is_early_stop, local_rank, args, pop_prob_list, Log_file, item_content, item_id_to_keys)

        if args.mode == 'test':
            return
        
        Log_file.info('\n')
        Log_file.info('epoch {} start'.format(now_epoch))
        Log_file.info('')
        
        model.train()
        if lr_scheduler is not None:
            Log_file.info('start of trainin epoch:  {} ,lr: {}'.format(now_epoch, lr_scheduler.get_lr()))

        for data in train_dl:
            if 'modal' == args.item_tower:
                sample_items_id, sample_items_text, sample_items_image, sample_items_video, log_mask = data
                sample_items_id, sample_items_text, sample_items_image, sample_items_video, log_mask = \
                    sample_items_id.to(local_rank), sample_items_text.to(local_rank), \
                        sample_items_image.to(local_rank), sample_items_video.to(local_rank), log_mask.to(local_rank)
                sample_items_text = sample_items_text.view(-1, args.num_words_title * 2)
                sample_items_image = sample_items_image.view(-1, 3, args.image_resize, args.image_resize)
                sample_items_video = sample_items_video.view(-1, args.frame_no, 3, 224, 224)
                sample_items_id = sample_items_id.view(-1)

            elif 'text' == args.item_tower:
                sample_items_id, sample_items_text, log_mask = data
                sample_items_id, sample_items_text, log_mask = \
                    sample_items_id.to(local_rank), sample_items_text.to(local_rank), log_mask.to(local_rank)
                sample_items_text = sample_items_text.view(-1, args.num_words_title * 2)
                sample_items_id = sample_items_id.view(-1)
                sample_items_image = None
                sample_items_video = None

            elif 'image' == args.item_tower:
                sample_items_id, sample_items_image, log_mask = data
                sample_items_id, sample_items_image, log_mask = \
                    sample_items_id.to(local_rank), sample_items_image.to(local_rank), log_mask.to(local_rank)
                sample_items_image =  sample_items_image.view(-1, 3, args.image_resize, args.image_resize)
                sample_items_id = sample_items_id.view(-1)
                sample_items_text = None
                sample_items_video = None

            elif 'video' == args.item_tower:
                sample_items_id, sample_items_video, log_mask = data
                sample_items_id, sample_items_video, log_mask = \
                    sample_items_id.to(local_rank), sample_items_video.to(local_rank), log_mask.to(local_rank)
                sample_items_video =  sample_items_video.view(-1, args.frame_no, 3, 224, 224)
                sample_items_id = sample_items_id.view(-1)
                sample_items_text = None
                sample_items_image = None

            elif 'id' == args.item_tower:
                sample_items, log_mask = data
                sample_items, log_mask = sample_items.to(local_rank), log_mask.to(local_rank)
                sample_items_id = sample_items.view(-1)
                sample_items_text = None
                sample_items_image = None
                sample_items_video = None

            optimizer.zero_grad()

            # # 混合精度（加速）
            with autocast(enabled=True):
                bz_loss, bz_align, bz_uniform = model(sample_items_id, sample_items_text, sample_items_image, sample_items_video, log_mask, local_rank, args)
                loss += bz_loss.data.float()
                align += bz_align.data.float()
                uniform += bz_uniform.data.float()

            scaler.scale(bz_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.module.parameters(), max_norm=5, norm_type=2)
            scaler.step(optimizer)
            scaler.update()

            if torch.isnan(loss.data):
                need_break = True
                break

            #steps_for_log = 1
            if batch_index % steps_for_log == 0:
                Log_file.info('Ed: {}, batch loss: {:.3f}, sum loss: {:.3f}, align: {:.3f}, uniform: {:.3f}'.format(
                    batch_index * args.batch_size, loss.data / batch_index, loss.data, align / batch_index, uniform / batch_index))
            batch_index += 1

        if dist.get_rank() == 0 and now_epoch % args.save_step == 0:
            save_model(now_epoch, model, model_dir, optimizer, torch.get_rng_state(), torch.cuda.get_rng_state(), Log_file)   # new

        Log_file.info('')
        next_set_start_time = report_time_train(batch_index, now_epoch, loss, align/batch_index, uniform/batch_index, next_set_start_time, start_time, Log_file)
        Log_screen.info('{} training: epoch {}/{}'.format(args.label_screen, now_epoch, args.epoch))

        if need_break:
            break
        
        if lr_scheduler is not None:
            lr_scheduler.step()
            Log_file.info('end of trainin epoch:  {} ,lr: {}'.format(now_epoch, lr_scheduler.get_lr()))

    Log_file.info('\n')
    Log_file.info('%' * 90)
    Log_file.info('max eval Hit10 {:0.5f}  in epoch {}'.format(max_eval_value * 100, max_epoch-1))
    Log_file.info('early stop in epoch {}'.format(early_stop_epoch))
    Log_file.info('the End')
    Log_screen.info('{} train end in epoch {}'.format(args.label_screen, early_stop_epoch))

def eval(now_epoch, max_epoch, early_stop_epoch, max_eval_value, early_stop_count, model, user_history, users_eval, batch_size, item_num,\
     mode, is_early_stop, local_rank, args, pop_prob_list, Log_file, item_content=None, item_id_to_keys=None):

    eval_start_time = time.time()
    Log_file.info('Validating based on {}'.format(args.item_tower))

    if 'text' == args.item_tower:
        Log_file.info('get_text_scoring...')
        item_scoring = get_item_text_score(model, item_content, batch_size, args, local_rank)

    elif 'image' == args.item_tower:
        Log_file.info('get_image_scoring...')
        item_scoring = get_item_image_score(model, item_num, item_id_to_keys, batch_size, args, local_rank)
    
    elif 'video' == args.item_tower:
        Log_file.info('get_video_scoring...')
        item_scoring = get_item_video_score(model, item_num, item_id_to_keys, batch_size, args, local_rank)
    
    elif 'modal' == args.item_tower:
        Log_file.info('get_text_scoring...')
        item_scoring_text = get_item_text_score(model, item_content, batch_size, args, local_rank)
        Log_file.info('get_image_scoring...')
        item_scoring_image = get_item_image_score(model, item_num, item_id_to_keys, batch_size, args, local_rank)
        Log_file.info('get_video_scoring...')
        item_scoring_video = get_item_video_score(model, item_num, item_id_to_keys, batch_size, args, local_rank)

        item_scoring = get_fusion_score(model ,item_scoring_text, item_scoring_image, item_scoring_video, local_rank, args)

    elif 'id' == args.item_tower:
        item_scoring = get_item_id_score(model, item_num, batch_size, args, local_rank)

    valid_Hit10, nDCG10 = eval_model(model, user_history, users_eval, item_scoring, batch_size, \
        args, item_num, Log_file, mode, pop_prob_list, local_rank, now_epoch)

    report_time_eval(eval_start_time, Log_file)
    Log_file.info('')
    need_break = False

    if valid_Hit10 > max_eval_value:
        max_eval_value = valid_Hit10
        max_epoch = now_epoch
        early_stop_count = 0
    else:
        early_stop_count += 1
        if early_stop_count > 5:
            if is_early_stop:
                need_break = True
            early_stop_epoch = now_epoch

    return max_eval_value, max_epoch, early_stop_epoch, early_stop_count, need_break

def main():
    args = parse_args()

    # ============== Distributed Computation Config ==============
    local_rank = int(os.environ['RANK'])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl')
    
    # ============== Experiment and Logging Config ===============
    setup_seed(42 + dist.get_rank())  # magic number

    assert args.item_tower in ['modal', 'text', 'image', 'id', 'video']
    dir_label =  str(args.behaviors).strip().split('.')[0] + '_'  + str(args.item_tower)
    
    tag = args.version
    if 'modal' == args.item_tower:
        log_paras = f'{tag}_{args.model}_blocknum_{args.block_num}_tau_{args.tau}_bs_{args.batch_size}' \
                    f'_ed_{args.embedding_dim}_lr_{args.lr}' \
                    f'_l2_{args.weight_decay}_flrText_{args.text_fine_tune_lr}_flrImg_{args.image_fine_tune_lr}'\
                    f'_{args.text_model_load}_{args.image_model_load}' \
                    f'_freeze_{args.text_freeze_paras_before}_{args.image_freeze_paras_before}'\
                    f'_maxLen_{args.max_seq_len}'
    elif 'text' == args.item_tower:
        log_paras = f'{tag}_{args.model}_blocknum_{args.block_num}_tau_{args.tau}_bs_{args.batch_size}' \
                    f'_ed_{args.embedding_dim}_lr_{args.lr}' \
                    f'_l2_{args.weight_decay}_flrText_{args.text_fine_tune_lr}'\
                    f'_{args.text_model_load}'\
                    f'_freeze_{args.text_freeze_paras_before}'\
                    f'_maxLen_{args.max_seq_len}'
    elif 'image' == args.item_tower:
        log_paras = f'{tag}_{args.model}_blocknum_{args.block_num}_tau_{args.tau}_bs_{args.batch_size}' \
                    f'_ed_{args.embedding_dim}_lr_{args.lr}' \
                    f'_l2_{args.weight_decay}_flrImg_{args.image_fine_tune_lr}'\
                    f'_{args.image_model_load}'\
                    f'_freeze_{args.image_freeze_paras_before}'\
                    f'_maxLen_{args.max_seq_len}'
    elif 'video' == args.item_tower:
        log_paras = f'{tag}_{args.model}_blocknum_{args.block_num}_tau_{args.tau}_bs_{args.batch_size}' \
                    f'_fi_{args.frame_interval}_fn_{args.frame_no}' \
                    f'_ed_{args.embedding_dim}_lr_{args.lr}' \
                    f'_l2_{args.weight_decay}_flrVideo_{args.video_fine_tune_lr}'\
                    f'_{args.video_model_load}'\
                    f'_freeze_{args.video_freeze_paras_before}'\
                    f'_maxLen_{args.max_seq_len}'
    elif 'id' == args.item_tower:
        log_paras = f'{tag}_{args.model}_blocknum_{args.block_num}_tau_{args.tau}_bs_{args.batch_size}' \
                    f'_ed_{args.embedding_dim}_lr_{args.lr}' \
                    f'_l2_{args.weight_decay}' \
                    f'_maxLen_{args.max_seq_len}'


    model_dir = os.path.join('./checkpoint/checkpoint_' + dir_label, f'cpt_' + log_paras)
    time_run = '' # avoid redundant log records
    Log_file, Log_screen = setuplogger(dir_label, log_paras, time_run, args.mode, dist.get_rank())
    Log_file.info(args)

    if not os.path.exists(model_dir):
        Path(model_dir).mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    train(args, model_dir, Log_file, Log_screen, start_time, local_rank)

    end_time = time.time()
    hour, minute, seconds = get_time(start_time, end_time)
    Log_file.info('#### (time) all: hours {} minutes {} seconds {} ####'.format(hour, minute, seconds))


if __name__ == '__main__':
    main()
