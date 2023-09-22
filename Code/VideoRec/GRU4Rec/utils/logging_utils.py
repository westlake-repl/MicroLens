import logging
import os
import torch
import argparse
import time
import math
import torch.distributed as dist

# def str2bool(v):
#     if isinstance(v, bool):
#         return v
#     if v.lower() in ('yes', 'true', 't', 'y', '1'):
#         return True
#     elif v.lower() in ('no', 'false', 'f', 'n', '0'):
#         return False
#     else:
#         raise argparse.ArgumentTypeError('Boolean value expected.')

def setuplogger(dir_label, log_paras, time_run, mode, rank):
    log_code = None
    if 'train' in mode or 'load' in mode:
        log_code = 'train'
    if 'test' in mode:
        log_code = 'test'

    formatter = logging.Formatter('[%(levelname)s %(asctime)s] %(message)s')
    Log_file = logging.getLogger('Log_file')
    Log_screen = logging.getLogger('Log_screen')

    if rank in [-1, 0]:
        log_path = os.path.join('./logs/logs_' + dir_label + '_' + log_code)
        if not os.path.exists(log_path):
            os.makedirs(log_path)

        log_file_name = os.path.join(log_path, 'log_' + log_paras + time_run + '.log')

        Log_file.setLevel(logging.INFO)
        Log_screen.setLevel(logging.INFO)

        th = logging.FileHandler(filename=log_file_name, encoding='utf-8')
        th.setLevel(logging.INFO)
        th.setFormatter(formatter)
        Log_file.addHandler(th)

        handler = logging.StreamHandler()
        handler.setLevel(logging.INFO)
        handler.setFormatter(formatter)
        Log_screen.addHandler(handler)
        Log_file.addHandler(handler)
    else:
        Log_file.setLevel(logging.WARN)
        Log_screen.setLevel(logging.WARN)
    return Log_file, Log_screen

def get_time(start_time, end_time):
    time_g = int(end_time - start_time)
    hour = int(time_g / 3600)
    minu = int(time_g / 60) % 60
    secon = time_g % 60
    return hour, minu, secon

def para_and_log(model, seq_num, batch_size, Log_file, logging_num, testing_num):
    total_num = sum(p.numel() for p in model.module.parameters())
    trainable_num = sum(p.numel() for p in model.module.parameters() if p.requires_grad)
    Log_file.info('##### total_num {} #####'.format(total_num))
    Log_file.info('##### trainable_num {} #####'.format(trainable_num))

    step_num = math.ceil(seq_num / dist.get_world_size() / batch_size)
    Log_file.info('##### all {} steps #####'.format(step_num))
    steps_for_log = int(step_num / logging_num)
    steps_for_test = int(step_num / testing_num)
    Log_file.info('##### {} logs/epoch; {} steps/log #####'.format(logging_num, steps_for_log))
    Log_file.info('##### {} tests/epoch; {} steps/test #####'.format(testing_num, steps_for_test))
    return steps_for_log, steps_for_test

def report_time_train(batch_index, now_epoch, loss, align, uniform, next_set_start_time, start_time, Log_file):
    loss /= batch_index
    Log_file.info('epoch: {} end, train_loss: {:.5f}, align: {:.5f}, uniform: {:.5f}'.format(now_epoch, loss, align, uniform))
    this_set_end_time = time.time()
    hour, minu, secon = get_time(next_set_start_time, this_set_end_time)
    Log_file.info('##### (time) this epoch set: {} hours {} minutes {} seconds #####'.format(hour, minu, secon))
    hour, minu, secon = get_time(start_time, this_set_end_time)
    Log_file.info('##### (time) start until now: {} hours {} minutes {} seconds #####'.format(hour, minu, secon))
    next_set_start_time = time.time()
    return next_set_start_time

def report_time_eval(start_time, Log_file):
    end_time = time.time()
    hour, minu, secon = get_time(start_time, end_time)
    Log_file.info('##### (time) eval(valid and test): {} hours {} minutes {} seconds #####'.format(hour, minu, secon))

def save_model(now_epoch, model, model_dir, optimizer, rng_state, cuda_rng_state, Log_file):
    ckpt_path = os.path.join(model_dir, f'epoch-{now_epoch}.pt')
    torch.save({'model_state_dict': model.module.state_dict(),  # 模型
                'optimizer': optimizer.state_dict(),  # 优化器状态
                'rng_state': rng_state,   # torch的随机数状态
                'cuda_rng_state': cuda_rng_state}, ckpt_path)  # torch.cuda 的随机数状态
    Log_file.info(f'Model saved to {ckpt_path}')