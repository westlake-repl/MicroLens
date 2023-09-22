from cProfile import run
import logging
from logging import getLogger
import torch
from REC.data import *
from REC.config import Config
from REC.utils import init_logger, get_model, init_seed, set_color
from REC.trainer import Trainer
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import os
import random
import numpy as np
import argparse
import torch.distributed as dist
import torch
from REC.data import LMDB_Image


def run_loop(local_rank,config_file=None,saved=True):

    config = Config(config_file_list=[config_file])
   
 
    device = torch.device("cuda", local_rank)
    config['device'] = device
    

    init_seed(config['seed'], config['reproducibility'])
  
    init_logger(config)
    logger = getLogger()
                
    dataload = load_data(config)
    train_loader, valid_loader, test_loader = bulid_dataloader(config, dataload)  
           
    model = get_model(config['model'])(config, dataload)    
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device) 
    
    model = DDP(model, device_ids=[local_rank], output_device=local_rank,find_unused_parameters=True)
    trainer = Trainer(config,model)
    
    world_size = torch.distributed.get_world_size()
    logger.info(set_color('\nWorld_Size', 'pink') + f' = {world_size} \n')
    logger.info(config)                    
    logger.info(dataload)
    logger.info(model.module)
    
    init_seed(config['seed'], config['reproducibility'])

                    
    best_valid_score, best_valid_result = trainer.fit(
        train_loader, valid_loader, saved=saved, show_progress=False
    )

    test_result = trainer.evaluate(test_loader, load_best_model=saved, show_progress=False)

    logger.info(set_color('best valid ', 'yellow') + f': {best_valid_result}')
    logger.info(set_color('test result', 'yellow') + f': {test_result}')

    return {
        'best_valid_score': best_valid_score,
        'valid_score_bigger': config['valid_metric_bigger'],
        'best_valid_result': best_valid_result,
        'test_result': test_result
    }




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", default=-1, type=str)
    args = parser.parse_args()
    local_rank = int(os.environ['LOCAL_RANK'])
    config_file = args.config_file
    
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl')

    run_loop(local_rank = local_rank,config_file=config_file)
   



