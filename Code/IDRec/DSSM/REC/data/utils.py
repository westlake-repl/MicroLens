import copy
import importlib
import os
import pickle
from logging import getLogger
from REC.data.dataset import *
from REC.utils import set_color
from functools import partial
from .data import Data
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random

def load_data(config):
    '''
    load data
    '''
  
    file = config['data_save_path']
    if file and os.path.exists(file):
        with open(file, 'rb') as f:
            dataload = pickle.load(f)

        logger = getLogger()
        logger.info(set_color('Load data from', 'pink') + f': [{file}]')
        return dataload

    dataload = Data(config)
    if config['save_data']:
        dataload.save()
    return dataload


def bulid_dataloader(config,dataload, use_DDP=True):
    '''
    split dataset, generate user history sequence, train/valid/test dataset
    '''
    dataset_dict = {

        'DSSM': ('PairTrainDataset', 'PairEvalDataset', 'pair_eval_collate'),        
        'VBPR': ('PairTrainDataset', 'PairEvalDataset', 'pair_eval_collate'),    
      'LightGCN': ('PairTrainDataset', 'PairEvalDataset', 'pair_eval_collate'),  
        'VidYTDNN': ('vidSampleTwoTowerTrainDataset', 'SeqEvalDataset', 'seq_eval_collate'),
        'NFM': ('SampleOneTowerTrainDataset', 'SeqEvalDataset', 'seq_eval_collate'),
        'DeepFM': ('SampleOneTowerTrainDataset', 'SeqEvalDataset', 'seq_eval_collate'),
    }
 

    model_name = config['model']
    dataload.build()
       
    dataset_module = importlib.import_module('REC.data.dataset')
    train_set_name, test_set_name, collate_fn_name = dataset_dict[model_name]
     
    if isinstance(train_set_name, tuple):
        train_set_class = getattr(dataset_module, train_set_name[0])
        train_collate_fn = getattr(dataset_module, train_set_name[1]) 
    else:
        train_set_class = getattr(dataset_module, train_set_name) 
        train_collate_fn = None        
    
    test_set_class = getattr(dataset_module, test_set_name)
    eval_collate_fn = getattr(dataset_module, collate_fn_name)
 

    train_data = train_set_class(config,dataload)
    valid_data = test_set_class(config, dataload, phase='valid')
    test_data = test_set_class(config, dataload, phase='test')
    
        
    logger = getLogger()
    logger.info(
        set_color('[Training]: ', 'pink') + set_color('train_batch_size', 'cyan') + ' = ' +
        set_color(f'[{config["train_batch_size"]}]', 'yellow') 
    )
    logger.info(
        set_color('[Evaluation]: ', 'pink') + set_color('eval_batch_size', 'cyan') + ' = ' +
        set_color(f'[{config["eval_batch_size"]}]', 'yellow') 
    )

    if use_DDP:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
        valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_data)
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_data)
    else:
        train_sampler = None
        valid_sampler = None
        test_sampler = None
    # valid_sampler = SequentialDistributedSampler(valid_data,config['eval_batch_size'])
    # test_sampler = SequentialDistributedSampler(test_data,config['eval_batch_size'])  
   
    num_workers = 10
    
    if use_DDP:
        rank = torch.distributed.get_rank() 
    else:
        rank = 0
    
    seed = torch.initial_seed()
    
    init_fn = partial( 
    worker_init_fn, num_workers=num_workers, rank=rank, 
    seed=seed)
    
    
    if train_collate_fn:
        train_loader = DataLoader(train_data, batch_size=config['train_batch_size'], num_workers=num_workers,
                          pin_memory=True, sampler=train_sampler, collate_fn = train_collate_fn , worker_init_fn=init_fn)
    else:
        train_loader = DataLoader(train_data, batch_size=config['train_batch_size'], num_workers=num_workers,
                          pin_memory=True, sampler=train_sampler, worker_init_fn=init_fn)
    valid_loader = DataLoader(valid_data, batch_size=config['eval_batch_size'], num_workers=num_workers,
                          pin_memory=True, sampler=valid_sampler, collate_fn=eval_collate_fn)

    test_loader = DataLoader(test_data, batch_size=config['eval_batch_size'], num_workers=num_workers,
                          pin_memory=True, sampler=test_sampler,collate_fn=eval_collate_fn)

    return train_loader, valid_loader, test_loader



def worker_init_fn(worker_id, num_workers, rank, seed): 
    # The seed of each worker equals to 
    # num_worker * rank + worker_id + user_seed 
    worker_seed = num_workers * rank + worker_id + seed 
    np.random.seed(worker_seed) 
    random.seed(worker_seed)



def worker_init_reset_seed(worker_id):
    initial_seed = torch.initial_seed() % 2 ** 31
    worker_seed = initial_seed + worker_id + torch.distributed.get_rank()
    random.seed(worker_seed)
    np.random.seed(worker_seed)




class SequentialDistributedSampler(torch.utils.data.sampler.Sampler):

    def __init__(self, dataset, batch_size, rank=None, num_replicas=None):
        if num_replicas is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = torch.distributed.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.batch_size = batch_size
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.batch_size / self.num_replicas)) * self.batch_size
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        # add extra samples to make it evenly divisible
        indices += [indices[-1]] * (self.total_size - len(indices))
        # subsample
        indices = indices[self.rank * self.num_samples : (self.rank + 1) * self.num_samples]
        return iter(indices)

    def __len__(self):
        return self.num_samples

class LMDB_Image:
    def __init__(self, image, id):
        self.image = image.tobytes()

class LMDB_Image1:
    def __init__(self, image, id):
        self.channels = image.shape[2]
        self.size = image.shape[:2]
        self.image = image.tobytes()
        self.id = id

    def get_image(self):
        image = np.frombuffer(self.image, dtype=np.uint8)
        return image.reshape(*self.size, self.channels)


class LMDB_VIDEO:
    def __init__(self, video):
        self.video = video.tobytes()
