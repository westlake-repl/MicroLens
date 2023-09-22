from torch.utils.data import Dataset
import torch
import numpy as np
import pandas as pd
from PIL import Image
import torchvision.transforms as transforms
import torchvision
import lmdb
import pickle
import random
import math
import os


Image_Mean = [0.5,  0.5,  0.5]
Image_Std = [0.5, 0.5,  0.5]
Resize = 224


class SEQTrainDataset(Dataset):
    def __init__(self,config,dataload):
        self.dataload = dataload
        self.config = config

        self.item_num = dataload.item_num 
        self.train_seq = dataload.train_feat['item_seq'] 
        
        self.length = len(self.train_seq)        
        
        self.max_seq_length = config['MAX_ITEM_LIST_LENGTH']+1
        self.device = config['device']        

           
    def __len__(self):
        return self.length
    

    def _neg_sample(self, item_set):
        item = random.randint(1, self.item_num - 1)
        while item in item_set:
            item = random.randint(1, self.item_num - 1)
        return item

    def _padding_sequence(self, sequence, max_length):
        pad_len = max_length - len(sequence)
        sequence = [0] * pad_len + sequence 
        sequence = sequence[-max_length:]  # truncate according to the max_length
        return torch.tensor(sequence, dtype=torch.long)

    
    def __getitem__(self, index): 
        item_seq = self.train_seq[index]      
        item_seq = self._padding_sequence(list(item_seq), self.max_seq_length)
        return item_seq 




class PairTrainDataset(Dataset):
    def __init__(self,config,dataload):    
        self.dataload = dataload        
        self.user_seq = dataload.user_seq
        self.item_num = dataload.item_num 
        self.train_uid = dataload.train_feat['user_id']
        self.train_iid = dataload.train_feat['item_id']
        self.length = len(self.train_uid)        
        
        self.device = config['device']       

           
    def __len__(self):
        return self.length
    
   
    def __getitem__(self, index): 
        user = self.train_uid[index]
        item_i = self.train_iid[index]

        item = torch.tensor(item_i)
        user = torch.tensor(user)
        return user, item
 



#数据形式为 [user_seq, pos_item, neg_item]
class TwoTowerTrainDataset(Dataset):
    def __init__(self,config,dataload):
        self.dataload = dataload
        self.config = config

        self.item_num = dataload.item_num 
        self.train_seq = dataload.train_feat['item_seq'] 
        self.length = len(self.train_seq)        
        
        self.max_seq_length = config['MAX_ITEM_LIST_LENGTH']+2
        self.device = config['device']         

           
    def __len__(self):
        return self.length
    

    def _neg_sample(self, item_set):
        item = random.randint(1, self.item_num - 1)
        while item in item_set:
            item = random.randint(1, self.item_num - 1)
        return item

    def _padding_sequence(self, sequence, max_length):
        pad_len = max_length - len(sequence)
        sequence = [0] * pad_len + sequence 
        sequence = sequence[-max_length:]  # truncate according to the max_length
        return torch.tensor(sequence, dtype=torch.long)
   
    def __getitem__(self, index): 
        item_seq = list(self.train_seq[index])  
        neg_item = self._neg_sample(item_seq)
        item_seq +=[neg_item]
        items = self._padding_sequence(item_seq, self.max_seq_length)     
        return items  




class vidSampleTwoTowerTrainDataset(Dataset):
    def __init__(self,config,dataload):
        self.dataload = dataload
        self.config = config

        self.item_num = dataload.item_num 
        self.iter_num = dataload.inter_num
       
        train_seq_list = dataload.train_feat['item_seq'] 
        train_aug_seq = []
        for train_seq in train_seq_list:
            train_seq = list(train_seq)
            for idx, item in enumerate(train_seq):
                item_list = train_seq[:idx] + train_seq[idx+1:] + [item] 
                train_aug_seq.append(item_list)
        
        self.train_seq = train_aug_seq
        self.length = len(self.train_seq)        
        
        self.max_seq_length = config['MAX_ITEM_LIST_LENGTH']+1
        self.device = config['device']         
        self.transform = transforms.Compose([
            torchvision.transforms.Resize((self.resize, self.resize)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        self.video_db_path = config['video_path']
        self.id2token = dataload.id2token['item_id']
        self.pad_video = np.array(5,3,224,224)
           
    def __len__(self):
        return self.length
    


    def _padding_sequence(self, sequence, max_length):
        pad_len = max_length - len(sequence)
        sequence = [0] * pad_len + sequence 
        sequence = sequence[-max_length:]  # truncate according to the max_length
        return torch.tensor(sequence, dtype=torch.long)
   
    def __getitem__(self, index):
        train_seq = list(self.train_seq[index])
        items_pad = self._padding_sequence(train_seq, self.max_seq_length)
        #sample_items_video = np.zeros((self.max_seq_length, 5, 3, 224, 224))
        env = lmdb.open(self.video_db_path, subdir=os.path.isdir(self.video_db_path),
                        readonly=True, lock=False, readahead=False, meminit=False)
        item_seq_token = self.id2token[items_pad]
        PAD_token = self.id2token[0]
        items_modal = []  
        with env.begin() as txn:
            for item in item_seq_token:
                if item == PAD_token:
                    VIDEO = self.pad_video                    
                else:
                    VIDEO = pickle.loads(txn.get(item.encode()))
                    VIDEO = np.copy(np.frombuffer(VIDEO.video, dtype=np.float32)).reshape(5, 3, 224, 224)                         
                     
                items_modal.append(torch.from_numpy(VIDEO)) 
                                                
        items_modal = torch.stack(items_modal)  #[max_len, 5, 3, 224, 224]  
    
        return items_pad, torch.FloatTensor(items_modal)

#数据形式为 [[pos_user_seq], [neg_user_seq]]
class OneTowerTrainDataset(Dataset):
    def __init__(self,config,dataload):
        self.dataload = dataload
        self.item_num = dataload.item_num 
        self.train_seq = dataload.train_feat['item_seq'] 
        self.length = len(self.train_seq)        
        
        self.max_seq_length = config['MAX_ITEM_LIST_LENGTH']+1
        self.device = config['device']       

           
    def __len__(self):
        return self.length
    
    def _neg_sample(self, item_set):
        item = random.randint(1, self.item_num - 1)
        while item in item_set:
            item = random.randint(1, self.item_num - 1)
        return item

    def _padding_sequence(self, sequence, max_length):
        pad_len = max_length - len(sequence)
        sequence = [0] * pad_len + sequence
        sequence = sequence[-max_length:]  # truncate according to the max_length
        return torch.tensor(sequence, dtype=torch.long)

    
    def __getitem__(self, index): 
        item_seq = list(self.train_seq[index])
        item_seq = self._padding_sequence(item_seq, self.max_seq_length)
        neg_item = item_seq.clone()
        neg_item[-1] = self._neg_sample(item_seq)
        items = torch.stack((item_seq,neg_item))     
        return items



class SampleOneTowerTrainDataset(Dataset):
    def __init__(self,config,dataload):
        self.dataload = dataload
        self.item_num = dataload.item_num 
        train_seq_list = dataload.train_feat['item_seq'] 
        train_aug_seq = []
        for train_seq in train_seq_list:
            train_seq = list(train_seq)
            for idx, item in enumerate(train_seq):
                item_list = train_seq[:idx] + train_seq[idx+1:] + [item] 
                train_aug_seq.append(item_list)
                
        self.train_seq = train_aug_seq    
        self.length = len(self.train_seq)        
        
        self.max_seq_length = config['MAX_ITEM_LIST_LENGTH']
        self.device = config['device']       

           
    def __len__(self):
        return self.length
    
    def _neg_sample(self, item_set):
        item = random.randint(1, self.item_num - 1)
        while item in item_set:
            item = random.randint(1, self.item_num - 1)
        return item

    def _padding_sequence(self, sequence, max_length):
        pad_len = max_length - len(sequence)
        sequence = [0] * pad_len + sequence 
        sequence = sequence[-max_length:]  # truncate according to the max_length
        return torch.tensor(sequence, dtype=torch.long)

    
    def __getitem__(self, index):
        item_list = list(self.train_seq[index])
        input_item = item_list[:-1]
        target = item_list[-1]
        pos_pad = self._padding_sequence(input_item, self.max_seq_length)
         
        return pos_pad, torch.tensor(target)




class BaseDataset(Dataset):
    def __init__(self,config,dataload):
        pass
                
    def __len__(self):
        return 0
    













