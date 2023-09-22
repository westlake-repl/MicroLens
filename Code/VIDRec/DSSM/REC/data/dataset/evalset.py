import torch
from torch.utils.data import Dataset
import numpy as np
class SeqEvalDataset(Dataset):
    def __init__(self, config, dataload, phase='valid'):
        self.dataload = dataload
        self.max_item_list_length = config['MAX_ITEM_LIST_LENGTH']
        self.user_seq = list(dataload.user_seq.values())
        self.phase = phase
        self.length = len(self.user_seq)
        self.item_num = dataload.item_num
        

    def __len__(self):
        return self.length

    def _padding_sequence(self, sequence, max_length):
        sequence = list(sequence)
        pad_len = max_length - len(sequence)
        sequence = [0] * pad_len + sequence 
        sequence = sequence[-max_length:]  # truncate according to the max_length
        return sequence
    
    def __getitem__(self,index):               
        if self.phase == 'valid':
            history_seq = self.user_seq[index][:-2]
            #item_length = min(len(history_seq), self.max_item_list_length)
            item_seq = self._padding_sequence(history_seq, self.max_item_list_length)
            item_target = self.user_seq[index][-2]            
        else:
            history_seq = self.user_seq[index][:-1]
            #item_length = min(len(history_seq), self.max_item_list_length)
            item_seq = self._padding_sequence(history_seq, self.max_item_list_length)
            item_target = self.user_seq[index][-1]
                               
        return torch.tensor(history_seq), item_seq, item_target  #, item_length




class PairEvalDataset(Dataset):
    def __init__(self, config, dataload, phase='valid'):

        self.dataload = dataload
        self.user_seq = dataload.user_seq
        self.user_list = list(dataload.user_seq.keys())
        self.length = len(self.user_seq)
        self.item_num = dataload.item_num
        self.phase = phase
        

    def __len__(self):
        return self.length
    
    def __getitem__(self,index):
        user_id = self.user_list[index]       

        if self.phase == 'valid':
            history_i = self.user_seq[user_id][:-2]
            positive_i = self.user_seq[user_id][-2]
            
        else:
            history_i = self.user_seq[user_id][:-1]
            positive_i = self.user_seq[user_id][-1]
                
        return torch.tensor(user_id), torch.tensor(history_i), torch.tensor([positive_i])





class CandiEvalDataset(Dataset):
    def __init__(self, config, dataload, phase='valid'):
        self.dataload = dataload
        self.max_item_list_length = config['MAX_ITEM_LIST_LENGTH']
        self.user_seq = list(dataload.user_seq.values())
        self.phase = phase
        self.length = len(self.user_seq)
        self.item_num = dataload.item_num
        self.item_token = torch.arange(self.item_num).unsqueeze(1)
        

    def __len__(self):
        return self.length

    def _padding_sequence(self, sequence, max_length):
        sequence = list(sequence)
        pad_len = max_length - len(sequence)
        sequence = [0] * pad_len + sequence 
        sequence = sequence[-max_length:]  # truncate according to the max_length
        return torch.tensor(sequence)


    def __getitem__(self,index): 
        if self.phase == 'valid':
            history_seq = self.user_seq[index][:-2]
            item_seq = self._padding_sequence(history_seq, self.max_item_list_length)
            item_target = self.user_seq[index][-2]            
        else:
            history_seq = self.user_seq[index][:-1]
            item_seq = self._padding_sequence(history_seq, self.max_item_list_length)
            item_target = self.user_seq[index][-1] 

        item_seq = item_seq.unsqueeze(0)
        item_seq = item_seq.repeat_interleave(self.item_num,0)   #[n_items, seq_len]
        
        item_seq = torch.cat((item_seq, self.item_token),dim=-1)  #[n_items, seq_len+1]  

        return torch.tensor(history_seq),item_seq, item_target



class VisRankEvalDataset(Dataset):
    def __init__(self, config, dataload, phase='valid'):
        self.dataload = dataload
        self.max_item_list_length = config['MAX_ITEM_LIST_LENGTH']
        self.user_seq = list(dataload.user_seq.values())
        self.phase = phase
        self.length = len(self.user_seq)
        self.item_num = dataload.item_num
        

    def __len__(self):
        return self.length

    def _padding_sequence(self, sequence, max_length):
        sequence = list(sequence)
        pad_len = max_length - len(sequence)
        sequence = [0] * pad_len + sequence 
        sequence = sequence[-max_length:]  # truncate according to the max_length
        return sequence
    
    def __getitem__(self,index): 
        #item_seq = self._padding_sequence(history_seq, self.max_item_list_length)              
        if self.phase == 'valid':
            history_seq = self.user_seq[index][:-2]            
            item_target = self.user_seq[index][-2]            
        else:
            history_seq = self.user_seq[index][:-1]
            item_target = self.user_seq[index][-1]
                               
        user = torch.zeros(1,dtype=torch.long)
        return torch.tensor(history_seq), \
               (user, history_seq), \
                user, torch.tensor(item_target)


class ACFEvalDataset(Dataset):
    def __init__(self, config, dataload, phase='valid'):
        self.dataload = dataload
        self.max_item_list_length = config['MAX_ITEM_LIST_LENGTH']+1  #add user_id in the end
        self.user_seq = list(dataload.user_seq.values())
        self.user_ids = list(dataload.user_seq.keys())
        self.phase = phase
        self.length = len(self.user_seq)
        self.item_num = dataload.item_num
        

    def __len__(self):
        return self.length

    def _padding_sequence(self, sequence, max_length):
        pad_len = max_length - len(sequence)
        sequence = [0] * pad_len + sequence 
        sequence = sequence[-max_length:]  # truncate according to the max_length
        return sequence
    
    def __getitem__(self,index): 
        user_id = self.user_ids[index]              
        if self.phase == 'valid':
            history_seq = self.user_seq[index][:-2]
            item_seq = list(history_seq) + [user_id]
            #item_length = min(len(history_seq), self.max_item_list_length)
            item_seq = self._padding_sequence(item_seq, self.max_item_list_length)
            item_target = self.user_seq[index][-2]            
        else:
            history_seq = self.user_seq[index][:-1]
            item_seq = list(history_seq) + [user_id]
            #item_length = min(len(history_seq), self.max_item_list_length)
            item_seq = self._padding_sequence(item_seq, self.max_item_list_length)
            item_target = self.user_seq[index][-1]
                               
        return torch.tensor(history_seq), item_seq, item_target  #, item_length
   


class GraphEvalDataset(Dataset):
    def __init__(self,config,dataload, phase='valid'):
        self.dataload = dataload
        self.max_item_list_length = config['MAX_ITEM_LIST_LENGTH']
        self.user_seq = list(dataload.user_seq.values())
        self.phase = phase
        self.length = len(self.user_seq)
        self.item_num = dataload.item_num        

           
    def __len__(self):
        return self.length
    
    def _padding_sequence(self, sequence, max_length):
        sequence = list(sequence)
        pad_len = max_length - len(sequence)
        sequence = sequence + [0] * pad_len 
        sequence = sequence[-max_length:]  # truncate according to the max_length
        return np.array(sequence, dtype=np.int)

    
    def __getitem__(self, index): 
        if self.phase == 'valid':
            history_seq = self.user_seq[index][:-2]
            item_length = min(len(history_seq), self.max_item_list_length)
            masked_index = [1]*item_length
            item_seq = self._padding_sequence(history_seq, self.max_item_list_length)
            masked_index = self._padding_sequence(masked_index, self.max_item_list_length)  
            item_target = self.user_seq[index][-2] 
        else:
            history_seq = self.user_seq[index][:-1]
            item_length = min(len(history_seq), self.max_item_list_length)
            masked_index = [1]*item_length
            item_seq = self._padding_sequence(history_seq, self.max_item_list_length)
            masked_index = self._padding_sequence(masked_index, self.max_item_list_length)  
            item_target = self.user_seq[index][-1]
     
        return torch.tensor(history_seq), item_seq, masked_index, item_target 