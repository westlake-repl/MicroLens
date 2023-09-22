import copy
import pickle
import os
import yaml
from collections import Counter
from logging import getLogger

import numpy as np
import pandas as pd
import torch

from REC.utils import set_color, ensure_dir
from REC.utils.enum_type import InputType
from torch_geometric.utils import degree

class Data:
    def __init__(self, config):
        self.config = config
        self.dataset_path = config['data_path']
        self.dataset_name = config['dataset']
        self.logger = getLogger()
        self._from_scratch()

    def _from_scratch(self):
        self.logger.debug(set_color(f'Loading {self.__class__} from scratch.', 'green'))
        self._load_inter_feat(self.dataset_name, self.dataset_path)
        self._data_processing()
    
    
    def _load_inter_feat(self, token, dataset_path):    
        inter_feat_path = os.path.join(dataset_path, f'{token}.inter')
        if not os.path.isfile(inter_feat_path):
            raise ValueError(f'File {inter_feat_path} not exist.')
        
        df = pd.read_csv(
            inter_feat_path, delimiter=',', dtype={'item_id':str, 'user_id':str, 'timestamp':int}, header=0, names=['item_id', 'user_id', 'timestamp'] 
        )
                   
        self.logger.debug(f'Interaction feature loaded successfully from [{inter_feat_path}].')
        self.inter_feat = df

    def _data_processing(self):
        
        self.id2token = {}
        self.token2id = {}
        remap_list = ['user_id', 'item_id']
        for feature in remap_list:
            feats = self.inter_feat[feature]
            new_ids_list, mp = pd.factorize(feats)
            mp = np.array(['[PAD]'] + list(mp))
            token_id = {t: i for i, t in enumerate(mp)}
            self.id2token[feature] = mp
            self.token2id[feature] = token_id
            self.inter_feat[feature] = new_ids_list+1
        
        self.user_num = len(self.id2token['user_id'])
        self.item_num = len(self.id2token['item_id'])
        self.inter_num = len(self.inter_feat)
        self.uid_field = 'user_id'
        self.iid_field = 'item_id'
        self.user_seq = None
        self.train_feat = None
        self.feat_name_list = ['inter_feat']       #self.inter_feat

   
    def build(self):
        
        self.sort(by='timestamp')
        user_list = self.inter_feat['user_id'].values
        item_list = self.inter_feat['item_id'].values
        grouped_index = self._grouped_index(user_list)
        
        user_seq = {}
        for uid, index in grouped_index.items():
            user_seq[uid] = item_list[index]
    
        self.user_seq = user_seq
        train_feat = dict()
        indices = []
       
        for index in grouped_index.values():
            indices.extend(list(index)[:-2])
        for k in self.inter_feat:
            train_feat[k] = self.inter_feat[k].values[indices]        
        
        if self.config['MODEL_INPUT_TYPE'] == InputType.SEQ:
            if self.config['data_augmentation']:
                train_feat = self._build_aug_seq(train_feat)            
            else:
                train_feat = self._build_seq(train_feat)
            
        self.train_feat = train_feat
    

    def _grouped_index(self, group_by_list):
        index = {}
        for i, key in enumerate(group_by_list):
            if key not in index:
                index[key] = [i]
            else:
                index[key].append(i)
        return index
    
    def _build_seq(self, train_feat):
        max_item_list_len = self.config['MAX_ITEM_LIST_LENGTH']+1
        
        # by = ['user_id', 'timestamp']
        # ascending = [True, True]
        # for b, a in zip(by[::-1], ascending[::-1]):
        #     index = np.argsort(train_feat[b], kind='stable')
        #     if not a:
        #         index = index[::-1]
        #     for k in train_feat:
        #         train_feat[k] = train_feat[k][index]       
               
        uid_list, item_list_index= [], []
        seq_start = 0
        save = False
        user_list = train_feat['user_id']
        user_list = np.append(user_list, -1)
        last_uid = user_list[0]
        for i, uid in enumerate(user_list):
            if last_uid != uid :
                save = True 
            if save:
                if i - seq_start > max_item_list_len:
                    offset = (i - seq_start) % max_item_list_len
                    seq_start += offset
                    x = torch.arange(seq_start, i)
                    sx = torch.split(x, max_item_list_len)
                    for sub in sx:
                        uid_list.append(last_uid)
                        item_list_index.append(slice(sub[0],sub[-1]+1)) 
                                             
                        
                else:
                    uid_list.append(last_uid)
                    item_list_index.append(slice(seq_start,i))
            
                    
                save = False
                last_uid = uid
                seq_start = i
        
        seq_train_feat = {}
        seq_train_feat['user_id'] = np.array(uid_list)
        seq_train_feat['item_seq'] = []
        for index in item_list_index:
            seq_train_feat['item_seq'].append(train_feat['item_id'][index])

        return seq_train_feat

    def _build_aug_seq(self, train_feat):
        max_item_list_len = self.config['MAX_ITEM_LIST_LENGTH']+1
        
        # by = ['user_id', 'timestamp']
        # ascending = [True, True]
        # for b, a in zip(by[::-1], ascending[::-1]):
        #     index = np.argsort(train_feat[b], kind='stable')
        #     if not a:
        #         index = index[::-1]
        #     for k in train_feat:
        #         train_feat[k] = train_feat[k][index]       
               
        uid_list, item_list_index= [], []
        seq_start = 0
        save = False
        user_list = train_feat['user_id']
        user_list = np.append(user_list, -1)
        last_uid = user_list[0]
        for i, uid in enumerate(user_list):
            if last_uid != uid :
                save = True 
            if save:
                if i - seq_start > max_item_list_len:
                    offset = (i - seq_start) % max_item_list_len
                    seq_start += offset
                    x = torch.arange(seq_start, i)
                    sx = torch.split(x, max_item_list_len)
                    for sub in sx:
                        uid_list.append(last_uid)
                        item_list_index.append(slice(sub[0],sub[-1]+1))                         
                else:
                    uid_list.append(last_uid)
                    item_list_index.append(slice(seq_start,i))                    
                save = False
                last_uid = uid
                seq_start = i
        
        seq_train_feat = {}
        aug_uid_list = []
        aug_item_list = []
        for uid, item_index in zip(uid_list, item_list_index):
            st = item_index.start
            ed = item_index.stop
            lens = ed - st 
            for sub_idx in range(1,lens):
                aug_item_list.append(train_feat['item_id'][slice(st, st+sub_idx+1 )])
                aug_uid_list.append(uid)

        seq_train_feat['user_id'] = np.array(aug_uid_list)
        seq_train_feat['item_seq'] = aug_item_list

        return seq_train_feat




    def sort(self, by, ascending=True):

        if isinstance(self.inter_feat, pd.DataFrame):
            self.inter_feat.sort_values(by=by, ascending=ascending, inplace=True)
        
        else:
            if isinstance(by, str):
                by = [by]

            if isinstance(ascending, bool):
                ascending = [ascending] 

            if len(by) != len(ascending):
                if len(ascending) == 1:
                    ascending = ascending * len(by)
                else:
                    raise ValueError(f'by [{by}] and ascending [{ascending}] should have same length.')
            for b, a in zip(by[::-1], ascending[::-1]):
                index = np.argsort(self.inter_feat[b], kind='stable')
                if not a:
                    index = index[::-1]
                for k in self.inter_feat:
                    self.inter_feat[k] = self.inter_feat[k][index]    
    
    
    def save(self):
        """Saving this :class:`Dataset` object to :attr:`config['checkpoint_dir']`.
        """
        save_dir = self.config['checkpoint_dir']
        ensure_dir(save_dir)
        file = os.path.join(save_dir, f'{self.config["dataset"]}-dataset.pth')
        self.logger.info(set_color('Saving filtered dataset into ', 'pink') + f'[{file}]')
        with open(file, 'wb') as f:
            pickle.dump(self, f)

    
    @property
    def avg_actions_of_users(self):
        """Get the average number of users' interaction records.

        Returns:
            numpy.float64: Average number of users' interaction records.
        """
        if isinstance(self.inter_feat, pd.DataFrame):
            return np.mean(self.inter_feat.groupby(self.uid_field).size())
        else:
            return np.mean(list(Counter(self.inter_feat[self.uid_field]).values()))

    @property
    def avg_actions_of_items(self):
        """Get the average number of items' interaction records.

        Returns:
            numpy.float64: Average number of items' interaction records.
        """
        if isinstance(self.inter_feat, pd.DataFrame):
            return np.mean(self.inter_feat.groupby(self.iid_field).size())
        else:
            return np.mean(list(Counter(self.inter_feat[self.iid_field]).values()))

    @property
    def sparsity(self):
        """Get the sparsity of this dataset.

        Returns:
            float: Sparsity of this dataset.
        """
        return 1 - self.inter_num / self.user_num / self.item_num

    
    def __repr__(self):
        return self.__str__()

    def __str__(self):
        info = [set_color(self.dataset_name, 'pink')]
        if self.uid_field:
            info.extend([
                set_color('The number of users', 'blue') + f': {self.user_num}',
                set_color('Average actions of users', 'blue') + f': {self.avg_actions_of_users}'
            ])
        if self.iid_field:
            info.extend([
                set_color('The number of items', 'blue') + f': {self.item_num}',
                set_color('Average actions of items', 'blue') + f': {self.avg_actions_of_items}'
            ])
        info.append(set_color('The number of inters', 'blue') + f': {self.inter_num}')
        if self.uid_field and self.iid_field:
            info.append(set_color('The sparsity of the dataset', 'blue') + f': {self.sparsity * 100}%')
     
        return '\n'.join(info)

    def copy(self, new_inter_feat):
        """Given a new interaction feature, return a new :class:`Dataset` object,
        whose interaction feature is updated with ``new_inter_feat``, and all the other attributes the same.

        Args:
            new_inter_feat (Interaction): The new interaction feature need to be updated.

        Returns:
            :class:`~Dataset`: the new :class:`~Dataset` object, whose interaction feature has been updated.
        """
        nxt = copy.copy(self)
        nxt.inter_feat = new_inter_feat
        return nxt


    def counter(self, field):
        if isinstance(self.inter_feat, pd.DataFrame):
            return Counter(self.inter_feat[field].values)
        else:
            return Counter(self.inter_feat[field])
  

    @property
    def user_counter(self):
        return self.counter('user_id')

    @property
    def item_counter(self):
        return self.counter('item_id')

    def get_norm_adj_mat(self):
        r"""Get the normalized interaction matrix of users and items.
        Construct the square matrix from the training data and normalize it
        using the laplace matrix.
        .. math::
            A_{hat} = D^{-0.5} \times A \times D^{-0.5}
        Returns:
            The normalized interaction matrix in Tensor.
        """

        row = torch.tensor(self.train_feat[self.uid_field])
        col = torch.tensor(self.train_feat[self.iid_field]) + self.user_num
        edge_index1 = torch.stack([row, col])
        edge_index2 = torch.stack([col, row])
        edge_index = torch.cat([edge_index1, edge_index2], dim=1)

        deg = degree(edge_index[0], self.user_num + self.item_num)

        norm_deg = 1. / torch.sqrt(torch.where(deg == 0, torch.ones([1]), deg))
        edge_weight = norm_deg[edge_index[0]] * norm_deg[edge_index[1]]

        return edge_index, edge_weight

    