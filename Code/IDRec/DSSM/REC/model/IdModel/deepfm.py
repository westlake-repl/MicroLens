import torch
import torch.nn as nn
from torch.nn.init import xavier_normal_, constant_
from REC.model.layers import MLPLayers, BaseFactorizationMachine
from REC.utils import InputType
from REC.model.basemodel import BaseModel
from logging import getLogger
import os
import numpy as np

class DeepFM(BaseModel):
    input_type = InputType.SEQ
    def __init__(self, config, dataload):
        super(DeepFM, self).__init__()

        # load parameters info
        self.mlp_hidden_size = config['mlp_hidden_size']
        self.dropout_prob = config['dropout_prob']
        self.embedding_size = config['embedding_size']
        self.out_size = self.mlp_hidden_size[-1] if len(self.mlp_hidden_size) else self.embedding_size

        self.device = config['device']
        self.max_seq_length = config['MAX_ITEM_LIST_LENGTH']

        self.item_num = dataload.item_num        
        self.item_embedding = nn.Embedding(self.item_num, self.embedding_size, padding_idx=0)
        self.fm = BaseFactorizationMachine(reduce_sum=True)

        self.dense1 = nn.Linear(self.embedding_size* (self.max_seq_length+1), self.embedding_size, bias=False)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(self.embedding_size, 1)

        self.eval_user_dense1 = None
        self.eval_item_dense1 = None

        self.device = config['device']
        path = os.path.join(config['data_path'], 'pop.npy')
        pop_prob_list = np.load(path)
        self.pop_prob_list = torch.FloatTensor(pop_prob_list).to(self.device)
        self.loss_func = nn.CrossEntropyLoss()
        # parameters initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight.data)
        elif isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)
            if module.bias is not None:
                constant_(module.bias.data, 0)

    
    
    def mask_emb(self, user_seq):
        mask = user_seq != 0  
        mask = mask.float()
        
        token_seq_embedding = self.item_embedding(user_seq)  
        mask = mask.unsqueeze(-1).expand_as(token_seq_embedding)
        masked_token_seq_embedding = token_seq_embedding * mask
 
        return masked_token_seq_embedding

    
    
    def forward(self, inputs):              
       
        input_seq, targets = inputs
        bs, seq_len = input_seq.shape
        all_items = torch.cat((input_seq, targets.unsqueeze(-1)), dim=-1)
        all_items_flat = all_items.view(-1).unsqueeze(-1)
        batch_input_seq = torch.cat(((input_seq).repeat_interleave(all_items_flat.shape[0], 0), all_items_flat.repeat(input_seq.shape[0], 1)),dim=-1)
        inputs_embedding = self.mask_emb(batch_input_seq)  
        
        y_fm = self.fm(inputs_embedding).view(bs, -1)   
        y_deep = self.dense2(self.relu(self.dense1(inputs_embedding.flatten(1,2)))).view(bs, -1)
        
        logits = y_fm + y_deep
        label = torch.arange(all_items.numel()).reshape(bs, -1)
        label = label[:, -1].to(self.device).view(-1)
        
        flatten_item_seq = all_items.view(-1)
        debias_logits = torch.log(self.pop_prob_list[flatten_item_seq])
        logits = logits - debias_logits
        
        user_history = torch.zeros(bs, seq_len + 2).type_as(flatten_item_seq)
        user_history[:, :-1] = all_items
        user_history = user_history.unsqueeze(-1).expand(-1, -1, len(flatten_item_seq))
        history_item_mask = (user_history == flatten_item_seq).any(dim=1)
        unused_item_mask = torch.scatter(history_item_mask, 1, label.view(-1, 1), False)
        logits[unused_item_mask] = -1e8
        loss = self.loss_func(logits, label)
        return loss
   

   

    @torch.no_grad()
    def predict(self,user_seq,item_feature):                                                
        user_embedding = self.mask_emb(user_seq)   
        user_avg_embedding = torch.sum(user_embedding, dim=1)
        y_fm = torch.matmul(user_avg_embedding,item_feature.t())
        u_out = torch.matmul(user_embedding.flatten(1,2), self.eval_user_dense1.t()) 
        output1 = self.relu(u_out.unsqueeze(1) + item_feature.unsqueeze(0))
        y_deep = self.dense2(output1).squeeze(-1)
        scores = y_fm + y_deep
        return scores

    @torch.no_grad()    
    def compute_item_all(self):
        dense1_weights = self.dense1.weight
        uweights = dense1_weights[:, :self.embedding_size*self.max_seq_length]
        iweights = dense1_weights[:, self.embedding_size*self.max_seq_length:]
        self.eval_user_dense1 = uweights

        return torch.matmul(self.item_embedding.weight, iweights.t()) 
 
    




