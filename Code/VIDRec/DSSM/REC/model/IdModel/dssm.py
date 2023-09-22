import torch
import torch.nn as nn
from torch.nn.init import xavier_normal_, constant_
from REC.model.layers import MLPLayers
from REC.utils import InputType
from REC.model.basemodel import BaseModel
import os
import numpy as np


class DSSM(BaseModel):

    input_type = InputType.PAIR
    
    def __init__(self, config, data):
        super(DSSM, self).__init__()
        self.mlp_hidden_size = config['mlp_hidden_size']
        self.dropout_prob = config['dropout_prob']
        self.embedding_size = config['embedding_size']
        self.out_size = self.mlp_hidden_size[-1] if len(self.mlp_hidden_size) else self.embedding_size

        self.device = config['device']
              
        self.user_num = data.user_num
        self.item_num = data.item_num
        

        user_size_list = [self.embedding_size] + self.mlp_hidden_size
        item_size_list = [self.embedding_size] + self.mlp_hidden_size

        # define layers and loss
        self.user_mlp_layers = MLPLayers(user_size_list, self.dropout_prob, activation='tanh', bn=True)
        self.item_mlp_layers = MLPLayers(item_size_list, self.dropout_prob, activation='tanh', bn=True)

        self.user_embedding = nn.Embedding(self.user_num, self.embedding_size)
        self.item_embedding = nn.Embedding(self.item_num, self.embedding_size)
        

        self.device = config['device']
        path = os.path.join(config['data_path'], 'pop.npy')
        pop_prob_list = np.load(path)
        self.pop_prob_list = torch.FloatTensor(pop_prob_list).to(self.device)
        self.loss_func = nn.CrossEntropyLoss() 
       
        self.apply(self._init_weights)
     

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight.data)
        elif isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)
            if module.bias is not None:
                constant_(module.bias.data, 0)
   
 
    def forward(self, input):
        user, item = input
        embed_user = self.user_embedding(user)
        embed_item = self.item_embedding(item)
        user_dnn_out = self.user_mlp_layers(embed_user)    #[batch, dim]
        item_dnn_out = self.item_mlp_layers(embed_item)    #[batch, dim]
  
        logits = torch.matmul(user_dnn_out, item_dnn_out.t())  #[batch, batch]
        label = torch.arange(item.numel()).to(self.device)
        
        flatten_item_seq = item.view(-1)

        debias_logits = torch.log(self.pop_prob_list[flatten_item_seq])
        logits = logits - debias_logits

        history = flatten_item_seq.unsqueeze(-1).expand(-1, len(flatten_item_seq))
        history_item_mask = (history == flatten_item_seq)
        unused_item_mask = torch.scatter(history_item_mask, 1, label.view(-1, 1), False)
        logits[unused_item_mask] = -1e8
        loss = self.loss_func(logits, label)
        return loss

      

        
        
    
    @torch.no_grad()
    def predict(self, user,item_feature):
    
        user_feature = self.user_embedding(user)
        user_dnn_out = self.user_mlp_layers(user_feature)
       
        scores = torch.matmul(user_dnn_out,item_feature.t())
        return scores

    @torch.no_grad()    # [num_item, 64]
    def compute_item_all(self):
        embed_item = self.item_embedding.weight
        return self.item_mlp_layers(embed_item)





