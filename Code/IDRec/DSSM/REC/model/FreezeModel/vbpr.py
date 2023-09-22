import torch
from torch import nn
from REC.utils.enum_type import InputType
from REC.model.basemodel import BaseModel
import numpy as np
from torch.nn.init import xavier_normal_, constant_

class VBPR(BaseModel):
    input_type = InputType.PAIR
    def __init__(self, config, dataload):
        super(VBPR, self).__init__()
        self.mlp_hidden_size = config['mlp_hidden_size']
        self.dropout_prob = config['dropout_prob']
        self.embedding_size = config['embedding_size'] // 2
    
        self.device = config['device']
              
        self.user_num = dataload.user_num
        self.item_num = dataload.item_num
        
        self.v_feat_path = config['v_feat_path']
        v_feat = np.load(self.v_feat_path, allow_pickle=True)   #xxx.npy
        # print(v_feat.shape) 
        # item_ids = dataload.id2token['item_id']
        # new_feat = []
        # for idx, token in enumerate(item_ids):
        #     if idx == 0 :
        #         new_feat.append(np.zeros_like(v_feat[0]))
        #     else:
        #         new_feat.append(v_feat[int(token)-1])
        # new_feat = np.array(new_feat)
        # np.save('feature/remap_visual_features', new_feat)
        # print(new_feat.shape)
        # import sys
        # sys.exit()

        self.v_feat = torch.tensor(v_feat,dtype=torch.float).to(self.device)
        self.weight = torch.tensor([[1.0],[-1.0]]).to(self.device)

        self.feature_dim = self.v_feat.shape[-1]
                
        # define layers and loss
        self.feature_projection = nn.Linear(self.feature_dim, self.embedding_size, bias=False)
        self.bias_projection = nn.Linear(self.feature_dim, 1, bias=False) 
        self.user_id_embedding = nn.Embedding(self.user_num, self.embedding_size)
        self.item_id_embedding = nn.Embedding(self.item_num, self.embedding_size)
        
        self.user_modal_embedding = nn.Embedding(self.user_num, self.embedding_size)

        import os
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
   
 
    def forward(self, inputs):
        user, item = inputs
        embed_id_user = self.user_id_embedding(user)
        embed_id_item = self.item_id_embedding(item)  

        embed_modal_user = self.user_modal_embedding(user)
        embed_modal_item = self.feature_projection(self.v_feat[item])   


        embed_user = torch.cat((embed_id_user, embed_modal_user), dim=-1)
        embed_item = torch.cat((embed_id_item, embed_modal_item), dim=-1)
        logits = torch.matmul(embed_user, embed_item.t())  #[batch, batch]
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
        embed_id_user = self.user_id_embedding(user) #[eval_batch,,32]
        embed_id_item = self.item_id_embedding.weight #[num_item,32]

        embed_modal_user = self.user_modal_embedding(user)
       
        #[eval_batch, num_item]
        score = torch.matmul(embed_id_user,embed_id_item.t()) + \
        torch.matmul(embed_modal_user,item_feature.t()) + \
            self.total_visual_bias
            #self.global_bias + self.user_bias + self.item_bias 
        
        return score

    @torch.no_grad()    # [num_item, 32]
    def compute_item_all(self):
        embed = self.feature_projection(self.v_feat)
        self.total_visual_bias = self.bias_projection(self.v_feat).squeeze(-1)
        return embed




