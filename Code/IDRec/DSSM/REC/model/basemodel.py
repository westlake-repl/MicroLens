import numpy as np
import torch
import torch.nn as nn

from REC.utils import set_color




class BaseModel(nn.Module):
    
    def __init__(self):
        super(BaseModel, self).__init__()
        
    
    
    def load_weights(self, path):
        checkpoint = torch.load(path,map_location='cpu')
        pretrained_dicts = checkpoint['state_dict']
        self.load_state_dict({k.replace('item_embedding.rec_fc', 'visual_encoder.item_encoder.fc')
        :v for k,v in pretrained_dicts.items()}, strict=False) 

    


    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + set_color('\nTrainable parameters', 'blue') + f': {params}'



