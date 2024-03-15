import torch
import torch.nn as nn

from .modules import TransformerEncoder
from torch.nn.init import xavier_normal_, constant_

class VitEncoder(torch.nn.Module):
    def __init__(self, image_net, args):
        super(VitEncoder, self).__init__()
        self.emb_size = args.embedding_dim
        self.word_emb = args.word_embedding_dim

        self.image_net = image_net
        self.activate = nn.ReLU() 

        self.image_proj = nn.Linear(self.word_emb, self.emb_size)
        xavier_normal_(self.image_proj.weight.data)
        if self.image_proj.bias is not None:
            constant_(self.image_proj.bias.data, 0)

    def forward(self, image):
        return self.image_proj(self.image_net(image)[0][:,0]) # get cls

class MaeEncoder(torch.nn.Module):
    def __init__(self, image_net, args):
        super(MaeEncoder, self).__init__()
        self.emb_size = args.embedding_dim
        self.word_emb = args.word_embedding_dim

        self.image_net = image_net
        # self.activate = nn.ReLU()

        self.image_proj = nn.Linear(self.word_emb, self.emb_size)
        xavier_normal_(self.image_proj.weight.data)
        if self.image_proj.bias is not None:
            constant_(self.image_proj.bias.data, 0)

        # self.dp = nn.Dropout(args.cv_dp) # help address overfitting in modal learning

    def forward(self, image):
        return self.image_proj(self.image_net(image)[0][:,0]) # get cls

class SwinEncoder(torch.nn.Module):
    def __init__(self, image_net, args):
        super(SwinEncoder, self).__init__()

        self.image_net = image_net
        num_fc_ftr = self.image_net.classifier.in_features
        self.image_net.classifier = nn.Linear(num_fc_ftr, args.embedding_dim)

        xavier_normal_(self.image_net.classifier.weight.data)
        if self.image_net.classifier.bias is not None:
            constant_(self.image_net.classifier.bias.data, 0)

        # self.dp = nn.Dropout(args.cv_dp) # help address overfitting in modal learning

    def forward(self, image):
        return self.image_net(image)[0]


class ResnetEncoder(torch.nn.Module):
    def __init__(self, image_net,args):
        super(ResnetEncoder, self).__init__()
        self.resnet = image_net

        # self.activate = nn.GELU() 

        num_fc_ftr = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_fc_ftr, args.embedding_dim)

        xavier_normal_(self.resnet.fc.weight.data)
        if self.resnet.fc.bias is not None:
            constant_(self.resnet.fc.bias.data, 0)

    def forward(self, image):
        return self.resnet(image)



class Wighted_Cat_Img_Text_fushion(torch.nn.Module):
    def __init__(self, args):
        super(Wighted_Cat_Img_Text_fushion, self).__init__()

        self.cv_embed = nn.ReLU(nn.Linear(args.embedding_dim, args.embedding_dim)) 
        self.text_embed = nn.ReLU(nn.Linear(args.embedding_dim, args.embedding_dim)) 
        self.dropout = nn.Dropout(args.drop_rate)

        self.layer_norm = nn.LayerNorm(args.embedding_dim, eps=1e-6)
        self.dense = nn.Linear(2 * args.embedding_dim, 1)
        self.activate = nn.Sigmoid()

    def forward(self, input_embs_text, input_embs_CV):

        # mapping    
        input_embs_all_text_ = self.text_embed(input_embs_text)
        input_embs_all_CV_ = self.cv_embed(input_embs_CV) 
        
        # normalization
        input_embs_all_text_nor = self.layer_norm(input_embs_all_text_)
        input_embs_all_CV_nor = self.layer_norm(input_embs_all_CV_)

        # fushion
        input_embs_all_CV_text_concat = torch.cat([input_embs_all_text_nor, input_embs_all_CV_nor], 1)
        alpha = self.activate(self.dense(input_embs_all_CV_text_concat)) 
        input_embs_all_CV_text_concat = alpha * input_embs_all_text_nor + (1 - alpha) * input_embs_all_CV_nor  # weighted fusion

        return input_embs_all_CV_text_concat


class Bottle_neck_Img_Text_fushion(torch.nn.Module):
    def __init__(self, args):
        super(Bottle_neck_Img_Text_fushion, self).__init__()
        self.MLPlayer = nn.Sequential(
            # layer 1
            nn.Dropout(args.drop_rate, inplace=False),
            nn.Linear(args.embedding_dim * 2, args.embedding_dim, bias=True),
            nn.ReLU(inplace=True),
            # layer 2
            nn.Dropout(args.drop_rate, inplace=False),
            nn.Linear(args.embedding_dim, args.embedding_dim * 2, bias=True),
            nn.ReLU(inplace=True),
            # layer 3
            nn.Dropout(args.drop_rate, inplace=False),
            nn.Linear(args.embedding_dim * 2, args.embedding_dim , bias=True),
            nn.ReLU(inplace=True),
        )
        self.layer_norm = nn.LayerNorm(args.embedding_dim, eps=1e-6)

    def forward(self, input_embs_text, input_embs_CV):
        
        # # fushion
        # input_embs_all_text_nor = self.layer_norm(input_embs_text)
        # input_embs_all_CV_nor = self.layer_norm(input_embs_CV)

        # input_embs_all_CV_text_concat = torch.cat([input_embs_all_text_nor, input_embs_all_CV_nor], 1)
        # input_embs_all_ = self.MLPlayer(input_embs_all_CV_text_concat)

        # return  input_embs_all_

        return  self.MLPlayer(torch.cat([self.layer_norm(input_embs_text), self.layer_norm(input_embs_CV)], 1))

