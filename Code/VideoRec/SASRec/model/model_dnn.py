import torch
import numpy as np
from torch import nn
from torch.nn.init import xavier_normal_, constant_
from collections import Counter
import torch.nn.functional as F

from .text_encoders import TextEmbedding
from .video_encoders import VideoMaeEncoder, R3D18Encoder, R3D50Encoder, C2D50Encoder
from .video_encoders import I3D50Encoder, CSN101Encoder, SLOW50Encoder, EX3DSEncoder
from .video_encoders import EX3DXSEncoder, X3DXSEncoder, X3DSEncoder, X3DMEncoder
from .video_encoders import X3DLEncoder, MVIT16Encoder, MVIT16X4Encoder, MVIT32X3Encoder
from .video_encoders import SLOWFAST50Encoder, SLOWFAST16X8101Encoder
from .image_encoders import VitEncoder, ResnetEncoder, MaeEncoder, SwinEncoder 
from .fusion_module import SumFusion, ConcatFusion, FiLM, GatedFusion 
from .user_encoders import UserEncoder

class MLP_Layers(torch.nn.Module):
    def __init__(self, layers, dnn_layers, drop_rate):
        super(MLP_Layers, self).__init__()
        self.layers = layers
        self.dnn_layers = dnn_layers
        if self.dnn_layers > 0:
            mlp_modules = []
            for idx, (input_size, output_size) in enumerate(zip(self.layers[:-1], self.layers[1:])):
                mlp_modules.append(nn.Dropout(p=drop_rate))
                mlp_modules.append(nn.Linear(input_size, output_size))
                mlp_modules.append(nn.GELU())
            self.mlp_layers = nn.Sequential(*mlp_modules)
            self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight.data)
        elif isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)
            if module.bias is not None:
                constant_(module.bias.data, 0)

    def forward(self, x):
        if self.dnn_layers > 0:
            return self.mlp_layers(x)
        else:
            return x

class Model(torch.nn.Module):
    def __init__(self, args, pop_prob_list, item_num, bert_model, image_net, video_net, text_content=None):
        super(Model, self).__init__()
        self.args = args
        self.max_seq_len = args.max_seq_len
        self.item_num = item_num
        self.pop_prob_list = torch.FloatTensor(pop_prob_list)

        self.mlp_layers = MLP_Layers(layers=[args.embedding_dim] * (args.dnn_layers + 1),
                              dnn_layers=args.dnn_layers,
                              drop_rate=args.drop_rate)

        self.user_encoder = UserEncoder(
            item_num=item_num,
            max_seq_len=args.max_seq_len,
            item_dim=args.embedding_dim,
            num_attention_heads=args.num_attention_heads,
            dropout=args.drop_rate,
            n_layers=args.transformer_block)

        if 'image' == args.item_tower or 'modal' == args.item_tower:
            if 'resnet' in args.image_model_load:
                self.image_encoder = ResnetEncoder(image_net=image_net, args=args)
            elif 'vit-b-32-clip' in args.image_model_load:
                self.image_encoder = VitEncoder(image_net=image_net, args=args)
            elif 'vit-base-mae' in args.image_model_load:
                self.image_encoder = MaeEncoder(image_net=image_net, args=args)
            elif 'swin_tiny' in args.image_model_load or 'swin_base' in args.image_model_load:
                self.image_encoder = SwinEncoder(image_net=image_net, args=args)

        if 'text' == args.item_tower or 'modal' == args.item_tower:
            self.text_content = torch.LongTensor(text_content)
            self.text_encoder = TextEmbedding(args=args, bert_model=bert_model)
        
        if 'video' == args.item_tower or 'modal' == args.item_tower:
            if 'mae' in args.video_model_load:
                self.video_encoder = VideoMaeEncoder(video_net=video_net, args=args)
            elif 'r3d18' in args.video_model_load:
                self.video_encoder = R3D18Encoder(video_net=video_net, args=args)
            elif 'r3d50' in args.video_model_load:
                self.video_encoder = R3D50Encoder(video_net=video_net, args=args)
            elif 'c2d50' in args.video_model_load:
                self.video_encoder = C2D50Encoder(video_net=video_net, args=args)
            elif 'i3d50' in args.video_model_load:
                self.video_encoder = I3D50Encoder(video_net=video_net, args=args)
            elif 'csn101' in args.video_model_load:
                self.video_encoder = CSN101Encoder(video_net=video_net, args=args)
            elif 'slow50' in args.video_model_load:
                self.video_encoder = SLOW50Encoder(video_net=video_net, args=args)
            elif 'efficient-x3d-s' in args.video_model_load:
                self.video_encoder = EX3DSEncoder(video_net=video_net, args=args)
            elif 'efficient-x3d-xs' in args.video_model_load:
                self.video_encoder = EX3DXSEncoder(video_net=video_net, args=args)
            elif 'x3d-xs' in args.video_model_load:
                self.video_encoder = X3DXSEncoder(video_net=video_net, args=args)
            elif 'x3d-s' in args.video_model_load:
                self.video_encoder = X3DSEncoder(video_net=video_net, args=args)
            elif 'x3d-m' in args.video_model_load:
                self.video_encoder = X3DMEncoder(video_net=video_net, args=args)
            elif 'x3d-l' in args.video_model_load:
                self.video_encoder = X3DLEncoder(video_net=video_net, args=args)
            elif 'mvit-base-16' in args.video_model_load:
                self.video_encoder = MVIT16Encoder(video_net=video_net, args=args)
            elif 'mvit-base-16x4' in args.video_model_load:
                self.video_encoder = MVIT16X4Encoder(video_net=video_net, args=args)
            elif 'mvit-base-32x3' in args.video_model_load:
                self.video_encoder = MVIT32X3Encoder(video_net=video_net, args=args)
            elif 'slowfast-50' in args.video_model_load:
                self.video_encoder = SLOWFAST50Encoder(video_net=video_net, args=args)
            elif 'slowfast16x8-101' in args.video_model_load:
                self.video_encoder = SLOWFAST16X8101Encoder(video_net=video_net, args=args)

        self.id_encoder = nn.Embedding(item_num + 1, args.embedding_dim, padding_idx=0)
        xavier_normal_(self.id_encoder.weight.data)

        self.criterion = nn.CrossEntropyLoss()

        fusion = args.fusion_method.lower()
        if fusion == 'concat' and args.item_tower == 'modal':
            self.fusion_module = ConcatFusion(args=args)

    def alignment(self, x, y):
        x, y = F.normalize(x, dim=-1), F.normalize(y, dim=-1)
        return (x - y).norm(p=2, dim=1).pow(2).mean()

    def uniformity(self, x):
        x = F.normalize(x, dim=-1)
        return torch.pdist(x, p=2).pow(2).mul(-2).exp().mean().log()

    def forward(self, sample_items_id, sample_items_text, sample_items_image, sample_items_video, log_mask, local_rank, args):
        self.pop_prob_list = self.pop_prob_list.to(local_rank)
        debias_logits = torch.log(self.pop_prob_list[sample_items_id.view(-1)])

        if 'modal' == args.item_tower:
            input_all_text = self.text_encoder(sample_items_text.long())
            input_all_image = self.image_encoder(sample_items_image)
            input_all_video = self.video_encoder(sample_items_video)
            input_embs = self.fusion_module(input_all_text, input_all_image, input_all_video)
        elif 'text' == args.item_tower:
            score_embs = self.text_encoder(sample_items_text.long())
        elif 'image' == args.item_tower:
            score_embs = self.image_encoder(sample_items_image)
        elif 'video' == args.item_tower:
            score_embs = self.mlp_layers(self.video_encoder(sample_items_video))
        elif 'id' == args.item_tower:
            score_embs = self.id_encoder(sample_items_id)

        input_embs = score_embs.view(-1, self.max_seq_len + 1, self.args.embedding_dim)        
        prec_vec = self.user_encoder(input_embs[:, :-1, :], log_mask, local_rank)
        prec_vec = prec_vec.view(-1, self.args.embedding_dim)

        ######################################  IN-BATCH CROSS-ENTROPY LOSS  ######################################
        # logits = torch.matmul(F.normalize(prec_vec, dim=-1), F.normalize(score_embs, dim=-1).t()) # (bs * max_seq_len, bs * (max_seq_len + 1))
        # logits = logits / self.args.tau - debias_logits
        logits = torch.matmul(prec_vec, score_embs.t())
        logits = logits - debias_logits

        ###################################### MASK USELESS ITEM ######################################
        bs, seq_len = log_mask.size(0), log_mask.size(1)
        label = torch.arange(bs * (seq_len + 1)).reshape(bs, seq_len + 1)
        label = label[:, 1:].to(local_rank).view(-1)

        flatten_item_seq = sample_items_id
        user_history = torch.zeros(bs, seq_len + 2).type_as(sample_items_id)
        user_history[:, :-1] = sample_items_id.view(bs, -1)
        user_history = user_history.unsqueeze(-1).expand(-1, -1, len(flatten_item_seq))
        history_item_mask = (user_history == flatten_item_seq).any(dim=1)
        history_item_mask = history_item_mask.repeat_interleave(seq_len, dim=0)
        unused_item_mask = torch.scatter(history_item_mask, 1, label.view(-1, 1), False)
        
        logits[unused_item_mask] = -1e4
        indices = torch.where(log_mask.view(-1) != 0)
        logits = logits.view(bs * seq_len, -1)
        loss = self.criterion(logits[indices], label[indices])

        ###################################### CALCULATE ALIGNMENT AND UNIFORMITY ######################################
        user = prec_vec.view(-1, self.max_seq_len, self.args.embedding_dim)[:, -1, :]
        item = score_embs.view(-1, self.max_seq_len + 1, self.args.embedding_dim)[:, -1, :]
        align = self.alignment(user, item)
        uniform = (self.uniformity(user) + self.uniformity(item)) / 2
        
        return loss, align, uniform
