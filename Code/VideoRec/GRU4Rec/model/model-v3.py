import torch
import numpy as np
from torch import nn
from torch.nn.init import xavier_normal_
from collections import Counter
import torch.nn.functional as F

from .text_encoders import TextEmbedding
from .video_encoders import VideoMaeEncoder, R3D18Encoder
from .image_encoders import VitEncoder, ResnetEncoder, MaeEncoder, SwinEncoder 
from .fusion_module import SumFusion, ConcatFusion, FiLM, GatedFusion 
from .user_encoders import UserEncoder

class Model(torch.nn.Module):
    def __init__(self, args, pop_prob_list, item_num, bert_model, image_net, video_net, text_content=None):
        super(Model, self).__init__()
        self.args = args
        self.max_seq_len = args.max_seq_len
        self.item_num = item_num
        self.pop_prob_list = torch.FloatTensor(pop_prob_list)
        self.bn = nn.BatchNorm1d(args.embedding_dim)

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

    def nxn_cos_sim(self, A, B, dim=1, eps=1e-8):
        numerator = A @ B.T
        A_l2 = torch.mul(A, A).sum(axis=dim)
        B_l2 = torch.mul(B, B).sum(axis=dim)
        denominator = torch.max(torch.sqrt(torch.outer(A_l2, B_l2)), torch.tensor(eps))
        return torch.div(numerator, denominator)

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
            score_embs = self.video_encoder(sample_items_video)
        elif 'id' == args.item_tower:
            score_embs = self.id_encoder(sample_items_id)

        # print(sample_items_text.shape) # 1408 60
        # print(score_embs.shape) # 1408 512
        input_embs = score_embs.view(-1, self.max_seq_len + 1, self.args.embedding_dim)
        # print(input_embs.shape) # 128 11 512
        prec_vec = self.user_encoder(input_embs[:, :-1, :], log_mask, local_rank)
        # print(prec_vec.shape) # 128 10 512
        prec_vec = prec_vec.view(-1, self.args.embedding_dim)
        # print(prec_vec.shape) # 1280 512

        ######################################  IN-BATCH CROSS-ENTROPY LOSS  ######################################
        # logits = torch.matmul(F.normalize(prec_vec, dim=-1), F.normalize(score_embs, dim=-1).t()) # (bs * max_seq_len, bs * (max_seq_len + 1))
        # logits = logits / self.args.tau - debias_logits
        logits = torch.matmul(prec_vec, score_embs.t())
        # print(score_embs.shape, score_embs.t().shape) # 1408 512, 512 1408
        logits = logits - debias_logits
        # print(logits.shape) # 1280 1408
        # print(logits)

        ###################################### MASK USELESS ITEM ######################################
        bs, seq_len = log_mask.size(0), log_mask.size(1)
        # print(bs, seq_len) # 128 10
        label = torch.arange(bs * (seq_len + 1)).reshape(bs, seq_len + 1)
        # print(label.shape, label) # 128 11
        label = label[:, 1:].to(local_rank).view(-1)
        # print(label.shape, label) # 1280

        flatten_item_seq = sample_items_id
        # print(flatten_item_seq.shape, flatten_item_seq) # 1408
        user_history = torch.zeros(bs, seq_len + 2).type_as(sample_items_id)
        # print(user_history.shape, user_history) # 128 12
        user_history[:, :-1] = sample_items_id.view(bs, -1)
        # print(user_history.shape, user_history) # 128 12
        user_history = user_history.unsqueeze(-1).expand(-1, -1, len(flatten_item_seq))
        # print(user_history.shape, user_history) # 128 12 1408
        history_item_mask = (user_history == flatten_item_seq).any(dim=1)
        # print(history_item_mask.shape, history_item_mask) # 128 1408
        history_item_mask = history_item_mask.repeat_interleave(seq_len, dim=0)
        # print(history_item_mask.shape, history_item_mask) # 1280 1408
        unused_item_mask = torch.scatter(history_item_mask, 1, label.view(-1, 1), False)
        # print(unused_item_mask.shape, unused_item_mask) # 1280 1408

        logits[unused_item_mask] = -1e4
        # print(logits.shape, logits) # 1280 1408
        indices = torch.where(log_mask.view(-1) != 0)
        # print(len(indices[0]), indices) # 
        logits = logits.view(bs * seq_len, -1)
        # print(logits.shape, logits) # 1280 1408
        # print(logits[indices].shape, label[indices].shape) # 467 1408, 467
        loss = self.criterion(logits[indices], label[indices])

        ###################################### CALCULATE ALIGNMENT AND UNIFORMITY ######################################
        user = prec_vec.view(-1, self.max_seq_len, self.args.embedding_dim)[:, -1, :]
        item = score_embs.view(-1, self.max_seq_len + 1, self.args.embedding_dim)[:, -1, :]
        align = self.alignment(user, item)
        uniform = (self.uniformity(user) + self.uniformity(item)) / 2
        # print(loss)
        
        ###################################### MOD-INDICATED LOSS ######################################
        bs, seq_len_add_one, ed = input_embs.shape[0], input_embs.shape[1], input_embs.shape[-1]

        input_embs = input_embs.view(-1, ed)
        input_embs = self.bn(input_embs)
        input_embs = input_embs.view(bs, seq_len_add_one, ed)
        input_embs = F.normalize(input_embs, dim=-1)

        # hist_embs = input_embs[:, :-1, :]
        prec_vec = self.bn(prec_vec)
        prec_vec = prec_vec.view(bs, seq_len_add_one - 1, ed)
        hist_embs = F.normalize(prec_vec, dim=-1)

        targets_embs = input_embs[:, -1, :]

        filter_mat = sample_items_id.reshape(bs, seq_len_add_one)
        filter_mat = filter_mat.unsqueeze(-1).expand(-1, -1, len(sample_items_id))
        original_seq = sample_items_id.unsqueeze(0)
        mask = (filter_mat != original_seq).all(dim=1)

        input_embs = input_embs.view(-1, ed)
        mi_loss = 0
        alpha = 0.00005
        for i in range(len(mask)):
            target_embs = targets_embs[i]
            current_indexs = mask[i]
            other_item_embs = input_embs[current_indexs]
            sim_mat = self.nxn_cos_sim(target_embs.unsqueeze(0), other_item_embs)
            closest_index = sim_mat.argmax(dim=1)
            closest_embs = other_item_embs[closest_index][0]

            mi_loss += (hist_embs[i] - closest_embs).norm(p=2, dim=1).mean() # calculate distance as loss to push together, alpha = 0.00005
            # mi_loss += self.nxn_cos_sim(closest_embs.unsqueeze(0), hist_embs[i]).mean() # calculate similarity as loss to pull apart, alpha = 0.001

        # print(loss)
        # print(mi_loss)
        # xxx

        return loss+alpha*mi_loss, align, uniform