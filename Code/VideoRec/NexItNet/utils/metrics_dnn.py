import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.distributed as dist
import os
import math
from .dataset import  EvalDataset, SequentialDistributedSampler, LmdbEvalDataset, IdEvalDataset, ItemsDataset



def item_collate_fn(arr):
    arr = torch.LongTensor(np.array(arr))
    return arr

def id_collate_fn(arr):
    arr = torch.LongTensor(arr)
    return arr

def print_metrics(x, Log_file, v_or_t):
    Log_file.info(v_or_t+'_results   {}'.format('\t'.join(['{:0.5f}'.format(i * 100) for i in x])))

def get_mean(arr):
    return [i.mean() for i in arr]

def distributed_concat(tensor, num_total_examples):
    output_tensors = [tensor.clone() for _ in range(dist.get_world_size())]
    dist.all_gather(output_tensors, tensor)
    concat = torch.cat(output_tensors, dim=0)
    return concat[:num_total_examples]

def eval_concat(eval_list, test_sampler):
    eval_result = []
    for eval_m in eval_list:
        eval_m_cpu = distributed_concat(eval_m, len(test_sampler.dataset))\
            .to(torch.device('cpu')).numpy()
        eval_result.append(eval_m_cpu.mean())
    return eval_result

def metrics_topK(y_score, y_true, item_rank, topK, local_rank):
    order = torch.argsort(y_score, descending=True)
    y_true = torch.take(y_true, order)
    rank = torch.sum(y_true * item_rank)
    eval_ra = torch.zeros(2).to(local_rank)
    if rank <= topK:
        eval_ra[0] = 1
        eval_ra[1] = 1 / math.log2(rank + 1)
    return rank, eval_ra

def get_item_id_score(model, item_num, test_batch_size, args, local_rank):
    model.eval()
    item_dataset = IdEvalDataset(data=np.arange(item_num + 1))
    item_dataloader = DataLoader(item_dataset, batch_size=test_batch_size,
                                 num_workers=args.num_workers, pin_memory=True, collate_fn=item_collate_fn)
    item_scoring = []
    with torch.no_grad():
        for input_ids in item_dataloader:
            input_ids = input_ids.to(local_rank)
            item_emb = model.module.id_encoder(input_ids).to(torch.device('cpu')).detach()
            item_scoring.extend(item_emb)
    return torch.stack(tensors=item_scoring, dim=0)

def get_item_text_score(model, item_content, test_batch_size, args, local_rank):
    model.eval()
    item_dataset = ItemsDataset(item_content)
    item_dataloader = DataLoader(item_dataset, batch_size=test_batch_size, num_workers=args.num_workers,
                                 pin_memory=True, collate_fn=item_collate_fn)
    item_scoring = []
    with torch.no_grad():
        for input_ids in item_dataloader:
            input_ids = input_ids.to(local_rank)
            item_emb = model.module.text_encoder(input_ids)
            item_scoring.extend(item_emb)
    return torch.stack(tensors=item_scoring, dim=0).to(torch.device('cpu')).detach()

def get_item_image_score(model, item_num, item_id_to_keys, test_batch_size, args, local_rank):
    model.eval()
    item_dataset = LmdbEvalDataset(data=np.arange(item_num + 1), item_id_to_keys=item_id_to_keys,
                                   db_path=os.path.join(args.root_data_dir, args.dataset, args.image_data),
                                   resize=args.image_resize, mode='image')
    item_dataloader = DataLoader(item_dataset, batch_size=test_batch_size,
                                 num_workers=args.num_workers, pin_memory=True)
    item_scoring = []
    with torch.no_grad():
        for input_ids in item_dataloader:
            input_ids = input_ids.to(local_rank)
            item_emb = model.module.image_encoder(input_ids)
            item_scoring.extend(item_emb)
    return torch.stack(tensors=item_scoring, dim=0).to(torch.device('cpu')).detach()

def get_item_video_score(model, item_num, item_id_to_keys, test_batch_size, args, local_rank):
    model.eval()
    item_dataset = LmdbEvalDataset(data=np.arange(item_num + 1), item_id_to_keys=item_id_to_keys,
                                   db_path=os.path.join(args.root_data_dir, args.dataset, args.video_data),
                                   resize=args.image_resize, mode='video', frame_no=args.frame_no)
    item_dataloader = DataLoader(item_dataset, batch_size=test_batch_size,
                                 num_workers=args.num_workers, pin_memory=True)
    item_scoring = []
    with torch.no_grad():
        for input_ids in item_dataloader:
            input_ids = input_ids.to(local_rank)
            item_emb = model.module.mlp_layers(model.module.video_encoder(input_ids))
            item_scoring.extend(item_emb)
    return torch.stack(tensors=item_scoring, dim=0).to(torch.device('cpu')).detach()

def get_fusion_score(model, item_scoring_text, item_scoring_image, item_scoring_video, local_rank, args):
    model.eval()
    with torch.no_grad():
        if 'modal' in args.item_tower:
            item_scoring_text = item_scoring_text.to(local_rank)
            item_scoring_image = item_scoring_image.to(local_rank)
            item_scoring_video = item_scoring_video.to(local_rank)
            item_scoring = model.module.fusion_module(item_scoring_text, item_scoring_image, item_scoring_video)

    return item_scoring.to(torch.device('cpu')).detach()


def eval_model(model, user_history, eval_seq, item_scoring, test_batch_size, args, item_num, Log_file, v_or_t, pop_prob_list, local_rank, epoch):
    from tqdm import tqdm
    from numpy import savetxt
    eval_dataset = EvalDataset(u2seq=eval_seq, item_content=item_scoring,
                                    max_seq_len=args.max_seq_len+1, item_num=item_num)
    test_sampler = SequentialDistributedSampler(eval_dataset, batch_size=test_batch_size)
    eval_dl = DataLoader(eval_dataset, batch_size=test_batch_size,
                         num_workers=args.num_workers, pin_memory=True, sampler=test_sampler)
    model.eval()
    topK = 10
    Log_file.info(v_or_t + '_methods   {}'.format('\t'.join(['Hit{}'.format(topK), 'nDCG{}'.format(topK)])))
    item_scoring = item_scoring.to(local_rank)
    with torch.no_grad():
        eval_all_user = []
        item_rank = torch.Tensor(np.arange(item_num) + 1).to(local_rank)
        user_list = []
        item_list = []
        rank_list = []
        for data in eval_dl:
            user_ids, input_embs, log_mask, labels = data
            user_ids, input_embs, log_mask, labels = \
                user_ids.to(local_rank), input_embs.to(local_rank),\
                log_mask.to(local_rank), labels.to(local_rank).detach()
            prec_emb = model.module.user_encoder(input_embs, log_mask, local_rank)[:, -1].detach()
            scores = torch.matmul(prec_emb, item_scoring.t()).squeeze(dim=-1).detach()
            for user_id, label, score in zip(user_ids, labels, scores):
                user_id = user_id[0].item()
                history = user_history[user_id].to(local_rank)
                score[history] = -np.inf
                score = score[1:]
                rank, res = metrics_topK(score, label, item_rank, topK, local_rank)
                rank_list.append(rank.detach().cpu())
                user_list.append(user_id)   
                item_list.append(pop_prob_list[eval_seq[user_id][-1]])
                eval_all_user.append(res)
        eval_all_user = torch.stack(tensors=eval_all_user, dim=0).t().contiguous()
        Hit10, nDCG10 = eval_all_user
        mean_eval = eval_concat([Hit10, nDCG10], test_sampler)
        dataset = 'ks'
        mode = args.item_tower
        np.save('./results/{}/{}/rank_list_{}.npy'.format(dataset, mode, epoch-1), np.array(rank_list))
        np.save('./results/{}/{}/user_list_{}.npy'.format(dataset, mode, epoch-1), np.array(user_list))
        np.save('./results/{}/{}/item_list_{}.npy'.format(dataset, mode, epoch-1), np.array(item_list))
        # savetxt('./embeddings/embeddings-{}.csv'.format(mean_eval[0] * 100), item_scoring.to(torch.device('cpu')).detach(), delimiter=',')
        print_metrics(mean_eval, Log_file, v_or_t)
    return mean_eval[0], mean_eval[1]
