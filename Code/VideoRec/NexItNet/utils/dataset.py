import os
import math
import lmdb
import torch
import pickle
import random

import numpy as np
import torchvision as tv
import torch.distributed as dist

import torchvision.transforms as transforms

from PIL import Image
from torch.utils.data import Dataset

class LMDB_VIDEO:
    def __init__(self, video):
        self.video = video.tobytes()
        
class LMDB_Image:
    def __init__(self, image, id):
        self.image = image.tobytes()


class ItemsDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return self.data.shape[0]

class ModalDataset(Dataset):
    def __init__(self, u2seq, item_content, max_seq_len, item_num, text_size, image_db_path, video_db_path, item_id_to_keys, resize, args):
        self.u2seq = u2seq
        self.item_content = item_content
        self.max_seq_len =  max_seq_len + 1
        self.item_num = item_num
        self.text_size = text_size
        self.image_db_path = image_db_path
        self.video_db_path = video_db_path
        self.item_id_to_keys = item_id_to_keys
        self.resize = resize
        self.args = args

        self.transform = transforms.Compose([
            tv.transforms.Resize((self.resize, self.resize)),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    def __len__(self):
        return len(self.u2seq)

    def worker_init_fn(self, worker_id):
        initial_seed = torch.initial_seed() % 2 ** 31
        worker_seed = initial_seed + worker_id + self.args.local_rank + 8 * self.args.node_rank 
        random.seed(worker_seed)
        np.random.seed(worker_seed)

    def __getitem__(self, index):
        seq = self.u2seq[index]
        seq_Len = len(seq)
        tokens = seq[:-1]
        tokens_Len = len(tokens)
        mask_len_head = self.max_seq_len - seq_Len
        log_mask = [0] * mask_len_head + [1] * tokens_Len

        sample_items_text = np.zeros((self.max_seq_len, self.text_size * 2))
        sample_items_image = np.zeros((self.max_seq_len, 3, self.resize, self.resize))
        sample_items_video = np.zeros((self.max_seq_len, self.args.frame_no, 3, 224, 224))  # 暂时写死
        sample_items_id = [0] * mask_len_head + seq

        ##################################### Text #####################################
        for i in range(tokens_Len):
            # pos
            sample_items_text[mask_len_head + i] = self.item_content[seq[i]]
        # target
        sample_items_text[mask_len_head + tokens_Len] = self.item_content[seq[-1]]
        sample_items_text = torch.LongTensor(sample_items_text)

        ##################################### Image #####################################
        env = lmdb.open(self.image_db_path, subdir=os.path.isdir(self.image_db_path),
                        readonly=True, lock=False, readahead=False, meminit=False)
        with env.begin() as txn:
            for i in range(tokens_Len):
                # pos
                IMAGE = pickle.loads(txn.get(self.item_id_to_keys[seq[i]].encode()))
                image_trans = np.copy(np.frombuffer(IMAGE.image, dtype=np.float32)).reshape(3, 224, 224) 
                sample_items_image[mask_len_head + i] = image_trans
            # target
            IMAGE = pickle.loads(txn.get(self.item_id_to_keys[seq[-1]].encode()))
            image_trans = self.transform(Image.fromarray(IMAGE.get_image()).convert('RGB'))
            sample_items_image[mask_len_head + tokens_Len] = image_trans
        sample_items_image = torch.FloatTensor(sample_items_image)

        ##################################### video #####################################
        env = lmdb.open(self.video_db_path, subdir=os.path.isdir(self.video_db_path),
                        readonly=True, lock=False, readahead=False, meminit=False)
        with env.begin() as txn:
            for i in range(tokens_Len):
                # pos
                VIDEO = pickle.loads(txn.get(self.item_id_to_keys[seq[i]].encode()))
                VIDEO = np.copy(np.frombuffer(VIDEO.video, dtype=np.float32)).reshape(self.args.frame_no, 3, 224, 224) 
                sample_items_video[mask_len_head + i] = VIDEO

            # target
            VIDEO = pickle.loads(txn.get(self.item_id_to_keys[seq[-1]].encode()))
            VIDEO = np.copy(np.frombuffer(VIDEO.video, dtype=np.float32)).reshape(self.args.frame_no, 3, 224, 224) 
            sample_items_video[mask_len_head + tokens_Len] = VIDEO
        sample_items_video = torch.FloatTensor(sample_items_video)
        return sample_items_id, sample_items_text, sample_items_image, sample_items_video, \
            torch.FloatTensor(log_mask)

class ImageDataset(Dataset):
    def __init__(self, u2seq, item_num, max_seq_len, db_path, item_id_to_keys, resize):
        self.u2seq = u2seq
        self.item_num = item_num
        self.max_seq_len = max_seq_len + 1
        self.db_path = db_path
        self.item_id_to_keys = item_id_to_keys
        self.resize = resize

        self.transform = transforms.Compose([
            tv.transforms.Resize((self.resize, self.resize)),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    def __len__(self):
        return len(self.u2seq) 

    def __getitem__(self, user_id):
        seq = self.u2seq[user_id]
        seq_Len = len(seq)
        tokens_Len = len(seq) - 1
        mask_len_head = self.max_seq_len - seq_Len
        log_mask = [0] * mask_len_head + [1] * tokens_Len

        sample_items = np.zeros((self.max_seq_len, 3, self.resize, self.resize))
        sample_id_items = [0] * mask_len_head + seq

        env = lmdb.open(self.db_path, subdir=os.path.isdir(self.db_path),
             readonly=True, lock=False,
             readahead=False, meminit=False)

        with env.begin() as txn:
            for i in range(tokens_Len):
                # pos
                IMAGE = pickle.loads(txn.get(self.item_id_to_keys[seq[i]].encode()))
                image_trans = np.copy(np.frombuffer(IMAGE.image, dtype=np.float32)).reshape(3, 224, 224) 
                sample_items[mask_len_head + i] = image_trans
            # target
            IMAGE = pickle.loads(txn.get(self.item_id_to_keys[seq[-1]].encode()))
            image_trans = np.copy(np.frombuffer(IMAGE.image, dtype=np.float32)).reshape(3, 224, 224) 
            sample_items[mask_len_head + tokens_Len] = image_trans

        sample_id_items = torch.LongTensor(sample_id_items)
        sample_items = torch.FloatTensor(sample_items)
        return sample_id_items, sample_items, torch.FloatTensor(log_mask)

class TextDataset(Dataset):
    def __init__(self, userseq, item_content, max_seq_len, item_num, text_size):
        self.userseq = userseq
        self.item_content = item_content
        self.max_seq_len =  max_seq_len + 1
        self.item_num = item_num
        self.text_size = text_size

    def __len__(self):
        return len(self.userseq)

    def __getitem__(self, index):
        seq = self.userseq[index]
        seq_Len = len(seq)
        tokens = seq[:-1]
        tokens_Len = len(tokens)
        mask_len_head = self.max_seq_len - seq_Len
        log_mask = [0] * mask_len_head + [1] * tokens_Len
        
        sample_id_items = [0] * mask_len_head + seq
        sample_items = np.zeros((self.max_seq_len, self.text_size * 2))
        for i in range(tokens_Len):
            # pos
            sample_items[mask_len_head + i] = self.item_content[seq[i]]
        # target
        sample_items[mask_len_head + tokens_Len] = self.item_content[seq[-1]]
        sample_items = torch.FloatTensor(sample_items)
        sample_id_items = torch.LongTensor(sample_id_items)
        return sample_id_items, sample_items, torch.FloatTensor(log_mask)

class VideoDataset(Dataset):
    def __init__(self, u2seq, item_num, max_seq_len, item_id_to_keys, db_path, frame_no):
        self.u2seq = u2seq
        self.item_num = item_num
        self.max_seq_len = max_seq_len + 1
        self.item_id_to_keys = item_id_to_keys
        self.video_lmdb_path = db_path
        self.frame_no = frame_no

    def __len__(self):
        return len(self.u2seq)

    def __getitem__(self, user_id):
        seq = self.u2seq[user_id]
        seq_Len = len(seq)
        tokens_Len = len(seq) - 1
        mask_len_head = self.max_seq_len - seq_Len
        log_mask = [0] * mask_len_head + [1] * tokens_Len

        sample_items = np.zeros((self.max_seq_len, self.frame_no, 3, 224, 224)) 
        sample_id_items = [0] * mask_len_head + seq

        env = lmdb.open(self.video_lmdb_path, subdir=os.path.isdir(self.video_lmdb_path),
                        readonly=True, lock=False, readahead=False, meminit=False)

        with env.begin() as txn:
            for i in range(tokens_Len):
                # pos
                vdo = pickle.loads(txn.get(self.item_id_to_keys[seq[i]].encode()))
                vdo = np.copy(np.frombuffer(vdo.video, dtype=np.float32)).reshape(self.frame_no, 3, 224, 224) 
                sample_items[mask_len_head + i] = vdo

            # target
            vdo = pickle.loads(txn.get(self.item_id_to_keys[seq[-1]].encode()))
            vdo = np.copy(np.frombuffer(vdo.video, dtype=np.float32)).reshape(self.frame_no, 3, 224, 224) 
            sample_items[mask_len_head + tokens_Len] = vdo

        sample_id_items = torch.LongTensor(sample_id_items)
        sample_items = torch.FloatTensor(sample_items)
        return sample_id_items, sample_items, torch.FloatTensor(log_mask)

class IdDataset(Dataset):
    def __init__(self, u2seq, item_num, max_seq_len, args):
        self.u2seq = u2seq
        self.item_num = item_num
        self.max_seq_len = max_seq_len + 1
        self.args = args

    def __len__(self):
        return len(self.u2seq)

    def __getitem__(self, user_id):
        seq = self.u2seq[user_id]
        seq_Len = len(seq)
        tokens_Len = seq_Len - 1
        mask_len_head = self.max_seq_len - seq_Len
        log_mask = [0] * mask_len_head + [1] * tokens_Len

        sample_items = [0] * mask_len_head + seq
        sample_items = torch.LongTensor(np.array(sample_items))

        return sample_items, torch.FloatTensor(log_mask)

class IdEvalDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return self.data.shape[0]

class EvalDataset(Dataset):
    def __init__(self, u2seq, item_content, max_seq_len, item_num):
        self.u2seq = u2seq
        self.item_content = item_content
        self.max_seq_len = max_seq_len
        self.item_num = item_num

    def __len__(self):
        return len(self.u2seq)

    def __getitem__(self, user_id):
        seq = self.u2seq[user_id]
        tokens = seq[:-1]
        target = seq[-1]
        mask_len = self.max_seq_len - len(seq)
        pad_tokens = [0] * mask_len + tokens
        log_mask = [0] * mask_len + [1] * len(tokens)
        input_embs = self.item_content[pad_tokens]
        labels = np.zeros(self.item_num)
        labels[target - 1] = 1.0
        return torch.LongTensor([user_id]), input_embs, torch.FloatTensor(log_mask), labels

class LmdbEvalDataset(Dataset):
    def __init__(self, data, item_id_to_keys, db_path, resize, mode, frame_no=-1):
        self.data = data
        self.item_id_to_keys = item_id_to_keys
        self.db_path = db_path
        self.resize = resize
        self.mode = mode
        self.frame_no = frame_no
        if mode == 'image':
            self.padding_emb = torch.zeros((3, 224, 224)) 
        else:
            self.padding_emb = torch.zeros((self.frame_no, 3, 224, 224)) 

        # self.transform = transforms.Compose([
        #         tv.transforms.Resize((self.resize, self.resize)),
        #         tv.transforms.ToTensor(),
        #         tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        #     ])

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        item_id = self.data[index]
        if index == 0:
            if self.mode == 'image':
                return torch.zeros((3, 224, 224)) 
            else:
                return torch.zeros((self.frame_no, 3, 224, 224)) 

        env = lmdb.open(self.db_path, subdir=os.path.isdir(self.db_path), \
            readonly=True, lock=False, readahead=False, meminit=False)
        with env.begin() as txn:
            byteflow = txn.get(self.item_id_to_keys[item_id].encode())
        if self.mode == 'image':
            IMAGE = pickle.loads(byteflow)
            output = np.frombuffer(IMAGE.image, dtype=np.float32).reshape(3, 224, 224) 
        else:
            VIDEO = pickle.loads(byteflow)
            output = np.frombuffer(VIDEO.video, dtype=np.float32).reshape(self.frame_no, 3, 224, 224) 
        return torch.FloatTensor(output)

class SequentialDistributedSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, dataset, batch_size, rank=None, num_replicas=None):
        if num_replicas is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = torch.distributed.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.batch_size = batch_size
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.batch_size / self.num_replicas)) * self.batch_size
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        # add extra samples to make it evenly divisible
        indices += [indices[-1]] * (self.total_size - len(indices))
        # subsample
        indices = indices[self.rank * self.num_samples : (self.rank + 1) * self.num_samples]
        return iter(indices)

    def __len__(self):
        return self.num_samples