import lmdb, os, pickle, torch
import numpy as np
from transformers import VideoMAEFeatureExtractor, VideoMAEModel, VideoMAEConfig
import torch.nn as nn
import json

device = torch.device('cuda:4' if torch.cuda.is_available() else "cpu") 

class LMDB_VIDEO:
    def __init__(self, video):
        self.video = video.tobytes()

frame_no = 5
video_lmdb_path = 'D:/video_frames_100k.lmdb'

video_number = 19738 # to input

avg_pool = nn.AdaptiveAvgPool2d((1, 768))
configuration = VideoMAEConfig(num_frames=frame_no)
video_model = VideoMAEModel.from_pretrained('E:/MM_DATASET/DATA_GENERATION/videomae_base', config = configuration).to(device)

env = lmdb.open(video_lmdb_path, subdir=os.path.isdir(video_lmdb_path),
                        readonly=True, lock=False, readahead=False, meminit=False)

all_scoring = torch.tensor([]).to(torch.float32).to('cpu')
with torch.no_grad():
    with env.begin() as txn:
        for i in range(1, video_number+1, 100):
            if i != (video_number-video_number%100+1):
                all_videos = np.zeros((100, 5, 3, 224, 224))
                max_step = 100
            else:
                all_videos = np.zeros((video_number%100, 5, 3, 224, 224))
                max_step = video_number%100
            k = 0
            for j in range(i, i+max_step):
                vdo = pickle.loads(txn.get(str(j).encode()))
                vdo = np.frombuffer(vdo.video, dtype=np.float32).reshape(frame_no, 3, 224, 224) 
                all_videos[k] = vdo
                k += 1
            all_videos = torch.from_numpy(all_videos).to(torch.float32).to(device)
            item_scoring = video_model(all_videos).last_hidden_state
            item_scoring = avg_pool(item_scoring)
            item_scoring = item_scoring.squeeze(1) # torch.Size([112, 512])
            item_scoring = item_scoring.to('cpu')
            all_scoring = torch.cat([all_scoring, item_scoring], 0)
            print(all_scoring.shape)
            torch.cuda.empty_cache()

print(all_scoring.shape)
all_scoring = all_scoring.data.cpu().numpy()
np.save('MicroLens-100k_video_features_VideoMAE.npy', all_scoring)


### npy to json file, str(item id) is the key, embedding (python list format) is the value.
all_scoring = np.load('MicroLens-100k_video_features_VideoMAE.npy')
print(all_scoring.shape)
feature_json = {}
for i in range(all_scoring.shape[0]):
    feature_json[str(i+1)] = all_scoring[i].tolist() # Convert NumPy array to list
with open('MicroLens-100k_video_features_VideoMAE.json', 'w') as f:
    json.dump(feature_json, f)

with open('MicroLens-100k_video_features_VideoMAE.json', 'r') as f:
    feature_json = json.load(f)
print(len(feature_json))
