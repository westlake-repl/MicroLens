import os, lmdb, tqdm, pickle, json
import numpy as np
import torchvision as tv
from PIL import Image
import torchvision.transforms as transforms
from transformers import VideoMAEFeatureExtractor, VideoMAEConfig

class LMDB_Image:
    def __init__(self, image):
        self.image = image.tobytes()

class LMDB_VIDEO:
    def __init__(self, video):
        self.video = video.tobytes()

transform = transforms.Compose([
        tv.transforms.Resize((224, 224)),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

def generate_image_lmdb():
	image_folder = 'cover_folder/' # to input
	all_image = os.listdir(image_folder)
	image_num = len(all_image)
	lmdb_path = 'xxx.lmdb' # to input
	isdir = os.path.isdir(lmdb_path)
	lmdb_env = lmdb.open(lmdb_path, subdir=isdir, map_size=image_num * np.zeros((3, 224, 224)).nbytes*10,
		readonly=False, meminit=False, map_async=True)
	txn = lmdb_env.begin(write=True)
	write_frequency = 100
	for idx, image in enumerate(tqdm.tqdm(all_image)):
		img = np.array(transform(Image.open(image_folder+image).convert('RGB')))
		image_id = int(image.replace('.jpg', ''))
		temp = LMDB_Image(img)
		txn.put(u'{}'.format(image_id).encode('ascii'), pickle.dumps(temp))
		if image_id % write_frequency == 0 and idx != 0:
			txn.commit()
			txn = lmdb_env.begin(write=True)
	txn.commit()
	keys = [u'{}'.format(k).encode('ascii') for k in range(image_num)]
	with lmdb_env.begin(write=True) as txn:
	    txn.put(b'__keys__', pickle.dumps(keys))
	    txn.put(b'__len__', pickle.dumps(len(keys)))
	print(len(keys))
	print("Flushing database ...")
	lmdb_env.sync()
	lmdb_env.close()

def generate_video_lmdb(pretrain_path, video_path, frame_no):

	configuration = VideoMAEConfig()
	feature_extractor = VideoMAEFeatureExtractor(configuration)
	feature_extractor = feature_extractor.from_pretrained(pretrain_path)

	frames_folder = 'video_frames/' # a folder that restores frames of videos
	# {video1-1.jpg, video1-2.jpg, video1-3.jpg, video1-4.jpg, video1-5.jpg} refer to 5 frames extracted from video1
	lmdb_path = 'video_frames.lmdb' # to input

	all_video = [x.replace('.mp4', '.jpg') for x in os.listdir(video_path)]
	video_num = len(all_video)
	isdir = os.path.isdir(lmdb_path)
	lmdb_env = lmdb.open(lmdb_path, subdir=isdir, map_size=video_num * np.zeros((3, 224, 224)).nbytes*10,
		readonly=False, meminit=False, map_async=True)
	txn = lmdb_env.begin(write=True)
	write_frequency = 100
	for idx, video in enumerate(tqdm.tqdm(all_video)):
		temp = []
		video_id = int(video)
		for frame_idx in range(1, frame_no+1):
			img = np.array(transform(Image.open(frames_folder+str(video_id)+'-'+str(frame_idx)+'.jpg').convert('RGB')))		
			temp.append(img)
		temp = list(np.array(temp))
		temp = feature_extractor(temp, return_tensors="pt").pixel_values[0]
		temp = LMDB_VIDEO(np.array(temp))
		txn.put(u'{}'.format(video_id).encode('ascii'), pickle.dumps(temp))
		if video_id % write_frequency == 0 and idx != 0:
			txn.commit()
			txn = lmdb_env.begin(write=True)
	txn.commit()
	keys = [u'{}'.format(k).encode('ascii') for k in range(video_num)]
	with lmdb_env.begin(write=True) as txn:
		txn.put(b'__keys__', pickle.dumps(keys))
		txn.put(b'__len__', pickle.dumps(len(keys)))
	print(len(keys))
	print("Flushing database ...")
	lmdb_env.sync()
	lmdb_env.close()



generate_image_lmdb()

pretrain_path = './videomae_base' # to input
video_path = './xxx_videos'
for frame_no in [1, 2, 3, 4, 5]:
	generate_video_lmdb(pretrain_path, video_path, frame_no)