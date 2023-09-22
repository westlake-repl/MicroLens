# import torch

# c2d_r50,
# csn_r101,
# efficient_x3d_s,
# efficient_x3d_xs,
# i3d_r50,
# mvit_base_16,
# mvit_base_16x4,
# mvit_base_32x3,
# r2plus1d_r50,
# slow_r50,
# slow_r50_detection, detection models need boxes as input
# slowfast_16x8_r101_50_50, 
# slowfast_r101,
# slowfast_r50,
# slowfast_r50_detection, detection models need boxes as input
# x3d_l,
# x3d_m,
# x3d_s,
# x3d_xs,

# model = torch.hub.load('./', 'c2d_r50', source='local', pretrained=False)
# for index, (name, param) in enumerate(model.named_parameters()):
# 	print(index, name)



import torch
device = "cpu"
# Pick a pretrained model and load the pretrained weights
model_name = "slowfast_r50"
model = torch.hub.load('./', model=model_name, source='local', head_pool_kernel_sizes=((1, 7, 7), (4, 7, 7)), pretrained=False) # slowfast_r50
# model = torch.hub.load('./', model=model_name, source='local', pretrained=True) # slowfast_16x8_r101_50_50
# since we input 5 frames for video recommendation, for slowfast_16x8_r101_50_50, change "head_pool_kernel_sizes=((16, 7, 7), (64, 7, 7))" to "head_pool_kernel_sizes=((1, 7, 7), (4, 7, 7))" in line 145 of models/hub/slowfast.py
model = model.to(device)
model = model.eval()
inputs = [torch.ones([5, 3, 2, 224, 224]), torch.ones([5, 3, 5, 224, 224])] # bs, channel, frames, H, W
preds = model(inputs)
print(preds[0][:10])

# import torch
# device = "cpu"
# # # slow_r50
# # # since we input 5 frames for video recommendation, for slow_r50, change "head_pool_kernel_size=(8, 7, 7)" to "head_pool_kernel_size=(1, 7, 7)" in line 67 of models/hub/resnet.py
# model = torch.hub.load('./', 'slow_r50', source='local', pretrained=False)
# model = model.eval()
# model = model.to(device)
# inputs = torch.ones([1, 3, 5, 224, 224]) # bs, channel, frames, H, W
# preds = model(inputs)
# print(preds[0][:10])

# import torch
# device = "cpu"
# model = torch.hub.load('./', 'r2plus1d_r50', source='local', head_pool_kernel_size=(1, 7, 7), pretrained=True)
# model = model.eval()
# model = model.to(device)
# inputs = torch.ones([1, 3, 5, 224, 224]) # bs, channel, frames, H, W
# preds = model(inputs)
# print(preds.shape)
# print(preds[0][:10])

# import torch
# device = "cpu"
# # # # since we input 5 frames for video recommendation, for csn_r101, change "head_pool_kernel_size=(4, 7, 7)" to "head_pool_kernel_size=(1, 7, 7)" in line 45 of models/hub/csn.py
# model = torch.hub.load('./', 'csn_r101', source='local', pretrained=True)
# model = model.eval()
# model = model.to(device)
# inputs = torch.ones([1, 3, 5, 224, 224]) # bs, channel, frames, H, W
# preds = model(inputs)
# print(preds.shape)
# print(preds[0][:10])

# import torch
# device = "cpu"
# model = torch.hub.load('./', 'c2d_r50', source='local', head_pool_kernel_size=(1, 7, 7), pretrained=True)
# model = model.eval()
# model = model.to(device)
# inputs = torch.ones([1, 3, 5, 224, 224]) # bs, channel, frames, H, W
# preds = model(inputs)
# print(preds.shape)
# print(preds[0][:10])

# import torch
# device = "cpu"
# model = torch.hub.load('./', 'efficient_x3d_s', source='local', pretrained=True) # efficient_x3d_s efficient_x3d_xs
# model = model.eval()
# model = model.to(device)
# inputs = torch.ones([1, 3, 5, 224, 224]) # bs, channel, frames, H, W
# preds = model(inputs)
# print(preds.shape)
# print(preds[0][:10])

# import torch
# device = "cpu"
# # # line 541 512 715-719 of models/x3d.py
# model = torch.hub.load('./', 'x3d_l', source='local', pretrained=True) # x3d_l x3d_m x3d_s x3d_xs
# model = model.eval()
# model = model.to(device)
# inputs = torch.ones([1, 3, 5, 224, 224]) # bs, channel, frames, H, W
# preds = model(inputs)
# print(preds.shape)
# print(preds[0][:10])

# import torch
# device = "cpu"
# model = torch.hub.load('./', 'i3d_r50', source='local', head_pool_kernel_size=(1, 7, 7), pretrained=True)
# model = model.eval()
# model = model.to(device)
# inputs = torch.ones([1, 3, 5, 224, 224]) # bs, channel, frames, H, W
# preds = model(inputs)
# print(preds.shape)
# print(preds[0][:10])
# mvit_base_16,
# mvit_base_16x4,
# mvit_base_32x3,

# # # for mvit_base_16, change "head_num_classes: int = 400" to "head_num_classes: int = 1000" in line 224 of models/vision_transformers.py

# # # since we input 5 frames for video recommendation, for mvit_base_16x4 and mvit_base_32x3, set "temporal_size" as 6 in line 23, 33 of models/hub/vision_transformers.py

# import torch
# from pytorchvideo.models.hub import mvit_base_16, mvit_base_16x4, mvit_base_32x3
# model = 'mvit_base_32x3'
# ckpt_path = "/root/.cache/torch/hub/checkpoints/MVIT_B_32x3_f294077834.pyth" # MVIT_B_16_f292487636.pyth MVIT_B_16x4.pyth MVIT_B_32x3_f294077834.pyth
# device = "cpu"
# model = mvit_base_32x3(pretrained=False)
# model.load_state_dict(torch.load(ckpt_path), strict=False)
# model = model.eval()
# model = model.to(device)
# inputs = torch.ones([2, 3, 5, 224, 224]) # bs, channel, frames, H, W
# preds = model(inputs)
# print(preds.shape)
# print(preds[0][:10])