import argparse
import os
import cv2
import torch
import numpy as np
import torch.nn as nn
from torch import optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from unet import Model
# from unet2 import Model
# from unet_att import Model

import time
parser = argparse.ArgumentParser(description='Train',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--asr', type=str, default="hubert")
parser.add_argument('--dataset', type=str, default="")  
parser.add_argument('--audio_feat', type=str, default="")
parser.add_argument('--save_path', type=str, default="")     # end with .mp4 please
parser.add_argument('--checkpoint', type=str, default="")
parser.add_argument('--debug', type=bool, default=False)
args = parser.parse_args()

checkpoint = args.checkpoint
save_path = args.save_path
debug = args.debug
if debug:
    if save_path.endswith(".mp4"):
        debug_dir = os.path.join(os.path.dirname(save_path), 'debug')
    else:
        raise ValueError('save path must end with .mp4 as video')
dataset_dir = args.dataset
audio_feat_path = args.audio_feat
mode = args.asr

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# def get_audio_features(features, index): # 这个逻辑跟datasets里面的逻辑相同
    # left = index - 4
    # right = index + 4
    # pad_left = 0
    # pad_right = 0
    # if left < 0:
        # pad_left = -left
        # left = 0
    # if right > features.shape[0]:
        # pad_right = right - features.shape[0]
        # right = features.shape[0]
    # auds = torch.from_numpy(features[left:right])
    # if pad_left > 0:
        # auds = torch.cat([torch.zeros_like(auds[:pad_left]), auds], dim=0)
    # if pad_right > 0:
        # auds = torch.cat([auds, torch.zeros_like(auds[:pad_right])], dim=0) # [8, 16]
    # return auds

# From Owen
def get_audio_features(features, index):  # 在当前音频帧前后各取4帧音频特征
    left = index - 8
    right = index + 8
    pad_left = 0
    pad_right = 0
    if left < 0:
        pad_left = -left
        left = 0
    if right > features.shape[0]:
        pad_right = right - features.shape[0]
        right = features.shape[0]
    auds = torch.from_numpy(features[left:right])
    if pad_left > 0:
        auds = torch.cat([torch.zeros_like(auds[:pad_left]), auds], dim=0)
    if pad_right > 0:
        auds = torch.cat([auds, torch.zeros_like(auds[:pad_right])], dim=0) # [8, 16]
    return auds

audio_feats = np.load(audio_feat_path)
img_dir = os.path.join(dataset_dir, "full_body_img/")
lms_dir = os.path.join(dataset_dir, "landmarks/")
len_img = len(os.listdir(img_dir)) - 1
exm_img = cv2.imread(img_dir+"0.jpg")
h, w = exm_img.shape[:2]

if mode=="hubert":
    video_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc('M','J','P', 'G'), 25, (w, h))
if mode=="wenet":
    video_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc('M','J','P', 'G'), 20, (w, h))
step_stride = 0
img_idx = 0

net = Model(6, mode).to(device)
net.load_state_dict(torch.load(checkpoint, map_location=device))
net.eval()
for i in tqdm(range(audio_feats.shape[0]), desc="Writing video"):
    if img_idx>len_img - 1:
        step_stride = -1  # step_stride 决定取图片的间隔，目前这个逻辑是从头开始一张一张往后，到最后一张后再一张一张往前
    if img_idx<1:
        step_stride = 1
    img_idx += step_stride
    img_path = img_dir + str(img_idx)+'.jpg'
    lms_path = lms_dir + str(img_idx)+'.lms'
    
    img = cv2.imread(img_path)
    lms_list = []
    with open(lms_path, "r") as f:
        lines = f.read().splitlines()
        for line in lines:
            arr = line.split(" ")
            arr = np.array(arr, dtype=np.float32)
            lms_list.append(arr)
    lms = np.array(lms_list, dtype=np.int32)  # 这个关键点检测模型之后之后可能会改掉
    xmin = lms[1][0]
    ymin = lms[52][1]

    xmax = lms[31][0]
    width = xmax - xmin
    ymax = ymin + width
    if debug:
        draw_crop_region = cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 0, 255), 3)
        os.makedirs(os.path.join(debug_dir, 'draw_crop_region'), exist_ok=True)
        cv2.imwrite(f"{debug_dir}/draw_crop_region/{i}.png", draw_crop_region)
        
    crop_img = img[ymin:ymax, xmin:xmax]
    h, w = crop_img.shape[:2]
    crop_img = cv2.resize(crop_img, (168, 168), cv2.INTER_AREA)
    crop_img_ori = crop_img.copy()
    img_real_ex = crop_img[4:164, 4:164].copy()
    img_real_ex_ori = img_real_ex.copy()
    img_masked = cv2.rectangle(img_real_ex_ori,(5,5,150,145),(0,0,0),-1)
    if debug:
        os.makedirs(os.path.join(debug_dir, 'img_masked'), exist_ok=True)
        cv2.imwrite(f"{debug_dir}/img_masked/{i}.png", img_masked)
    
    img_masked = img_masked.transpose(2,0,1).astype(np.float32)
    img_real_ex = img_real_ex.transpose(2,0,1).astype(np.float32)
    
    img_real_ex_T = torch.from_numpy(img_real_ex / 255.0).to(device)
    img_masked_T = torch.from_numpy(img_masked / 255.0).to(device)  
    img_concat_T = torch.cat([img_real_ex_T, img_masked_T], axis=0)[None]
    # if debug:
        # os.makedirs(os.path.join(debug_dir, 'img_concat_T'), exist_ok=True)
        # cv2.imwrite(f"{debug_dir}/img_concat_T/{i}.png", img_concat_T)
    # 这个地方逻辑和dataset里面完全一样，只是不需要另外取一张参考图 而是用要推理的这张图片即可
    
    audio_feat = get_audio_features(audio_feats, i)
    if mode=="hubert":
        audio_feat = audio_feat.reshape(32,32,32)
    if mode=="wenet":
        audio_feat = audio_feat.reshape(128,16,32)
    audio_feat = audio_feat[None]
    audio_feat = audio_feat.to(device)
    img_concat_T = img_concat_T.to(device)
    
    with torch.no_grad():
        pred = net(img_concat_T, audio_feat)[0]
        
    pred = pred.cpu().numpy().transpose(1,2,0)*255
    pred = np.array(pred, dtype=np.uint8)
    crop_img_ori[4:164, 4:164] = pred
    crop_img_ori = cv2.resize(crop_img_ori, (w, h))
    if debug:
        os.makedirs(os.path.join(debug_dir, 'prediction_resized'), exist_ok=True)
        cv2.imwrite(f"{debug_dir}/prediction_resized/{i}.png", crop_img_ori)
    img[ymin:ymax, xmin:xmax] = crop_img_ori
    video_writer.write(img)
video_writer.release()

# ffmpeg -i test_video.mp4 -i test_audio.pcm -c:v libx264 -c:a aac result_test.mp4
import subprocess

if mode=="hubert":
    audio_path = audio_feat_path.replace('_hu.npy', '.wav')
elif mode=="wenet":
    audio_path = audio_feat_path.replace('_wenet.npy', '.wav')
else:
    raise NotImplementedError(f"unknown audio feature extractor '{mode}'")

output_path = save_path.replace('.mp4', '_.mp4')

# FFmpeg command
cmd = [
    "ffmpeg",
    "-i", save_path,
    "-i", audio_path,
    "-c:v", "libx264",
    "-c:a", "aac",
    output_path
]

subprocess.run(cmd, check=True)
