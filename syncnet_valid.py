#!/usr/bin/env python3
"""
Train SyncNet with an automatic train/validation split and per-epoch
validation loss reporting.

Changes vs. original
---------------------
* Adds --val_split flag (default 0.10) to carve out a validation set
* Prints avg train loss **and** val loss every epoch
* Saves `best.pth` whenever val loss improves
* Fixes missing `optimizer.zero_grad()` bug
"""

import os, cv2, random, argparse
import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split


class Dataset(object):
    def __init__(self, dataset_dir, mode):
        self.img_path_list = []
        self.lms_path_list = []
        for i in range(len(os.listdir(f"{dataset_dir}/full_body_img/"))):
            self.img_path_list.append(f"{dataset_dir}/full_body_img/{i}.jpg")
            self.lms_path_list.append(f"{dataset_dir}/landmarks/{i}.lms")

        audio_feats_path = (f"{dataset_dir}/aud_wenet.npy" if mode == "wenet"
                            else f"{dataset_dir}/aud_hu.npy")
        self.mode = mode
        self.audio_feats = np.load(audio_feats_path).astype(np.float32)

    def __len__(self):
        return self.audio_feats.shape[0] - 1

    def get_audio_features(self, features, index):
        
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
    
    def process_img(self, img, lms_path, img_ex, lms_path_ex):

        lms_list = []
        with open(lms_path, "r") as f:
            lines = f.read().splitlines()
            for line in lines:
                arr = line.split(" ")
                arr = np.array(arr, dtype=np.float32)
                lms_list.append(arr)
        lms = np.array(lms_list, dtype=np.int32)
        xmin = lms[1][0]
        ymin = lms[52][1]
        
        xmax = lms[31][0]
        width = xmax - xmin
        ymax = ymin + width
        
        crop_img = img[ymin:ymax, xmin:xmax]
        crop_img = cv2.resize(crop_img, (168, 168), cv2.INTER_AREA)
        img_real = crop_img[4:164, 4:164].copy()
        img_real_ori = img_real.copy()
        img_real_ori = img_real_ori.transpose(2,0,1).astype(np.float32)
        img_real_T = torch.from_numpy(img_real_ori / 255.0)
        
        return img_real_T

    def __getitem__(self, idx):
        img = cv2.imread(self.img_path_list[idx])
        lms_path = self.lms_path_list[idx]
        
        ex_int = random.randint(0, self.__len__()-1)
        img_ex = cv2.imread(self.img_path_list[ex_int])
        lms_path_ex = self.lms_path_list[ex_int]
        
        img_real_T = self.process_img(img, lms_path, img_ex, lms_path_ex)
        audio_feat = self.get_audio_features(self.audio_feats, idx) # 
        # print(audio_feat.shape)
        if self.mode=="wenet":
            audio_feat = audio_feat.reshape(256,16,32)
        if self.mode=="hubert":
            audio_feat = audio_feat.reshape(32,32,32)
        y = torch.ones(1).float()
        
        return img_real_T, audio_feat, y


class Conv2d(nn.Module):
    def __init__(self, cin, cout, kernel_size, stride, padding, residual=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_block = nn.Sequential(
                            nn.Conv2d(cin, cout, kernel_size, stride, padding),
                            nn.BatchNorm2d(cout)
                            )
        self.act = nn.ReLU()
        self.residual = residual

    def forward(self, x):
        out = self.conv_block(x)
        if self.residual:
            out += x
        return self.act(out)

class nonorm_Conv2d(nn.Module):
    def __init__(self, cin, cout, kernel_size, stride, padding, residual=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_block = nn.Sequential(
                            nn.Conv2d(cin, cout, kernel_size, stride, padding),
                            )
        self.act = nn.LeakyReLU(0.01, inplace=True)

    def forward(self, x):
        out = self.conv_block(x)
        return self.act(out)

class Conv2dTranspose(nn.Module):
    def __init__(self, cin, cout, kernel_size, stride, padding, output_padding=0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_block = nn.Sequential(
                            nn.ConvTranspose2d(cin, cout, kernel_size, stride, padding, output_padding),
                            nn.BatchNorm2d(cout)
                            )
        self.act = nn.ReLU()

    def forward(self, x):
        out = self.conv_block(x)
        return self.act(out)


class SyncNet_color(nn.Module):
    def __init__(self, mode):
        super(SyncNet_color, self).__init__()

        self.face_encoder = nn.Sequential(
            Conv2d(3, 32, kernel_size=(7, 7), stride=1, padding=3),

            Conv2d(32, 64, kernel_size=5, stride=2, padding=1),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=0),
            Conv2d(512, 512, kernel_size=1, stride=1, padding=0),)
        
        p1 = 256
        p2 = (1, 2)
        if mode == "hubert":
            p1 = 32
            p2 = (2, 2)
        
        self.audio_encoder = nn.Sequential(
            Conv2d(p1, 256, kernel_size=3, stride=1, padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            
            Conv2d(256, 256, kernel_size=3, stride=p2, padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(256, 256, kernel_size=3, stride=2, padding=2),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(512, 512, kernel_size=3, stride=1, padding=0),
            Conv2d(512, 512, kernel_size=1, stride=1, padding=0),)

    def forward(self, face_sequences, audio_sequences): # audio_sequences := (B, dim, T)
        face_embedding = self.face_encoder(face_sequences)
        audio_embedding = self.audio_encoder(audio_sequences)

        audio_embedding = audio_embedding.view(audio_embedding.size(0), -1)
        face_embedding = face_embedding.view(face_embedding.size(0), -1)

        audio_embedding = F.normalize(audio_embedding, p=2, dim=1)
        face_embedding = F.normalize(face_embedding, p=2, dim=1)
        
        return audio_embedding, face_embedding


# ──────────────────────────────  Loss helpers  ────────────────────────────── #

# Quick fix for: Assertion `input_val >= zero && input_val <= one` failed
logloss = nn.BCELoss()
def cosine_loss(a, v, y):
    p = torch.clamp((F.cosine_similarity(a, v).unsqueeze(1) + 1) * 0.5,
                    1e-7, 1 - 1e-7)
    return logloss(p, y)

# logloss = nn.BCELoss()
# def cosine_loss(a, v, y):
    # d = F.cosine_similarity(a, v)
    # return logloss(d.unsqueeze(1), y)

@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    total = 0.0
    for imgT, audioT, y in loader:
        imgT, audioT, y = imgT.cuda(), audioT.cuda(), y.cuda()
        a_emb, v_emb = model(imgT, audioT)
        total += cosine_loss(a_emb, v_emb, y).item()
    return total / len(loader)

# ────────────────────────────────  Training  ──────────────────────────────── #
def train(opts):
    os.makedirs(opts.save_dir, exist_ok=True)

    full_set = Dataset(opts.dataset_dir, opts.asr)
    val_len = int(len(full_set) * opts.val_split)
    train_len = len(full_set) - val_len
    train_set, val_set = random_split(
        full_set, [train_len, val_len],
        generator=torch.Generator().manual_seed(opts.seed))

    train_loader = DataLoader(train_set,  batch_size=opts.batch_size,
                              shuffle=True,  num_workers=opts.num_workers)
    val_loader   = DataLoader(val_set,    batch_size=opts.batch_size,
                              shuffle=False, num_workers=opts.num_workers)

    model = SyncNet_color(opts.asr).cuda()
    opt   = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                       lr=opts.lr)

    best_val = float("inf")
    for epoch in range(opts.epochs):
        model.train()
        running = 0.0
        for imgT, audioT, y in train_loader:
            imgT, audioT, y = imgT.cuda(), audioT.cuda(), y.cuda()
            opt.zero_grad()
            a_emb, v_emb = model(imgT, audioT)
            loss = cosine_loss(a_emb, v_emb, y)
            loss.backward()
            opt.step()
            running += loss.item()

        train_loss = running / len(train_loader)
        val_loss   = evaluate(model, val_loader)
        print(f"{epoch:02d}  train={train_loss:.6e}  val={val_loss:.6e}")

        torch.save(model.state_dict(),
                   os.path.join(opts.save_dir, f"{epoch}.pth"))
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(),
                       os.path.join(opts.save_dir, "best.pth"))

# ──────────────────────────────  CLI glue  ───────────────────────────────── #
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--save_dir',     required=True)
    p.add_argument('--dataset_dir',  required=True)
    p.add_argument('--asr',          choices=['wenet', 'hubert'], required=True)
    p.add_argument('--epochs',       type=int, default=50)
    p.add_argument('--batch_size',   type=int, default=16)
    p.add_argument('--lr',           type=float, default=1e-3)
    p.add_argument('--val_split',    type=float, default=0.1,
                   help="Fraction of data reserved for validation.")
    p.add_argument('--num_workers',  type=int, default=4)
    p.add_argument('--seed',         type=int, default=42)
    opts = p.parse_args()

    train(opts)
