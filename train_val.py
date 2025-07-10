import argparse
import os
import cv2
import torch
import numpy as np
import torch.nn as nn
from torch import optim
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset
from datasetsss import MyDataset
from syncnet import SyncNet_color
from unet import Model
import random
import torchvision.models as models


device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_args():
    parser = argparse.ArgumentParser(description='Train',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--use_syncnet', action='store_true', help="if use syncnet, you need to set 'syncnet_checkpoint'")
    parser.add_argument('--syncnet_checkpoint', type=str, default="")
    parser.add_argument('--dataset_dir', type=str)
    parser.add_argument('--save_dir', type=str, help="trained model save path.")
    parser.add_argument('--see_res', action='store_true')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batchsize', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--asr', type=str, default="hubert")
    parser.add_argument('--val_split', type=float, default=0.2, help='Fraction of data reserved for validation.')
    parser.add_argument('--eval_interval', type=int, default=10, help='Run validation every N epochs')

    return parser.parse_args()

args = get_args()
use_syncnet = args.use_syncnet

class PerceptualLoss():
    def contentFunc(self):
        conv_3_3_layer = 14
        cnn = models.vgg19(pretrained=True).features.to(device)
        model = nn.Sequential().to(device)
        for i, layer in enumerate(list(cnn)):
            model.add_module(str(i), layer)
            if i == conv_3_3_layer:
                break
        return model

    def __init__(self, loss):
        self.criterion = loss
        self.contentFunc = self.contentFunc()

    def get_loss(self, fakeIm, realIm):
        f_fake = self.contentFunc.forward(fakeIm)
        f_real = self.contentFunc.forward(realIm)
        loss = self.criterion(f_fake, f_real.detach())
        return loss

logloss = nn.BCELoss()
def cosine_loss(a, v, y):
    d = nn.functional.cosine_similarity(a, v)
    return logloss(d.unsqueeze(1), y)

@torch.no_grad()
def evaluate(model, val_loader, loss_fn, content_loss, syncnet=None):
    model.eval()
    total_loss = 0.0
    for imgs, labels, audio_feat in val_loader:
        imgs = imgs.to(device)
        labels = labels.to(device)
        audio_feat = audio_feat.to(device)
        preds = model(imgs, audio_feat)
        pixel_loss = loss_fn(preds, labels)
        perceptual = content_loss.get_loss(preds, labels)
        loss = pixel_loss + 0.01 * perceptual
        if syncnet:
            y = torch.ones([preds.shape[0], 1]).float().to(device)
            a, v = syncnet(preds, audio_feat)
            loss += 10 * cosine_loss(a, v, y)
        total_loss += loss.item()
    return total_loss / len(val_loader)

def train(net, epoch, batch_size, lr):
    content_loss = PerceptualLoss(torch.nn.MSELoss())
    syncnet = None
    if use_syncnet:
        if args.syncnet_checkpoint == "":
            raise ValueError("Using syncnet, you need to set 'syncnet_checkpoint'. Please check README")
        syncnet = SyncNet_color(args.asr).eval().to(device)
        syncnet.load_state_dict(torch.load(args.syncnet_checkpoint))

    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)

    dataset = MyDataset(args.dataset_dir, args.asr)
    val_len = int(len(dataset) * args.val_split)
    indices = list(range(len(dataset)))
    train_indices = indices[:-val_len]
    val_indices = indices[-val_len:]
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, drop_last=False, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, drop_last=False, num_workers=4)

    optimizer = optim.Adam(net.parameters(), lr=lr)
    criterion = nn.L1Loss()

    best_val = float('inf')
    for e in range(epoch):
        net.train()
        with tqdm(total=len(train_dataset), desc=f'Epoch {e + 1}/{epoch}', unit='img') as p:
            for batch in train_loader:
                imgs, labels, audio_feat = batch
                imgs = imgs.to(device)
                labels = labels.to(device)
                audio_feat = audio_feat.to(device)
                preds = net(imgs, audio_feat)
                y = torch.ones([preds.shape[0], 1]).float().to(device) if use_syncnet else None
                loss_PerceptualLoss = content_loss.get_loss(preds, labels)
                loss_pixel = criterion(preds, labels)
                loss = loss_pixel + loss_PerceptualLoss * 0.01
                if use_syncnet:
                    a, v = syncnet(preds, audio_feat)
                    sync_loss = cosine_loss(a, v, y)
                    loss += 10 * sync_loss

                p.set_postfix(**{'loss (batch)': loss.item()})
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                p.update(imgs.shape[0])

        if (e + 1) % args.eval_interval == 0:
            val_loss = evaluate(net, val_loader, criterion, content_loss, syncnet)
            print(f"[Epoch {e+1}] Validation loss: {val_loss:.6f}")
            if val_loss < best_val:
                best_val = val_loss
                torch.save(net.state_dict(), os.path.join(save_dir, 'best.pth'))

        if e % 10 == 0:
            torch.save(net.state_dict(), os.path.join(save_dir, f'{e}.pth'))

        if args.see_res:
            net.eval()
            img_concat_T, img_real_T, audio_feat = dataset.__getitem__(random.randint(0, len(dataset)))
            img_concat_T = img_concat_T[None].to(device)
            audio_feat = audio_feat[None].to(device)
            with torch.no_grad():
                pred = net(img_concat_T, audio_feat)[0]
            pred = (pred.cpu().numpy().transpose(1,2,0) * 255).astype(np.uint8)
            img_real = (img_real_T.numpy().transpose(1,2,0) * 255).astype(np.uint8)
            os.makedirs("./train_tmp_img", exist_ok=True)
            cv2.imwrite(f"./train_tmp_img/epoch_{e}.jpg", pred)
            cv2.imwrite(f"./train_tmp_img/epoch_{e}_real.jpg", img_real)

if __name__ == '__main__':
    net = Model(6, args.asr).to(device)
    train(net, args.epochs, args.batchsize, args.lr)
