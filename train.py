import warnings
warnings.filterwarnings('ignore')

import argparse
import math
import os
import gc
import random
import sys
import time
import cv2
import timm
import pandas as pd
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from glob import glob
from sklearn.metrics import f1_score

import wandb
import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.cuda import amp
# import torch.distributed as dist
# import torchvision.transforms as transforms
#from torch.optim import SGD, Adam, AdamW, lr_scheduler
from torch.utils.data import Dataset
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from transformers.optimization import AdamW, get_cosine_schedule_with_warmup

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
seed_everything(22)

os.chdir(os.path.dirname(os.path.realpath(__file__)))

# GET DATA
train_img_paths = sorted(glob('data/train/*.png'))
test_img_paths = sorted(glob('data/test/*.png'))

train_y = pd.read_csv('data/train_df.csv')
train_labels = train_y["label"]

label_encoder = sorted(np.unique(train_labels))
label_encoder = {key:value for key,value in zip(label_encoder, range(len(label_encoder)))}
label_decoder = {value:key for key,value in label_encoder.items()}

train_labels = [label_encoder[k] for k in train_labels]

import albumentations as A
import albumentations.pytorch.transforms as Apt

def trainAlbumentation():
    if opt.img_size == 640:
        mean = [0.43306863, 0.40349263, 0.39418206]
        std = [0.1821309, 0.17450397, 0.16369207]
    elif opt.img_size == 512:
        mean = [0.43324712, 0.40364919, 0.39435242]
        std = [0.18257473, 0.17486729, 0.16405263]
    elif opt.img_size == 384:
        mean = [0.43272011, 0.40312686, 0.39382899]
        std = [0.18284354, 0.175067, 0.16427512]
    elif opt.img_size == 256:
        mean = [0.43324574, 0.40365004, 0.39434799]
        std = [0.18257991, 0.17486855, 0.16405118]
    elif opt.img_size == 224:
        mean = [0.43265206, 0.40305811, 0.39374737]
        std = [0.18287216, 0.17509091, 0.16429985]
        
    transform = A.Compose([
        A.Resize(opt.img_size, opt.img_size),
        A.CLAHE(p=opt.percentage),
        A.RandomBrightnessContrast(p=opt.percentage),
        A.ColorJitter(p=opt.percentage),
        A.RGBShift(p=opt.percentage),
        A.RandomSnow(p=opt.percentage),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomResizedCrop(height=opt.img_size, width=opt.img_size, p=opt.percentage),
        A.ShiftScaleRotate(shift_limit=0.5, scale_limit=0.5, rotate_limit=0, interpolation=0, border_mode=4, p=opt.percentage),
        A.Rotate(p=opt.percentage),
        A.RandomRotate90(p=opt.percentage),    
        A.Normalize(mean, std),
        Apt.ToTensorV2(p=1.0),   
    ], p=1.)        
    return transform

def testAlbumentation():
    if opt.img_size == 640:
        mean = [0.41829, 0.39314, 0.3866658]
        std = [0.19555221, 0.19046031, 0.18092135]
    elif opt.img_size == 512:
        mean = [0.4184459, 0.39327951, 0.38681376]
        std = [0.19594851, 0.19077888, 0.18122797]
    elif opt.img_size == 384:
        mean = [0.41794201, 0.3927824, 0.38631485]
        std = [0.19614866, 0.19092344, 0.18138667]
    elif opt.img_size == 256:
        mean = [0.41845142, 0.39328347, 0.38681326]
        std = [0.19595109, 0.19077791, 0.18122525]
    elif opt.img_size == 256:
        mean = [0.41787571, 0.39271571, 0.38624564]
        std = [0.19616894, 0.19093925, 0.18140095]
        
    transform = A.Compose([        
        A.Resize(opt.img_size, opt.img_size),        
        A.Normalize(mean, std),
        Apt.ToTensorV2(p=1.0),       
    ], p=1.)        
    return transform

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2

def cutmix(data, target, alpha):
    indices = torch.randperm(data.size(0))
    shuffled_data = data[indices]
    shuffled_target = target[indices]

    lam = np.clip(np.random.beta(alpha, alpha),0.3,0.4)
    bbx1, bby1, bbx2, bby2 = rand_bbox(data.size(), lam)
    new_data = data.clone()
    new_data[:, :, bby1:bby2, bbx1:bbx2] = data[indices, :, bby1:bby2, bbx1:bbx2]
    # adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (data.size()[-1] * data.size()[-2]))
    targets = (target, shuffled_target, lam)

    return new_data, targets

class CustomDataset(Dataset):
    def __init__(self, img_paths, label_paths, mode):
        self.img_paths = img_paths
        self.label_paths = label_paths
        self.mode = mode
        
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, i):
        #img = img_load(self.img_paths[i])
        #img = cv2.imread(self.img_paths[i])
        img = cv2.imread(self.img_paths[i], cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.uint8)
        
        if self.mode == 'train':            
            transform = trainAlbumentation()
            img = transform(image=img)['image']
        else: 
            transform = testAlbumentation()
            img = transform(image=img)['image']
            
        label = self.label_paths[i]
        
        return img, label

class Network(nn.Module):
    def __init__(self, opt):
        super(Network, self).__init__()
        self.model = timm.create_model(f'{opt.model}', pretrained=True, num_classes=opt.num_class)
    
    def forward(self, inputs):
        output = self.model(inputs)
        return output

def score_function(real, pred):
    score = f1_score(real, pred, average="macro")
    return score

class CustomTrainer:
    def __init__(self, model):
        self.model = model        
        
        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=opt.label_smoothing)
        self.scaler = torch.cuda.amp.GradScaler()
        self.epochs = opt.epochs
        
        self.optimizer = AdamW(self.model.parameters(), lr=opt.learning_rate)
        total_steps = int(len(train_dataset) * opt.epochs / opt.batch_size)
        warmup_steps = 1149
        self.scheduler = get_cosine_schedule_with_warmup(self.optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
        
        wandb.watch(model)
        self.best_score = 0.0  

    def run(self, train_dataloader, val_dataloader, test_dataloader):
        for epoch in range(self.epochs):
            start=time.time()
            gc.collect()
            
            lr = self.optimizer.param_groups[0]['lr']
            print(f'=============== lr:{lr:.6f}, epoch:{epoch+1} ===============')
            train_loss, train_score = self.train_f(train_dataloader, epoch)
                      
            TIME = time.time() - start
            print(f'time : {TIME:.0f}s/{TIME*(self.epochs-epoch-1):.0f}s, train_loss: {train_loss:.6f}, train_score: {train_score:.6f}')         
            print(f'====================================================')
            print(' ') 
            print(f'==================== val, epoch:{epoch+1} ===================')
            with torch.no_grad():
                val_loss, val_score = self.val_f(val_dataloader)
                
            print(f'val_loss: {val_loss:.6f}, val_score: {val_score:.6f}, best_score: {self.best_score:.6f}')
            wandb.log({f"{opt.model}/train_loss": train_loss, f"{opt.model}/train_score": train_score, f"{opt.model}/val_loss": val_loss, f"{opt.model}/val_score": val_score})
            
            if val_score >= 0.85 and val_score >= self.best_score:                
                self.best_score = val_score
                best_model = self.model
                best_epoch = epoch+1
                torch.save(model.state_dict(), f'{save_dir}/{opt.model}_{epoch+1}_{self.best_score:.6f}.pth')
                print(f'Model is Saved when epoch is : {epoch+1}')
            print(f'====================================================')
            print(' ')
        print(f'BEST SCORE : {self.best_score:.6f}, EPOCH : {best_epoch}' )    
        self.model = best_model
        pred = self.test_f(test_dataloader)
        return pred
    
    def train_f(self, train_dataloader, epoch):
        self.model.train()
        
        train_loss = 0
        train_output, train_label = [], []
        
        for batch in tqdm(train_dataloader, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}'):
            img = torch.tensor(batch[0], dtype=torch.float32, device=device)
            label = torch.tensor(batch[1], dtype=torch.long, device=device)

            self.optimizer.zero_grad()
            
            if opt.cutmix != False and epoch < self.epochs-10:
                mix_decision = np.random.rand()
                if mix_decision < 0.3:
                    img, mix_labels = cutmix(img, label, 1.0)
                else:
                    pass
            else: mix_decision = 1
            
            if self.epochs-10 <= epoch:
                assert mix_decision == 1
            
            with torch.cuda.amp.autocast():
                output = model(img)
                if mix_decision < 0.3:
                    loss = self.loss_fn(output, mix_labels[0])*(mix_labels[2]) + self.loss_fn(output, mix_labels[1])*(1-mix_labels[2])
                else:
                    loss = self.loss_fn(output, label)
            
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()            
            self.scheduler.step()
            
            train_loss += loss.item() / len(train_dataloader)
            train_output += output.argmax(1).detach().cpu().numpy().tolist()
            train_label += label.detach().cpu().numpy().tolist()
        
        train_score = score_function(train_label, train_output)
        return train_loss, train_score
    
    def val_f(self, val_dataloader):
        self.model.eval()
        
        val_loss = 0
        val_output, val_label =[], []
        for batch in tqdm(val_dataloader, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}'):
            img = torch.tensor(batch[0], dtype=torch.float32, device=device)
            label = torch.tensor(batch[1], dtype=torch.long, device=device)
            
            output = self.model(img)
            loss = self.loss_fn(output, label)
            
            val_loss += loss.item() / len(val_dataloader)
            val_output += output.argmax(1).detach().cpu().numpy().tolist()
            val_label += label.detach().cpu().numpy().tolist()
        
        
        val_score = score_function(val_label, val_output)
        return val_loss, val_score
    
    def test_f(self, test_dataloader):
        model.eval()
    
        pred = []
        with torch.no_grad():
            for batch in tqdm(test_dataloader, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}'):
                img = torch.tensor(batch[0], dtype = torch.float32, device = device)
                with torch.cuda.amp.autocast():
                    output = model(img)
                pred.extend(output.argmax(1).detach().cpu().numpy().tolist())
        
        return pred
                
def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='', help='resnet50 efficientnet_b3 densenet201 eca_vovnet39b resnet18 res2net50_26w_4s resnext50_32x4d')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--img-size', type=int, default=512)
    parser.add_argument('--num-class', type=int, default=88)
    parser.add_argument('--learning-rate', type=float, default=2e-4)
    parser.add_argument('--cutmix', type=bool, default=True)
    parser.add_argument('--label-smoothing', type=float, default=0.1)
    parser.add_argument('--percentage', type=float, default=0.2)
    
    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt
    
if __name__ == "__main__":
    opt = parse_opt()    
    
    project = f'{opt.model}-aug{opt.percentage}-cutmix{opt.cutmix}-{opt.img_size}-{opt.epochs}-{opt.learning_rate}'
    wandb.init(project="anomaly-detection",
            group="baseline",
            name=f'{project}',
            entity="kgw5430")
    
    save_dir = f'models/{project}'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    device = torch.device('cuda:0')
    
    train_img_pathss, val_img_paths, train_label_pathss, val_label_paths = train_test_split(train_img_paths, train_labels, test_size=0.1, shuffle=True, stratify=train_labels, random_state=22)
 
    train_dataset = CustomDataset(np.array(train_img_pathss), np.array(train_label_pathss), mode='train')
    val_dataset = CustomDataset(np.array(val_img_paths), np.array(val_label_paths), mode='test')
    test_dataset = CustomDataset(np.array(test_img_paths), np.array(["tmp"]*len(test_img_paths)), mode='test')
    
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False)
    
    model = Network(opt).to(device)
    
    custom_trainer = CustomTrainer(model=model)
    pred = custom_trainer.run(train_dataloader, val_dataloader, test_dataloader)
    
    f_result = [label_decoder[result] for result in pred]
    submission = pd.read_csv("data/sample_submission.csv")
    submission["label"] = f_result
    submission.to_csv(f"{save_dir}/{project}.csv", index = False)