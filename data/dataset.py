import cv2

import albumentations as A
import albumentations.pytorch.transforms as Apt

import torch
from torch.utils.data import Dataset

def get_albumentation(img_size: int, transforms: bool):
    if transforms:
        return A.Compose(
            [
                A.Resize(img_size, img_size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.CLAHE(p=0.5),
                A.RandomBrightnessContrast(p=0.5),
                A.RandomRotate90(p=0.5),
                A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                Apt.ToTensorV2(p=1.0),   
            ])
    else:
        return A.Compose(
            [      
                A.Resize(img_size, img_size),        
                A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                Apt.ToTensorV2(p=1.0),       
            ])

class CustomDataset(Dataset):
    def __init__(self, img_paths, labels, mode, img_size, transforms):
        self.img_paths = img_paths
        self.label_paths = labels
        self.mode = mode
        self.img_size = img_size
        self.transforms = transforms
        
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, i):
        img = cv2.imread(self.img_paths[i], cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        albumentation = get_albumentation(self.img_size, transforms=self.transforms)
        img = albumentation(image=img)["image"]
        
        if self.mode == 'train':
            label = self.label_paths[i]
            return img, torch.tensor(label, dtype=torch.long)
        else:
            return img