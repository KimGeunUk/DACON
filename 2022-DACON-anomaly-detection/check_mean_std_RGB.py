import numpy as np
from tqdm import tqdm
import cv2
from glob import glob
import matplotlib.pyplot as plt
import albumentations as A
import albumentations.pytorch.transforms as Apt
import torchvision.transforms as transforms

def img_load(path, img_size):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.uint8)
    img = cv2.resize(img, (img_size, img_size))
    return img
   
paths = sorted(glob('D:/STUDY/DACON/anomaly-detection/data/test/*.png'))
img_size = 512

imgs = [img_load(m, img_size) for m in tqdm(paths)]

imgs = np.array(imgs)

meanRGB = [np.mean(x, axis=(0,1)) for x in imgs]
stdRGB = [np.std(x, axis=(0,1)) for x in imgs]

meanR = '{:.8}'.format(np.mean([m[0] for m in meanRGB])/255.)
meanG = '{:.8}'.format(np.mean([m[1] for m in meanRGB])/255.)
meanB = '{:.8}'.format(np.mean([m[2] for m in meanRGB])/255.)

stdR = '{:.8}'.format(np.mean([s[0] for s in stdRGB])/255.)
stdG = '{:.8}'.format(np.mean([s[1] for s in stdRGB])/255.)
stdB = '{:.8}'.format(np.mean([s[2] for s in stdRGB])/255.)

print("평균", meanR, meanG, meanB)
print("표준편차", stdR, stdG, stdB)

mean = [meanR, meanG, meanB]
std = [stdR, stdG, stdB]

print(mean, std)
