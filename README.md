# Computer Vision 이상 탐지(Anomaly Detection) 알고리즘 경진대회

Mvtec Ad Dataset 사용, 사물의 종류 분류 및 정상 샘플 비정상 샘플 분류

- 주제
사물의 종류와 상태를 분류하는 컴퓨터 비전 알고리즘 개발

- 주최/주관
 DACON

- 참가 대상
일반인, 학생, 직장인 등 누구나

  [DACON](https://dacon.io/competitions/official/235894/overview/description)

- 리더보드 PRIVATE 16th, 0.87786 (16/481, 3.5%) 

<hr/>

## Development Environmnet

- Window 10 & Colab

- GTX 1650 Ti

<hr/>

## Solution

- Data split -> train : valid = 9 : 1 
- Fine Tuned timm tf_efficientnet_b7_ns
- Image Size : 512
- Image nomalization : mean = [0.43324712, 0.40364919, 0.39435242], std = [0.18257473, 0.17486729, 0.16405263]
- Image Augmentation : 
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
        A.Normalize(mean_train, std_train),
        Apt.ToTensorV2(p=1.0),
