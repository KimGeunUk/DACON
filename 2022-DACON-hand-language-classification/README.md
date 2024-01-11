# DACON-2022-hand-language-classification

## 수화 이미지 분류 경진대회

- 0부터 10까지 수화 이미지를 통해 숫자를 분류

- 리더보드 Private 7th, 0.98148 (7/421, 1.6%)

</br>

## Development Environmnet

- Ubuntu 18.04

- GTX 3090 1EA

</br>

## Solution

- model : tf_efficientnet_b3_ns
- image size : 512
- image augmentation
- loss : focal loss
- StratifiedKFold (5 Fold)
- scheduler : CosineAnnealingLR
- optimizer : AdamW
