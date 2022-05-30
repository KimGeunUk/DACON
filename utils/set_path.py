import os
from glob import glob
import pandas as pd

TRAIN_IMG_PATH = "../../DACON/hand_language/datasets/train"
TEST_IMG_PATH = "../../DACON/hand_language/datasets/test"

TRAIN_CSV_PATH = "../../DACON/hand_language/datasets/train.csv"
TEST_CSV_PATH = "../../DACON/hand_language/datasets/test.csv"
SUBMISSION_CSV_PATH = "../../DACON/hand_language/datasets/sample_submission.csv"

train_csv = pd.read_csv(TRAIN_CSV_PATH)

train_labels = train_csv["label"]

train_labels[train_labels == '10-1'] = 10 ## label : 10-1 -> 10
train_labels[train_labels == '10-2'] = 0 ## Label : 10-2 -> 0
LABELS = train_labels.apply(lambda x : int(x)) ## Dtype : object -> int

TRAIN_IMG_PATHS = sorted(glob(TRAIN_IMG_PATH + "/*.png"))
TEST_IMG_PATHS = sorted(glob(TEST_IMG_PATH + "/*.png"))
TRAIN_IMG_LABELS = LABELS

MODEL_SAVE_PATH = "../../DACON/hand_language/save_models/"