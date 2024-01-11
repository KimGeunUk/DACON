import warnings
warnings.filterwarnings('ignore')

import os
import gc
import wandb
import argparse
import numpy as np

import torch
from torch.optim import SGD, Adam, AdamW, lr_scheduler

from data.dataset import CustomDataset
from data.data_loader import get_data_loader

from models.runner.inference_runner import CustomInference
from models.model.pytorch_cnn import CNNclassification
from models.model.pytorch_timm import TimmModel

from utils.set_seed import seed_everything
from utils.set_path import *
from utils.focal_loss import FocalLoss

from sklearn.model_selection import StratifiedKFold

def kfold_inference(args, test_img_paths, fold_model_dir):
    test_dataset = CustomDataset(img_paths=test_img_paths, labels=None, mode='test', img_size=args.img_size, transforms=False)
    
    test_dataloader= torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers)

    # model = CNNclassification().to(device)
    model = TimmModel(args).to(device)
    
    #self.loss_fn = nn.CrossEntropyLoss(label_smoothing=opt.label_smoothing)
    loss_fn = FocalLoss(label_smoothing=args.label_smoothing)        
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, betas=args.betas, eps=args.eps, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.T_max, eta_min=args.eta_min)
    
    inferencer = CustomInference(model=model, optimizer=optimizer, scheduler=scheduler, loss_fn=loss_fn, device=device)
    preds = inferencer.test_run(test_dataloader, fold_model_dir)
    
    print("len(preds) : ", len(preds))
    return preds    
    
def arg_parse(known=False):
    parser = argparse.ArgumentParser()    
    parser.add_argument('--model', type=str, default='tf_efficientnet_b3_ns')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--img-size', type=int, default=768)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    
    parser.add_argument('--seed', type=int, default=5430)
    parser.add_argument('--num-classes', type=int, default=11)
    parser.add_argument('--label-smoothing', type=float, default=0.1)
    parser.add_argument('--num_folds', type=int, default=5)
    parser.add_argument('--num_workers', type=int, default=12)
    parser.add_argument('--patience', type=int, default=20)
    
    parser.add_argument('--betas', type=tuple, default=(0.9, 0.999))
    parser.add_argument('--eps', type=float, default=1e-8)
    parser.add_argument('--weight_decay', type=float, default=1e-3)
    parser.add_argument('--T_max', type=int, default=10)
    parser.add_argument('--eta_min', type=float, default=1e-6)
    
    args = parser.parse_known_args()[0] if known else parser.parse_args()
    return args

if __name__=='__main__':
    args = arg_parse()   
     
    seed_everything(args.seed)
    
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    save_dir = MODEL_SAVE_PATH + f"{args.model}"
    save_dir = glob(save_dir + "*")[-1]
    sub_dir = MODEL_SAVE_PATH + "sub"
    
    print(f"TRAIN IMG PATHS LEN : {len(TRAIN_IMG_PATHS)}")
    print(f"TEST IMG PATHS LEN : {len(TEST_IMG_PATHS)}")
    
    infer_results = []
    
    save_dir = sorted(glob(save_dir+"*"))[-1]
    fold_dirs = sorted(glob(save_dir+"/*"))
    
    for idx, fold_dir in enumerate(fold_dirs):
        if idx == 4:
            print(f"\t\t\tFold {str(idx+1)} IN START!!!!")
            print(' ')
                    
            fold_model_dir = sorted(glob(fold_dir+"/*"))[0]
            
            infer_result = kfold_inference(args,
                    np.array(TEST_IMG_PATHS),
                    fold_model_dir)
            infer_results.append(infer_result) 
    
    print("Soft Voting")
    # predict = (infer_results[0] + infer_results[1] + infer_results[2] + infer_results[3] + infer_results[4])
    predict = infer_results[0]
    # predict = predict / 5
    predict = [np.argmax(i) for i in predict]
    
    submission = pd.read_csv(SUBMISSION_CSV_PATH)
    submission['label'] = predict
    submission['label'][submission['label'] == 10] = '10-1' ## label : 10 -> '10-1'
    submission['label'][submission['label'] == 0] = '10-2' ## Label : 0 -> '10-2'
    submission['label'] = submission['label'].apply(lambda x : str(x)) ## Dtype : int -> object

    print("Save Submission")
    submission.to_csv(f"{sub_dir}/{args.model}_sub.csv", index = False)
    print("Done")