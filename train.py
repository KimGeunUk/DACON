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

from models.runner.train_runner import CustomTrainer
from models.model.pytorch_cnn import CNNclassification
from models.model.pytorch_timm import TimmModel

from utils.set_seed import seed_everything
from utils.set_path import *
from utils.focal_loss import FocalLoss

from sklearn.model_selection import StratifiedKFold

def kfold_main(args, train_img_paths, train_labels, valid_img_paths, valid_labels, test_img_paths, fold_num):
    train_dataset = CustomDataset(img_paths=train_img_paths, labels=train_labels, mode='train', img_size=args.img_size, transforms=True)
    valid_dataset = CustomDataset(img_paths=valid_img_paths, labels=valid_labels, mode='train', img_size=args.img_size, transforms=False)
    for i in range(10):
        train_dataset += CustomDataset(img_paths=train_img_paths, labels=train_labels, mode='train', img_size=args.img_size, transforms=True)
        valid_dataset += CustomDataset(img_paths=valid_img_paths, labels=valid_labels, mode='train', img_size=args.img_size, transforms=False)
    
    test_dataset = CustomDataset(img_paths=test_img_paths, labels=None, mode='test', img_size=args.img_size, transforms=False)
    
    train_dataloader, valid_dataloader, _ = get_data_loader(
        train_dataset=train_dataset, valid_dataset=valid_dataset, test_dataset=None, batch_size=args.batch_size, num_workers=args.num_workers
        )
    
    # model = CNNclassification().to(device)
    model = TimmModel(args).to(device)
        
    #self.loss_fn = nn.CrossEntropyLoss(label_smoothing=opt.label_smoothing)
    loss_fn = FocalLoss(label_smoothing=args.label_smoothing)        
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, betas=args.betas, eps=args.eps, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.T_max, eta_min=args.eta_min)
    
    trainer = CustomTrainer(model=model, optimizer=optimizer, scheduler=scheduler, loss_fn=loss_fn, device=device)
    
    ealry_stopping = 0
    
    best_loss = np.inf
    best_acc = 0
    best_f1 = 0
    best_epoch = 0    
    
    for epoch in range(args.epochs):
        gc.collect()
        
        lr = optimizer.param_groups[0]['lr']
        print('='*25, f'TRAIN epoch:{epoch+1}', '='*25, f'lr:{lr:.6f}')
        train_loss, train_f1_score, train_acc_score = trainer.train_run(dataloader=train_dataloader,epoch=epoch)

        print(f'train_loss: {train_loss:.6f}, train_f1_score: {train_f1_score:.6f}, train_acc_score : {train_acc_score:.6f}')         
        print(' ')
        print('='*25, f'VALID epoch:{epoch+1}', '='*25)
        with torch.no_grad():
            val_loss, val_f1_score, val_acc_score = trainer.val_run(dataloader=valid_dataloader,epoch=epoch)
            
        print(f'val_loss: {val_loss:.6f}, val_f1_score: {val_f1_score:.6f}, val_acc_score: {val_acc_score:.6f}')
        wandb.log({f"{args.model}/Fold{fold_num}/train_loss": train_loss,
                   f"{args.model}/Fold{fold_num}/train_f1_score": train_f1_score,
                   f"{args.model}/Fold{fold_num}/train_acc_score": train_acc_score,
                   f"{args.model}/Fold{fold_num}/val_loss": val_loss,
                   f"{args.model}/Fold{fold_num}/val_f1_score": val_f1_score,
                   f"{args.model}/Fold{fold_num}/val_acc_score": val_acc_score})
        
        if val_loss < best_loss or epoch == args.epochs:
            ealry_stopping = 0
            best_epoch = epoch+1
            best_loss = val_loss
            best_acc = val_acc_score
            best_f1 = val_f1_score
            
            torch.save({'epoch':epoch,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        }, f'{fold_save_dir}{args.model}_{best_epoch}_{best_loss:.4f}.pth')
            print(' ')
            print(f'Model is Saved when epoch is : {epoch+1}')
        else:
            ealry_stopping += 1
            
        print(' ')
        print(f'BEST LOSS : {best_loss:.4f}, BEST F1 : {best_f1:.4f}, BEST ACC : {best_acc:.4f}, EPOCH : {best_epoch}')
        print('='*70)
        
        if ealry_stopping == args.patience:
            print(' ')
            print(f'Model is early stopped and saved when epoch is : {best_epoch}')
            break
        

def arg_parse(known=False):
    parser = argparse.ArgumentParser()    
    parser.add_argument('--model', type=str, default='tf_efficientnet_b3_ns')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--img-size', type=int, default=768)
    parser.add_argument('--learning-rate', type=float, default=2e-3)
    
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
    
    wandb.init(project="sign_language-classification",
            group="local",
            name=f'{args.model}_{args.batch_size}_{args.img_size}_{args.learning_rate}',
            entity="kgw5430")       # 181acb831adde2210eae1618f343dfe466026445
    
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    save_dir = MODEL_SAVE_PATH + f"{args.model}"
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    num_folder = len(glob(save_dir + "*"))
    save_dir = save_dir + f"_{num_folder+1}/"
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    
    print(f"TRAIN IMG PATHS LEN : {len(TRAIN_IMG_PATHS)}")
    print(f"TEST IMG PATHS LEN : {len(TEST_IMG_PATHS)}")    
    
    folds = []
    skf = StratifiedKFold(n_splits=args.num_folds, shuffle=True, random_state=args.seed)
    
    for (train_idx, valid_idx) in skf.split(TRAIN_IMG_PATHS, TRAIN_IMG_LABELS):
        folds.append((train_idx, valid_idx))
    
    for idx in range(len(folds)):
        print(f"\t\t\tFold {str(idx+1)} START!!!!")
        print(' ')
        train_idx, valid_idx = folds[idx]
        
        fold_save_dir = save_dir + f"fold{idx+1}/"
        if not os.path.exists(fold_save_dir):
            os.mkdir(fold_save_dir)
        
        kfold_main(args,
                   np.array(TRAIN_IMG_PATHS)[train_idx], 
                   np.array(TRAIN_IMG_LABELS)[train_idx],
                   np.array(TRAIN_IMG_PATHS)[valid_idx],
                   np.array(TRAIN_IMG_LABELS)[valid_idx],
                   np.array(TEST_IMG_PATHS),
                   idx+1)
        