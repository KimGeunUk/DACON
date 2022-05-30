import torch
import ttach as tta
import numpy as np
from tqdm.auto import tqdm
from glob import glob

from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score

class CustomInference():
    def __init__(self, model, optimizer, scheduler, loss_fn, device):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.device = device
        
    def test_run(self, dataloader, fold_model_dir):
        print("Run inference !!")
        
        status = torch.load(fold_model_dir)
        self.model.load_state_dict(status["state_dict"])
        self.optimizer.load_state_dict(status["optimizer"])
        self.scheduler.load_state_dict(status["scheduler"])
        self.tta_transforms = tta.Compose(
            [
                tta.HorizontalFlip(),
                tta.VerticalFlip(),
            ]
        )
        
        self.model = tta.ClassificationTTAWrapper(self.model, self.tta_transforms)
        self.model.to(self.device)
        
        preds = []
        
        self.model.eval()
        with torch.no_grad():
            for batch in tqdm(dataloader, bar_format='{l_bar}{bar:50}{r_bar}{bar:-10b}'):
                img = torch.tensor(batch, dtype = torch.float32, device = self.device)            
                output = self.model(img)
                preds.extend(output.data.cpu().numpy())    
        preds=np.array(preds)
        return preds
    
    