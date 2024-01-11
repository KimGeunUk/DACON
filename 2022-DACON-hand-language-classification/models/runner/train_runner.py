import torch
from tqdm.auto import tqdm
import wandb

from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score

def f1_function(real, pred):
    score = f1_score(real, pred, average="macro")
    return score

def acc_function(real, pred):
    score = accuracy_score(real, pred)
    return score   

class CustomTrainer():
    def __init__(self, model, optimizer, scheduler, loss_fn, device):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.device = device 
        self.scaler = torch.cuda.amp.GradScaler()
               
        wandb.watch(model)
    
    def train_run(self, dataloader, epoch):
        self.model.train()
        
        train_loss = 0.0
        train_pred, train_label = [], []
        
        for batch in tqdm(dataloader, bar_format='{l_bar}{bar:50}{r_bar}{bar:-10b}'):
            self.optimizer.zero_grad()
            
            img = torch.tensor(batch[0], dtype=torch.float32, device=self.device)
            label = torch.tensor(batch[1], dtype=torch.long, device=self.device)                 
            
            with torch.cuda.amp.autocast():
                output = self.model(img)
                loss = self.loss_fn(output, label)
            
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()            
            self.scheduler.step()
            
            train_loss += loss.item() / len(dataloader)
            train_pred += output.argmax(1).detach().cpu().numpy().tolist()
            train_label += label.detach().cpu().numpy().tolist()
        
        train_f1_score = f1_function(train_label, train_pred)
        train_acc_score = acc_function(train_label, train_pred)
        return train_loss, train_f1_score, train_acc_score
    
    def val_run(self, dataloader, epoch):
        self.model.eval()
        
        val_loss = 0.0
        val_pred, val_label =[], []
        
        for batch in tqdm(dataloader, bar_format='{l_bar}{bar:50}{r_bar}{bar:-10b}'):
            img = torch.tensor(batch[0], dtype=torch.float32, device=self.device)
            label = torch.tensor(batch[1], dtype=torch.long, device=self.device)
            
            output = self.model(img)
            loss = self.loss_fn(output, label)
            
            val_loss += loss.item() / len(dataloader)
            val_pred += output.argmax(1).detach().cpu().numpy().tolist()
            val_label += label.detach().cpu().numpy().tolist()
        
        val_f1_score = f1_function(val_label, val_pred)
        val_acc_score = acc_function(val_label, val_pred)
        return val_loss, val_f1_score, val_acc_score
