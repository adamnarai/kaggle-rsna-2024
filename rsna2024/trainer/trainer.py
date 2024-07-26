import time
import numpy as np
from tqdm import tqdm

import torch
from torch import nn

import wandb

class Trainer:
    def __init__(self, model, train_loader, valid_loader, loss_fn, optimizer=None, scheduler=None, device=None, state_filename=None, metrics=None, num_epochs=None, wandb_log=False, trainer_type='standard'):
        self.model = model
        self.train_dataloader = train_loader
        self.valid_dataloader = valid_loader
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.state_filename = state_filename
        self.metrics = metrics
        self.num_epochs = num_epochs
        self.wandb_log = wandb_log
        self.best_metric = np.inf
        self.last_metric = np.inf
        self.epoch_count = 0
        self.scaler = torch.cuda.amp.GradScaler()
        self.trainer_type = trainer_type

    def train_epochs(self, num_epochs=None, validate=True):
        if num_epochs is not None:
            self.num_epochs = num_epochs
        if num_epochs == 0:
            return self.model, 0
        since = time.time()

        for epoch in range(self.num_epochs):
            print(f'Epoch {epoch+1}/{self.num_epochs}')
            print('-' * 10)

            train_loss = self.train()
            if validate:
                test_loss, metrics = self.validate()
                main_metric = metrics[self.metrics[0]]
                self.last_metric = main_metric
                if main_metric < self.best_metric:
                    self.best_metric = main_metric
                    print(f"New best loss: {self.best_metric:.4f}\nSaving model to {self.state_filename}")
                    self.save_state(self.state_filename.replace('.pt', '_best.pt'))
            else:
                test_loss = np.nan
                metrics = {m: np.nan for m in self.metrics}
                self.best_metric = np.nan
            
            print(f"lr: {self.scheduler.get_last_lr()}")
            self.scheduler.step()
            print(f"train loss: {train_loss:.4f}, valid loss: {test_loss:.4f}, {metrics}\n")
            if self.wandb_log:
                log_dict = {'train_loss': train_loss, 'valid_loss': test_loss}
                log_dict.update(metrics)
                wandb.log(log_dict)

        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Final {metrics}\n')
        self.epoch_count += self.num_epochs
        return
    
    def unfreeze(self):
        for param in self.model.parameters():
            param.requires_grad = True
        return
    
    def train(self):
        num_batches = len(self.train_dataloader)
        self.model.train()
        
        train_loss = 0
        for batch in tqdm(self.train_dataloader, total=num_batches):
            self.optimizer.zero_grad()
            if self.trainer_type == 'standard':
                *X, y = batch
                X = [x.to(self.device, non_blocking=True) for x in X]
                y = y.to(self.device, non_blocking=True)
                
                with torch.cuda.amp.autocast():
                    pred = self.model(*X)
                    loss = self.loss_fn(pred, y)

            # Backpropagation
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()

            train_loss += loss.item()

        train_loss /= num_batches

        return train_loss

    def validate(self):
        num_batches = len(self.valid_dataloader)
        self.model.eval()
        
        valid_loss = 0
        metrics = {m: 0 for m in self.metrics}
        with torch.no_grad():
            for batch in tqdm(self.valid_dataloader, total=num_batches):
                if self.trainer_type == 'standard':
                    *X, y = batch
                    X = [x.to(self.device, non_blocking=True) for x in X]
                    y = y.to(self.device, non_blocking=True)
                    
                    pred = self.model(*X)
                    loss = self.loss_fn(pred, y)

                    valid_loss += loss.item()

                    if 'detailed_loss' in metrics.keys():
                        weights = torch.tensor([1.0, 2.0, 4.0]).to(self.device)
                        detailed_loss_fn = nn.CrossEntropyLoss(weight=weights, reduction='none').to(self.device)
                        detailed_loss = detailed_loss_fn(torch.unflatten(pred, 1, [3, -1]), y).to('cpu').numpy()
                        detailed_loss = detailed_loss.sum(axis=0)
                        detailed_loss = detailed_loss * len(detailed_loss) / weights[y].sum().to('cpu').numpy()  # Reproduce 'mean' reduction
                        metrics['detailed_loss'] += detailed_loss


        if 'loss' in metrics.keys():
            metrics['loss'] = valid_loss

        valid_loss /= num_batches
        for m in metrics:
            metrics[m] /= num_batches

        return valid_loss, metrics
    
    def predict(self):
        num_batches = len(self.valid_dataloader)
        self.model.eval()
        
        preds, ys = [], []
        with torch.no_grad():
            for batch in tqdm(self.valid_dataloader, total=num_batches):
                *X, y = batch
                X = [x.to(self.device, non_blocking=True) for x in X]
                y = y.to(self.device, non_blocking=True)
                
                pred = self.model(*X)
                # pred = torch.unflatten(pred, 1, [3, -1])
                preds.append(pred)
                ys.append(y)
        preds = torch.cat(preds, dim=0).to('cpu').numpy()
        ys = torch.cat(ys, dim=0).to('cpu').numpy()

        return preds, ys
    
    def save_state(self, filename):
        torch.save(self.model.state_dict(), filename)
        return
    
    def load_state(self, filename):
        self.model.load_state_dict(torch.load(filename))
    
    def load_best_state(self):
        self.load_state(self.state_filename.replace('.pt', '_best.pt'))
    
    def get_model(self):
        return self.model
    
    def set_dataloaders(self, dataloaders):
        self.train_dataloader = dataloaders['train']
        self.valid_dataloader = dataloaders['validation']