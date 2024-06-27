import time
import numpy as np
from tqdm import tqdm

import torch

import wandb

class Trainer:
    def __init__(self, model, train_loader, valid_loader, loss_fn, optimizer, scheduler, device, state_filename, metric, num_epochs, wandb_log=False):
        self.model = model
        self.train_dataloader = train_loader
        self.valid_dataloader = valid_loader
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.state_filename = state_filename
        self.metric = metric
        self.num_epochs = num_epochs
        self.wandb_log = wandb_log
        self.best_metric = np.inf
        self.last_metric = np.inf
        self.epoch_count = 0
        self.test_y = []
        self.test_pred = []

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
                test_loss, metric = self.valid()
                self.last_metric = metric
                if metric < self.best_metric:
                    self.best_metric = metric
                    print(f"New best {self.metric}: {self.best_metric:.4f}\nSaving model to {self.state_filename}")
                    self.save_state(self.state_filename.replace('.pt', '_best.pt'))
            else:
                test_loss, metric = np.nan, np.nan
                self.best_metric = np.nan
            
            print(f"lr: {self.scheduler.get_last_lr()}")
            self.scheduler.step()
            print(f"train loss: {train_loss:.4f}, test loss: {test_loss:.4f}, {self.metric}: {metric:.4f}\n")
            if self.wandb_log:
                wandb.log({'train_loss': train_loss, 'valid_loss': test_loss, self.metric: metric})

        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Final {self.metric}: {metric:4f}\n')
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
            if len(batch) == 2:
                X, y = batch
                X, y = X.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
                pred = self.model(X)
            elif len(batch) == 3:
                X1, X2, y = batch
                X1, X2, y = X1.to(self.device, non_blocking=True), X2.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
                pred = self.model(X1, X2)
            elif len(batch) == 4:
                X1, X2, X3, y = batch
                X1, X2, X3, y = X1.to(self.device, non_blocking=True), X2.to(self.device, non_blocking=True), X3.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
                pred = self.model(X1, X2, X3)

            # Compute prediction error
            loss = self.loss_fn(torch.unflatten(pred, 1, [3, 25]), y)

            # Backpropagation
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            train_loss += loss.item()

        train_loss /= num_batches
        return train_loss

    def valid(self):
        num_batches = len(self.valid_dataloader)
        self.model.eval()
        
        valid_loss = 0
        metric = 0
        with torch.no_grad():
            for batch in tqdm(self.valid_dataloader, total=num_batches):
                if len(batch) == 2:
                    x, y = batch
                    x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
                    pred = self.model(x)
                elif len(batch) == 3:
                    x1, x2, y = batch
                    x1, x2, y = x1.to(self.device, non_blocking=True), x2.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
                    pred = self.model(x1, x2)
                elif len(batch) == 4:
                    x1, x2, x3, y = batch
                    x1, x2, x3, y = x1.to(self.device, non_blocking=True), x2.to(self.device, non_blocking=True), x3.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
                    pred = self.model(x1, x2, x3)

                valid_loss += self.loss_fn(torch.unflatten(pred, 1, [3, 25]), y).item()

        valid_loss /= num_batches

        return valid_loss, metric
    
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