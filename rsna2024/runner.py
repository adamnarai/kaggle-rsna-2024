import os
import random
import json
import numpy as np
import pandas as pd
import wandb
from sklearn.model_selection import KFold

import torch
from torch import nn, optim

from rsna2024.trainer import Trainer
from rsna2024 import model as module_model
import rsna2024.data_loader.augmentation as module_aug
import rsna2024.data_loader.data_loaders as module_data
import rsna2024.utils.loss as module_loss


class RunnerBase:
    def __init__(self, cfg, model_name=None, model_dir=None):
        self.cfg = cfg
        self.device = self.get_device()
        self.loss_fn = self.get_instance(module_loss, 'loss', cfg)
        self.model_name = model_name
        self.data_dir = None
        self.df = None
        self.model_dir = model_dir

    def get_device(self):
        return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def get_instance(self, module, name, cfg, *args):

        # Make list values torch.tensor
        cfg_kvargs = cfg[name]['args'].copy()
        for k, v in cfg_kvargs.items():
            if k in ['weight'] and isinstance(v, list):
                cfg_kvargs[k] = torch.tensor(v).to(self.device)

        return getattr(module, cfg[name]['type'])(*args, **cfg_kvargs)
    
    def seed_everything(self):
        seed = self.cfg['seed']
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = self.cfg['deterministic']
        torch.backends.cudnn.benchmark = self.cfg['benchmark']

    def init_wandb(self, project_name_prefix=''):
        wandb.login()
        run = wandb.init(project='{}{}'.format(project_name_prefix, self.cfg['project_name']), config=self.cfg, tags=self.cfg['tags'], notes=self.cfg['notes'])
        return run

    def create_cv_splits(self):
        kf = KFold(n_splits=self.cfg['trainer']['cv_fold'], random_state=self.cfg['seed'], shuffle=True)
        splits = []
        split_indices = []
        y = self.df[self.cfg['out_vars']].sum(axis=1)
        for train_index, valid_index in kf.split(X=np.zeros(len(self.df)), y=y):
            split_indices.append((train_index, valid_index))
            splits.append((self.df.iloc[train_index].copy(), self.df.iloc[valid_index].copy()))
        return splits, split_indices

    def save_splits(self, splits):
        assert os.path.exists(self.model_dir)
        df = pd.DataFrame()
        for i, (df_train, df_validation) in enumerate(splits):
            df_train.loc[:,'split'] = 'train'
            df_train.loc[:,'fold'] = i + 1
            df_validation.loc[:,'split'] = 'validation'
            df_validation.loc[:,'fold'] = i + 1
            df = pd.concat([df, df_train, df_validation])
        df = df.sort_values(by=['fold', 'split']).reset_index(drop=True)
        df.to_csv(os.path.join(self.model_dir, 'splits.csv'), index=False)

    def load_splits(self):
        df = pd.read_csv(os.path.join(self.model_dir, 'splits.csv'), dtype={'study_id': 'str'})
        splits = []
        for i in range(1, self.cfg['trainer']['cv_fold'] + 1):
            df_train = df[(df['fold'] == i) & (df['split'] == 'train')].drop(columns=['fold', 'split'])
            df_valid = df[(df['fold'] == i) & (df['split'] == 'validation')].drop(columns=['fold', 'split'])
            splits.append((df_train, df_valid))
        return splits

    def save_config(self):
        assert os.path.exists(self.model_dir)
        with open(os.path.join(self.model_dir, 'config.json'), 'w', encoding='utf-8') as f:
            json.dump(self.cfg, f, indent=4)

    def create_model_dir(self, model_name):
        self.model_dir = os.path.join(self.cfg['root'], 'models', model_name)
        os.makedirs(self.model_dir, exist_ok=True)

    def train_model(self, df_train, df_valid, state_filename, validate=True):
        model = self.get_instance(module_model, 'model', self.cfg).to(self.device)
        optimizer = self.get_instance(optim, 'optimizer', self.cfg, model.parameters())
        scheduler = self.get_instance(optim.lr_scheduler, 'scheduler', self.cfg, optimizer)

        train_transform = self.get_instance(module_aug, 'train_transform', self.cfg).get_transform()
        train_loader = self.get_instance(module_data, 'data_loader', self.cfg, df_train, train_transform, 'train', self.data_dir, self.cfg['out_vars'])

        valid_transform = self.get_instance(module_aug, 'valid_transform', self.cfg).get_transform()
        valid_loader = self.get_instance(module_data, 'data_loader', self.cfg, df_valid, valid_transform, 'valid', self.data_dir, self.cfg['out_vars'])

        # Training
        trainer = Trainer(model, train_loader, valid_loader, self.loss_fn, optimizer, scheduler, self.device, state_filename=state_filename, wandb_log=self.cfg['use_wandb'], 
                          num_epochs=self.cfg['trainer']['epochs'], metrics=self.cfg['trainer']['metrics'], trainer_type=self.cfg['trainer']['type'])
        trainer.train_epochs(num_epochs=self.cfg['trainer']['epochs'], validate=validate)
        trainer.save_state(state_filename)

        return trainer
    
    def validate_model(self, df_valid, state_filename):
        model = self.get_instance(module_model, 'model', self.cfg).to(self.device)

        valid_transform = self.get_instance(module_aug, 'valid_transform', self.cfg).get_transform()
        valid_loader = self.get_instance(module_data, 'data_loader', self.cfg, df_valid, valid_transform, 'valid', self.data_dir, self.cfg['out_vars'])

        # Training
        trainer = Trainer(model, None, valid_loader, self.loss_fn, device=self.device, metrics=self.cfg['trainer']['metrics'], trainer_type=self.cfg['trainer']['type'])
        trainer.load_state(state_filename)
        valid_loss, metrics = trainer.validate()

        return valid_loss, metrics
    
    def get_predictions(self, df, state_filename):
        model = self.get_instance(module_model, 'model', self.cfg).to(self.device)

        valid_transform = self.get_instance(module_aug, 'valid_transform', self.cfg).get_transform()
        valid_loader = self.get_instance(module_data, 'data_loader', self.cfg, df, valid_transform, 'valid', self.data_dir, self.cfg['out_vars'])

        # Training
        trainer = Trainer(model, None, valid_loader, self.loss_fn, device=self.device, metrics=self.cfg['trainer']['metrics'])
        trainer.load_state(state_filename)
        preds, ys = trainer.predict()

        return preds, ys


class Runner(RunnerBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_dir = os.path.join(self.cfg['root'], 'data', 'raw')
        self.df = self.load_data(self.data_dir)

    def load_data(self, data_dir):
        df = pd.read_csv(os.path.join(data_dir, 'train.csv'), dtype={'study_id': 'str'})
        pd.set_option('future.no_silent_downcasting', True)
        df = df.replace(self.cfg['labels'])
        return df

    def train(self):
        # Wandb
        if self.cfg['use_wandb']:
            run = self.init_wandb(project_name_prefix='kaggle-')
            run_name = run.name
        else:
            run_name = 'debug'
        
        # Model dir
        self.model_name = '{}-{}'.format(self.cfg['project_name'], run_name)
        self.create_model_dir(self.model_name)

        # Seed
        self.seed_everything()

        # Cross-validation splits
        splits, _ = self.create_cv_splits()
        self.save_splits(splits)

        # Save config
        self.save_config()

        last_metric_list = []
        best_metric_list = []
        for cv, (df_train, df_valid) in enumerate(splits):
            print(f"Cross-validation fold {cv+1}/{self.cfg['trainer']['cv_fold']}")
            state_filename = os.path.join(self.model_dir, f'{self.model_name}-cv{cv+1}.pt')
            trainer = self.train_model(df_train, df_valid, state_filename)
            best_metric_list.append(trainer.best_metric)
            last_metric_list.append(trainer.last_metric)
            if self.cfg['use_wandb']:
                wandb.log({f'metric_cv{cv+1}': trainer.last_metric})
                wandb.log({f'best_metric_cv{cv+1}': trainer.best_metric})
            if self.cfg['trainer']['one_fold']:
                break
        if self.cfg['use_wandb']:
            wandb.log({'mean_metric': np.mean(last_metric_list)})
            wandb.log({'mean_best_metric': np.mean(best_metric_list)})
            wandb.finish()

    def validate(self, state_type='best'):
        self.model_dir = os.path.join(self.cfg['root'], 'models', self.model_name)

        self.seed_everything()
        splits = self.load_splits()

        metric_list = []
        for cv, (_, df_valid) in enumerate(splits):
            print(f'Cross-validation fold {cv+1}/{self.cfg['trainer']['cv_fold']}')
            if state_type == 'best':
                state_filename_suffix = '_best'
            elif state_type == 'last':
                state_filename_suffix = ''
            else:
                raise ValueError('state_type must be "best" or "last"')
            state_filename = os.path.join(self.model_dir, f'{self.model_name}-cv{cv+1}{state_filename_suffix}.pt')
            valid_loss, metrics = self.validate_model(df_valid, state_filename)
            metric_list.append(metrics)
            print(f'valid loss: {valid_loss:.4f}, {metrics}\n')

            if self.cfg['trainer']['one_fold']:
                break
        return metric_list
    
    def predict(self, state_type='best', oof=True):
        self.model_dir = os.path.join(self.cfg['root'], 'models', self.model_name)

        self.seed_everything()
        splits = self.load_splits()

        preds, ys = [], []
        data = pd.DataFrame()
        for cv, (df_train, df_valid) in enumerate(splits):
            print(f'Cross-validation fold {cv+1}/{self.cfg['trainer']['cv_fold']}')
            if state_type == 'best':
                state_filename_suffix = '_best'
            elif state_type == 'last':
                state_filename_suffix = ''
            else:
                raise ValueError('state_type must be "best" or "last"')
            state_filename = os.path.join(self.model_dir, f'{self.model_name}-cv{cv+1}{state_filename_suffix}.pt')
            if oof:
                pred, y = self.get_predictions(df_valid, state_filename)
                data = pd.concat([data, df_valid])
            else:
                pred, y = self.get_predictions(df_train, state_filename)
                data = pd.concat([data, df_train])
            preds.append(pred)
            ys.append(y)
        preds = np.concatenate(preds)
        preds = np.moveaxis(preds, 1, -1)
        ys = np.concatenate(ys)

        return preds, ys, data

    def get_sample_batch(self):
        transform = self.get_instance(module_aug, 'train_transform', self.cfg).get_transform()
        data_loader = self.get_instance(module_data, 'data_loader', self.cfg, self.df, transform, 'train', self.data_dir, self.cfg['out_vars'])
        return next(iter(data_loader))
