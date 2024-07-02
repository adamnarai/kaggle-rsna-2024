import os
import pandas as pd
import numpy as np
import pydicom
import cv2
cv2.setNumThreads(0)

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from rsna2024.utils import natural_sort

class RSNA2024Dataset(Dataset):
    def __init__(self, df, data_dir, out_vars, img_num=(10, 10, 10), transform=None):
        self.df = df
        self.df_series = self.load_series_info(data_dir)
        self.img_dir = os.path.join(data_dir, 'train_images')
        self.out_vars = out_vars
        self.transform = transform
        self.img_num = img_num

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        x = np.zeros((512, 512, sum(self.img_num)), dtype=np.uint8)
        label = self.df[self.out_vars].iloc[idx].values
        label = np.nan_to_num(label.astype(float), nan=0).astype(np.int64)

        # Sagittal T1
        x = self.add_series(x, row.study_id, 'Sagittal T1', img_num=self.img_num[0], offset=0)
        x = self.add_series(x, row.study_id, 'Sagittal T2/STIR', img_num=self.img_num[1], offset=self.img_num[0])
        x = self.add_series(x, row.study_id, 'Axial T2', img_num=self.img_num[2], offset=sum(self.img_num[:2]))
        
        if self.transform:
            x = self.transform(x)
        
        # # Albumentations
        # if self.transform:
        #     x = self.transform(image=x)['image']

        return x, label
    
    def load_series_info(self, data_dir):
        return pd.read_csv(os.path.join(data_dir, 'train_series_descriptions.csv'), dtype={'study_id': 'str', 'series_id': 'str'})

    def get_series_id(self, study_id, series_description):
        series_list = self.df_series[(self.df_series['study_id']==study_id) & (self.df_series['series_description']==series_description)]['series_id']

        if len(series_list) == 0:
            return None
            
        return series_list.values[0]
  
    def add_series(self, x, study_id, series_description, img_num=10, offset=0, resolution=(512, 512)):
        series_id = self.get_series_id(study_id, series_description)
        if series_id is None:
            return x
        
        series_dir = os.path.join(self.img_dir, study_id, series_id)
        file_list = natural_sort(os.listdir(series_dir))

        if len(file_list) >= img_num:
            start_index = (len(file_list)-img_num) // 2
            file_list = file_list[start_index:start_index + img_num]

        for i, filename in enumerate(file_list):
            ds = pydicom.dcmread(os.path.join(series_dir, filename))
            img = ds.pixel_array.astype(np.float32)
            img = cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
            img = cv2.resize(img, resolution, interpolation=cv2.INTER_CUBIC)

            x[..., i + offset] = img

        return x
