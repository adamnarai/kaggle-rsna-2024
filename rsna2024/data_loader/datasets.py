import os
import pandas as pd
import numpy as np
import pydicom
import cv2
import random

from torch.utils.data import Dataset

from rsna2024.utils import natural_sort

class RSNA2024Dataset(Dataset):
    def __init__(self, df, data_dir, out_vars, img_num, transform=None):
        self.df = df
        self.df_series = self.load_series_info(data_dir)
        self.df_coordinates = self.load_coordinates_info(data_dir)
        self.img_dir = os.path.join(data_dir, 'train_images')
        self.out_vars = out_vars
        self.transform = transform
        self.img_num = img_num

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        label = self.df[self.out_vars].iloc[idx].values
        label = np.nan_to_num(label.astype(float), nan=0).astype(np.int64)

        # Sagittal T1
        x1 = self.get_series(row.study_id, 'Sagittal T1', img_num=self.img_num[0])
        x2 = self.get_series(row.study_id, 'Sagittal T2/STIR', img_num=self.img_num[1])
        x3 = self.get_series(row.study_id, 'Axial T2', img_num=self.img_num[2])
        x = np.concatenate([x1, x2, x3], axis=2)

        if self.transform:
            x = self.transform(image=x)['image']

        return x, label

    def load_series_info(self, data_dir):
        return pd.read_csv(os.path.join(data_dir, 'train_series_descriptions.csv'), dtype={'study_id': 'str', 'series_id': 'str'})
    
    def load_coordinates_info(self, data_dir):
        df_coordinates = pd.read_csv(os.path.join(data_dir, '..', 'processed', 'train_label_coordinates.csv'), dtype={'study_id': 'str', 'series_id': 'str'})
        return df_coordinates

    def get_series_id(self, study_id, series_description, series_id_list=None):
        series_list = self.df_series[(self.df_series['study_id']==study_id) & (self.df_series['series_description']==series_description)]['series_id'].tolist()

        if len(series_list) == 0:
            return None

        if series_id_list is not None:
            series_intersect = list(set(series_list) & set(series_id_list))
            if len(series_intersect) > 0:
                series_list = series_intersect

        return random.sample(series_list, 1)[0]
    
    def get_coordinates(self, study_id):
        study_coordinates = self.df_coordinates[self.df_coordinates['study_id']==study_id]
        study_coordinates = study_coordinates.sort_values(by=['row_id'], key=lambda column: column.map(lambda e: self.out_vars.index(e)))
        x_values = np.nan_to_num(study_coordinates['x_norm'].values.astype(np.float32), nan=0)
        y_values = np.nan_to_num(study_coordinates['y_norm'].values.astype(np.float32), nan=0)
        xy_values = np.concatenate([x_values, y_values], axis=0)
        if len(xy_values) != 50:
            xy_values = np.zeros(50, dtype=np.float32)

        
        instance_number = np.nan_to_num(study_coordinates['instance_number'].values.astype(float), nan=0).astype(np.int64)
        
        series_with_cooord = self.df_coordinates[self.df_coordinates['study_id']==study_id]['series_id'].unique().astype(str)
        return xy_values, instance_number, series_with_cooord

    def to_shape(self, a, shape):
        y_, x_ = shape
        y, x = a.shape
        y_pad = y_ - y
        x_pad = x_ - x
        return np.pad(
            a,
            ((y_pad // 2, y_pad // 2 + y_pad % 2), (x_pad // 2, x_pad // 2 + x_pad % 2)),
            mode='constant',
        )

    def get_series(self, study_id, series_description, img_num, resolution=(512, 512), series_id_list=None):
        x = np.zeros((*resolution, img_num), dtype=np.float32)
        series_id = self.get_series_id(study_id, series_description, series_id_list)
        if series_id is None:
            return x

        series_dir = os.path.join(self.img_dir, study_id, series_id)
        file_list = natural_sort(os.listdir(series_dir))

        if len(file_list) > img_num:
            start_index = (len(file_list)-img_num) // 2
            file_list = file_list[start_index:start_index + img_num]

        for i, filename in enumerate(file_list):
            ds = pydicom.dcmread(os.path.join(series_dir, filename))
            img = ds.pixel_array.astype(np.float32)
            img = cv2.resize(img, resolution, interpolation=cv2.INTER_CUBIC)

            x[..., i] = img

        # Standardize series
        x = (x - x.mean()) / x.std()
        
        return x

class RSNA2024SplitDataset(RSNA2024Dataset):
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        label = self.df[self.out_vars].iloc[idx].values
        label = np.nan_to_num(label.astype(float), nan=0).astype(np.int64)

        # Sagittal T1
        x1 = self.get_series(row.study_id, 'Sagittal T1', img_num=self.img_num[0])
        x2 = self.get_series(row.study_id, 'Sagittal T2/STIR', img_num=self.img_num[1])
        x3 = self.get_series(row.study_id, 'Axial T2', img_num=self.img_num[2])
        
        if self.transform:
            x1 = self.transform(image=x1)['image']
            x2 = self.transform(image=x2)['image']
            x3 = self.transform(image=x3)['image']

        return x1, x2, x3, label
    
class RSNA2024SplitCoordDataset(RSNA2024Dataset):
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        label = self.df[self.out_vars].iloc[idx].values
        label = np.nan_to_num(label.astype(float), nan=0).astype(np.int64)
        xy_values, instance_number, series_id_list = self.get_coordinates(row.study_id)

        # Sagittal T1
        x1 = self.get_series(row.study_id, 'Sagittal T1', img_num=self.img_num[0], series_id_list=series_id_list)
        x2 = self.get_series(row.study_id, 'Sagittal T2/STIR', img_num=self.img_num[1], series_id_list=series_id_list)
        x3 = self.get_series(row.study_id, 'Axial T2', img_num=self.img_num[2], series_id_list=series_id_list)
        
        if self.transform:
            x1 = self.transform(image=x1)['image']
            x2 = self.transform(image=x2)['image']
            x3 = self.transform(image=x3)['image']

        return x1, x2, x3, (label, xy_values)
