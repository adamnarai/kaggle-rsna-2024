import os
import pandas as pd
import numpy as np
import pydicom
import cv2
import random

from torch.utils.data import Dataset

from rsna2024.utils import natural_sort

class RSNA2024Dataset(Dataset):
    def __init__(self, df, data_dir, out_vars, img_num, resolution=(512, 512), transform=None):
        self.df = df
        self.df_series = self.load_series_info(data_dir)
        self.df_coordinates = self.load_coordinates_info(data_dir).merge(self.df_series, how='left', on=['study_id', 'series_id'])
        self.img_dir = os.path.join(data_dir, 'train_images')
        self.out_vars = out_vars
        self.transform = transform
        self.img_num = img_num
        self.resolution = resolution

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

    def get_series_id(self, study_id, series_description):
        series_list = self.df_series[(self.df_series['study_id']==study_id) & (self.df_series['series_description']==series_description)]['series_id'].tolist()

        if len(series_list) == 0:
            return None

        return random.sample(series_list, 1)[0]

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

    def get_series(self, study_id, series_description, img_num):
        x = np.zeros((*self.resolution, img_num), dtype=np.float32)
        series_id = self.get_series_id(study_id, series_description)
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
            img = cv2.resize(img, self.resolution, interpolation=cv2.INTER_CUBIC)

            x[..., i] = img

        # Standardize series
        x = (x - x.mean()) / x.std()
        
        return x
    
    def get_series_with_coord(self, study_id, series_description, img_num, kpmap=False):
        if kpmap:
            img_num += 1
        x = np.zeros((*self.resolution, img_num), dtype=np.float32)
        study_coord = self.df_coordinates[self.df_coordinates['study_id']==study_id]
        series_coord = study_coord[study_coord['series_description']==series_description]
        series_coord = series_coord.sort_values(by=['row_id'], 
            key=lambda column: column.map(lambda e: self.out_vars.index(e))).reset_index(drop=True)
        series_id = series_coord['series_id'].unique()

        # Keypoints
        if series_description == 'Sagittal T1':
            expected_vars = [x for x in self.out_vars if 'neural_foraminal_narrowing' in x]
        elif series_description == 'Sagittal T2/STIR':
            expected_vars = [x for x in self.out_vars if 'spinal_canal_stenosis' in x]
        elif series_description == 'Axial T2':
            expected_vars = [x for x in self.out_vars if 'subarticular_stenosis' in x]
        series_coord_padded = series_coord.merge(pd.DataFrame({'row_id': expected_vars}), on='row_id', how='right').fillna(0)

        keypoints = list(zip(series_coord_padded['x_norm'].values * self.resolution[0], 
                             series_coord_padded['y_norm'].values * self.resolution[1]))
        
        if len(series_id) == 0:
            return x, keypoints
        
        # If multiple series, get the one with the most coordinates
        series_id = series_coord['series_id'].value_counts().sort_values(ascending=False).index[0]

        series_dir = os.path.join(self.img_dir, study_id, series_id)
        file_list = natural_sort(os.listdir(series_dir))

        if len(file_list) > img_num:
            start_index = (len(file_list)-img_num) // 2
            file_list = file_list[start_index:start_index + img_num]

        for i, filename in enumerate(file_list):
            ds = pydicom.dcmread(os.path.join(series_dir, filename))
            img = ds.pixel_array.astype(np.float32)
            img = cv2.resize(img, self.resolution, interpolation=cv2.INTER_CUBIC)

            x[..., i] = img

        # Standardize series
        x = (x - x.mean()) / x.std()

        if kpmap:
            keypoint_img = np.zeros(self.resolution, dtype=np.float32)
            for x_kp, y_kp in keypoints:
                x_kp, y_kp = int(x_kp), int(y_kp)
                cv2.circle(keypoint_img, (x_kp, y_kp), 20, 1, -1)
            x[..., -1] = keypoint_img
        
        return x, keypoints

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

        # Sagittal T1
        x1, kp1 = self.get_series_with_coord(row.study_id, 'Sagittal T1', img_num=self.img_num[0])
        x2, kp2 = self.get_series_with_coord(row.study_id, 'Sagittal T2/STIR', img_num=self.img_num[1])
        x3, kp3 = self.get_series_with_coord(row.study_id, 'Axial T2', img_num=self.img_num[2])
        
        if self.transform:
            t1 = self.transform(image=x1, keypoints=kp1)
            x1, kp1 = t1['image'], t1['keypoints']
            t2 = self.transform(image=x2, keypoints=kp2)
            x2, kp2 = t2['image'], t2['keypoints']
            t3 = self.transform(image=x3, keypoints=kp3)
            x3, kp3 = t3['image'], t3['keypoints']

        keypoints = np.concatenate((np.array(kp1).flatten(), np.array(kp2).flatten(), np.array(kp3).flatten())).astype(np.float32)
        if self.resolution[0] == self.resolution[1]:
            keypoints /= self.resolution[0]

        return x1, x2, x3, (label, keypoints)
    
class RSNA2024SplitBestseriesDataset(RSNA2024SplitCoordDataset):
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        label = self.df[self.out_vars].iloc[idx].values
        label = np.nan_to_num(label.astype(float), nan=0).astype(np.int64)

        # Sagittal T1
        x1, _ = self.get_series_with_coord(row.study_id, 'Sagittal T1', img_num=self.img_num[0])
        x2, _ = self.get_series_with_coord(row.study_id, 'Sagittal T2/STIR', img_num=self.img_num[1])
        x3, _ = self.get_series_with_coord(row.study_id, 'Axial T2', img_num=self.img_num[2])
        
        if self.transform:
            x1 = self.transform(image=x1)['image']
            x2 = self.transform(image=x2)['image']
            x3 = self.transform(image=x3)['image']

        return x1, x2, x3, label
    
class RSNA2024SplitKpmapDataset(RSNA2024Dataset):
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        label = self.df[self.out_vars].iloc[idx].values
        label = np.nan_to_num(label.astype(float), nan=0).astype(np.int64)

        # Sagittal T1
        x1, _ = self.get_series_with_coord(row.study_id, 'Sagittal T1', img_num=self.img_num[0], kpmap=True)
        x2, _ = self.get_series_with_coord(row.study_id, 'Sagittal T2/STIR', img_num=self.img_num[1], kpmap=True)
        x3, _ = self.get_series_with_coord(row.study_id, 'Axial T2', img_num=self.img_num[2], kpmap=True)
        
        if self.transform:
            x1 = self.transform(image=x1)['image']
            x2 = self.transform(image=x2)['image']
            x3 = self.transform(image=x3)['image']

        return x1, x2, x3, label,
