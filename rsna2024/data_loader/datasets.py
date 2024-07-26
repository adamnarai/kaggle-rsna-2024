import os
import random
from itertools import compress
import pandas as pd
import numpy as np
import pydicom
import cv2

import torch
from torch.utils.data import Dataset

from rsna2024.utils import natural_sort


class BaseDataset(Dataset):
    def __init__(
        self,
        df,
        root_dir,
        data_dir,
        out_vars,
        img_num,
        resolution,
        block_position=('middle', 'middle', 'middle'),
        series_mask=(0, 0, 0),
        transform=None,
    ):
        self.df = df
        self.data_dir = os.path.join(root_dir, data_dir)
        self.df_series = self.load_series_info()
        self.df_coordinates = self.load_coordinates_info().merge(
            self.df_series, how='left', on=['study_id', 'series_id']
        )
        self.img_dir = os.path.join(self.data_dir, 'train_images')
        self.out_vars = out_vars
        self.transform = transform
        self.img_num = img_num
        self.resolution = resolution
        self.block_position = block_position
        self.series_mask = series_mask
        self.series_descriptions = ['Sagittal T1', 'Sagittal T2/STIR', 'Axial T2']

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        label = self.df[self.out_vars].iloc[idx].values
        label = np.nan_to_num(label.astype(float), nan=0).astype(np.int64)

        x1, x2, x3 = None, None, None
        if self.series_mask[0]:
            x1 = self.get_series(
                row.study_id,
                'Sagittal T1',
                img_num=self.img_num[0],
                block_position=self.block_position[0],
            )
        if self.series_mask[1]:
            x2 = self.get_series(
                row.study_id,
                'Sagittal T2/STIR',
                img_num=self.img_num[1],
                block_position=self.block_position[1],
            )
        if self.series_mask[2]:
            x3 = self.get_series(
                row.study_id,
                'Axial T2',
                img_num=self.img_num[2],
                block_position=self.block_position[2],
            )

        x = np.concatenate(list(compress([x1, x2, x3], self.series_mask)), axis=2)

        if self.transform:
            x = self.transform(image=x)['image']

        return x, label

    def load_series_info(self):
        return pd.read_csv(
            os.path.join(self.data_dir, 'train_series_descriptions.csv'),
            dtype={'study_id': 'str', 'series_id': 'str'},
        )

    def load_coordinates_info(self):
        df_coordinates = pd.read_csv(
            os.path.join(self.data_dir, '..', 'processed', 'train_label_coordinates.csv'),
            dtype={'study_id': 'str', 'series_id': 'str'},
        )
        return df_coordinates

    def get_random_series_id(self, study_id, series_description):
        series_list = self.df_series[
            (self.df_series['study_id'] == study_id)
            & (self.df_series['series_description'] == series_description)
        ]['series_id'].tolist()

        if len(series_list) == 0:
            return None

        return random.sample(series_list, 1)[0]

    def get_series(self, study_id, series_description, img_num, block_position='middle'):
        x = np.zeros((*self.resolution, img_num), dtype=np.float32)
        series_id = self.get_random_series_id(study_id, series_description)
        if series_id is None:
            return x

        series_dir = os.path.join(self.img_dir, study_id, series_id)
        file_list = natural_sort(os.listdir(series_dir))
        slice_num = len(file_list)

        # Fix direction
        ds_first = pydicom.dcmread(os.path.join(series_dir, file_list[0]))
        ds_last = pydicom.dcmread(os.path.join(series_dir, file_list[-1]))
        pos_diff = np.array(ds_last.ImagePositionPatient) - np.array(ds_first.ImagePositionPatient)
        pos_diff = pos_diff[np.abs(pos_diff).argmax()]
        if pos_diff < 0:
            file_list.reverse()

        if slice_num > img_num:
            if block_position == 'start':
                file_list = file_list[:img_num]
            elif block_position == 'end':
                file_list = file_list[-img_num:]
            elif block_position == 'middle':
                start_index = (slice_num - img_num) // 2
                file_list = file_list[start_index : start_index + img_num]

        for i, filename in enumerate(file_list):
            ds = pydicom.dcmread(os.path.join(series_dir, filename))
            img = ds.pixel_array.astype(np.float32)
            img = cv2.resize(img, self.resolution, interpolation=cv2.INTER_CUBIC)

            x[..., i] = img

        # Standardize series
        x = (x - x.mean()) / x.std()

        return x


class SplitDataset(BaseDataset):
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        label = self.df[self.out_vars].iloc[idx].values
        label = np.nan_to_num(label.astype(float), nan=0).astype(np.int64)

        # Sagittal T1
        x1 = self.get_series(
            row.study_id,
            'Sagittal T1',
            img_num=self.img_num[0],
            block_position=self.block_position[0],
        )
        x2 = self.get_series(
            row.study_id,
            'Sagittal T2/STIR',
            img_num=self.img_num[1],
            block_position=self.block_position[1],
        )
        x3 = self.get_series(
            row.study_id, 'Axial T2', img_num=self.img_num[2], block_position=self.block_position[2]
        )

        if self.transform:
            x1 = self.transform(image=x1)['image']
            x2 = self.transform(image=x2)['image']
            x3 = self.transform(image=x3)['image']

        return x1, x2, x3, label


class SplitCoordDataset(BaseDataset):
    def __init__(self, df, data_dir, out_vars, img_num, resolution, heatmap_std, transform=None):
        self.df = df
        self.df_series = self.load_series_info(data_dir)
        self.df_coordinates = self.load_coordinates_info(data_dir).merge(
            self.df_series, how='left', on=['study_id', 'series_id']
        )
        self.img_dir = os.path.join(data_dir, 'train_images')
        self.out_vars = out_vars
        self.transform = transform
        self.img_num = img_num
        self.resolution = resolution
        self.heatmap_std = heatmap_std
        self.series_descriptions = ['Sagittal T1', 'Sagittal T2/STIR', 'Axial T2']

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        x1, coord1, slicenum1 = self.get_series_with_coord(
            row.study_id, 'Sagittal T2/STIR', img_num=self.img_num[0]
        )
        coord1 = np.array(coord1, dtype=np.float32)

        def gaussian_heatmap(width, height, center, std_dev):
            """
            Args:
            - width (int): Width of the heatmap
            - height (int): Height of the heatmap
            - center (tuple): The (x, y) coordinates of the Gaussian peak
            - std_dev (int, optional): Standard deviation of the Gaussian

            """
            x_axis = torch.arange(width).float() - center[0]
            y_axis = torch.arange(height).float() - center[1]
            x, y = torch.meshgrid(y_axis, x_axis, indexing='ij')

            return torch.exp(-((x**2 + y**2) / (2 * std_dev**2)))

        heatmaps = []
        for i in range(5):
            if np.isnan(coord1[i, :]).any():
                heatmaps.append(torch.zeros(*self.resolution))
            else:
                heatmaps.append(
                    gaussian_heatmap(*self.resolution, coord1[i, :], std_dev=self.heatmap_std)
                )
        heatmaps = torch.stack(heatmaps, dim=-1).numpy()

        if self.transform:
            t1 = self.transform(image=x1, mask=heatmaps)
            x1, heatmaps = t1['image'], t1['mask']
            # t2 = self.transform(image=x2, keypoints=kp2)
            # x2, kp2 = t2['image'], t2['keypoints']
            # t3 = self.transform(image=x3, keypoints=kp3)
            # x3, kp3 = t3['image'], t3['keypoints']

        heatmaps = np.transpose(heatmaps, (2, 0, 1))

        return x1, heatmaps

    def get_series_with_coord(
        self, study_id, series_description, img_num, series_id_selection='random'
    ):
        x = np.zeros((*self.resolution, img_num), dtype=np.float32)

        # Get expected vars in standard order
        if series_description == 'Sagittal T1':
            expected_vars = [x for x in self.out_vars if 'neural_foraminal_narrowing' in x]
        elif series_description == 'Sagittal T2/STIR':
            expected_vars = [x for x in self.out_vars if 'spinal_canal_stenosis' in x]
        elif series_description == 'Axial T2':
            expected_vars = [x for x in self.out_vars if 'subarticular_stenosis' in x]

        # Get coords in standard order, padded with nans
        study_coord = self.df_coordinates[self.df_coordinates['study_id'] == study_id]
        series_coord = study_coord[study_coord['series_description'] == series_description]
        series_coord_padded = series_coord.merge(
            pd.DataFrame({'row_id': expected_vars}), on='row_id', how='right'
        )

        # Return if no series
        series_list = series_coord['series_id'].unique().tolist()
        if len(series_list) == 0:
            return (
                x,
                list(zip([np.nan] * len(expected_vars), [np.nan] * len(expected_vars))),
                np.ones(len(expected_vars), dtype=np.float32) * np.nan,
            )

        # Select series_id
        if series_id_selection == 'random':
            series_id = random.sample(series_list, 1)[0]
        elif series_id_selection == 'concat':
            raise NotImplementedError('Concat series_id selection method not implemented')
        else:
            raise ValueError('Invalid series_id selection method')
        series_dir = os.path.join(self.img_dir, study_id, series_id)
        file_list = natural_sort(os.listdir(series_dir))
        slice_num = len(file_list)

        # Fix direction
        ds_first = pydicom.dcmread(os.path.join(series_dir, file_list[0]))
        ds_last = pydicom.dcmread(os.path.join(series_dir, file_list[-1]))
        pos_diff = np.array(ds_last.ImagePositionPatient) - np.array(ds_first.ImagePositionPatient)
        pos_diff = pos_diff[np.abs(pos_diff).argmax()]
        if pos_diff < 0:
            file_list.reverse()

        def fix_dir(row):
            if row.slice_dir == 1:
                return 1 - row.instance_number_norm
            elif row.slice_dir == 0:
                return row.instance_number_norm

        series_coord_padded['instance_number_norm'] = series_coord_padded.apply(fix_dir, axis=1)

        # Select slices
        if slice_num > img_num:
            if series_description == 'Sagittal T2/STIR':
                # centered around the middle slice
                start_index = (slice_num - img_num) // 2
                file_list = file_list[start_index : start_index + img_num]
            elif series_description == 'Sagittal T1':
                raise NotImplementedError('Sagittal T1 slice selection method not implemented')
            elif series_description == 'Axial T2':
                raise NotImplementedError('Axial T2 slice selection method not implemented')
        elif slice_num < img_num:
            # pad with None symmetrically
            file_list = (
                [None] * ((img_num - slice_num) // 2)
                + file_list
                + [None] * ((img_num - slice_num) // 2 + (img_num - slice_num) % 2)
            )

        # Load data
        for i, filename in enumerate(file_list):
            if filename is None:
                continue
            ds = pydicom.dcmread(os.path.join(series_dir, filename))
            img = ds.pixel_array.astype(np.float32)
            img = cv2.resize(img, self.resolution, interpolation=cv2.INTER_CUBIC)
            x[..., i] = img

        # Standardize series
        x = (x - x.mean()) / x.std()

        coordinates = list(
            zip(
                series_coord_padded['x_norm'].values * self.resolution[0],
                series_coord_padded['y_norm'].values * self.resolution[1],
            )
        )
        slice_num = series_coord_padded['instance_number_norm'].values.astype(np.float32)

        return x, coordinates, slice_num


class TilesSagt2Dataset(Dataset):
    def __init__(self, df, data_dir, img_num, resolution, proportion, labels, transform=None):
        self.df = df
        self.img_dir = os.path.join(
            os.path.dirname(data_dir),
            'processed',
            'tiles_sagt2',
            f'imgnum{img_num}_prop{int(proportion*100)}_res{resolution}',
        )
        df_tiles = pd.read_csv(
            os.path.join(self.img_dir, 'info.csv'), dtype={'study_id': 'str', 'series_id': 'str'}
        )
        df_tiles['label'] = df_tiles['label'].map(labels)
        self.df_tiles = df_tiles.merge(self.df[['study_id']], how='inner', on='study_id')
        self.transform = transform

    def __len__(self):
        return len(self.df_tiles)

    def __getitem__(self, idx):
        row = self.df_tiles.iloc[idx]
        label = row.label
        label = np.nan_to_num(label.astype(float), nan=0).astype(np.int64)

        filename = f'{row.study_id}_{row.row_id[-5:]}.npy'
        x = np.load(os.path.join(self.img_dir, filename))

        if self.transform:
            x = self.transform(image=x)['image']

        return x, label


class MeanposDataset(BaseDataset):
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        label = self.df[self.out_vars].iloc[idx].values
        label = np.nan_to_num(label.astype(float), nan=0).astype(np.int64)

        x1, x2, x3 = None, None, None
        if self.series_mask[0]:
            x1 = self.get_series_meanpos(row.study_id, 'Sagittal T1', img_num=self.img_num[0])
        if self.series_mask[1]:
            x2 = self.get_series_meanpos(row.study_id, 'Sagittal T2/STIR', img_num=self.img_num[1])
        if self.series_mask[2]:
            x3 = self.get_series_meanpos(row.study_id, 'Axial T2', img_num=self.img_num[2])

        x = np.concatenate(list(compress([x1, x2, x3], self.series_mask)), axis=2)

        if self.transform:
            x = self.transform(image=x)['image']

        return x, label

    def get_series_meanpos(self, study_id, series_description, img_num):
        if series_description == 'Sagittal T1':
            img_num_mult = 2
        elif series_description == 'Sagittal T2/STIR':
            img_num_mult = 1
        elif series_description == 'Axial T2':
            img_num_mult = 5
        x = np.zeros((*self.resolution, img_num * img_num_mult), dtype=np.float32)
        series_id = self.get_random_series_id(study_id, series_description)
        if series_id is None:
            return x

        series_dir = os.path.join(self.img_dir, study_id, series_id)
        file_list = natural_sort(os.listdir(series_dir))
        slice_num = len(file_list)

        # Fix direction
        ds_first = pydicom.dcmread(os.path.join(series_dir, file_list[0]))
        ds_last = pydicom.dcmread(os.path.join(series_dir, file_list[-1]))
        pos_diff = np.array(ds_last.ImagePositionPatient) - np.array(ds_first.ImagePositionPatient)
        pos_diff = pos_diff[np.abs(pos_diff).argmax()]
        if pos_diff < 0:
            file_list.reverse()

        if slice_num > img_num:
            if series_description == 'Sagittal T1':
                lef_start = round(slice_num * 0.7040) - img_num // 2
                right_start = round(slice_num * 0.2592) - img_num // 2
                file_list = (
                    file_list[lef_start : lef_start + img_num]
                    + file_list[right_start : right_start + img_num]
                )
            elif series_description == 'Sagittal T2/STIR':
                midline_start = round(slice_num * 0.4925) - img_num // 2
                file_list = file_list[midline_start : midline_start + img_num]
            elif series_description == 'Axial T2':
                l1_start = round(slice_num * 0.8160) - img_num // 2
                l2_start = round(slice_num * 0.6502) - img_num // 2
                l3_start = round(slice_num * 0.4925) - img_num // 2
                l4_start = round(slice_num * 0.3374) - img_num // 2
                l5_start = round(slice_num * 0.1617) - img_num // 2
                file_list = (
                    file_list[l1_start : l1_start + img_num]
                    + file_list[l2_start : l2_start + img_num]
                    + file_list[l3_start : l3_start + img_num]
                    + file_list[l4_start : l4_start + img_num]
                    + file_list[l5_start : l5_start + img_num]
                )
        elif slice_num < img_num:  # pad with None symmetrically
            file_list = (
                [None] * ((img_num - slice_num) // 2)
                + file_list
                + [None] * ((img_num - slice_num) // 2)
            )

        for i, filename in enumerate(file_list):
            if filename is None:
                continue
            ds = pydicom.dcmread(os.path.join(series_dir, filename))
            img = ds.pixel_array.astype(np.float32)
            img = cv2.resize(img, self.resolution, interpolation=cv2.INTER_CUBIC)

            x[..., i] = img

        # Standardize series
        x = (x - x.mean()) / x.std()

        return x


class SplitMeanposDataset(MeanposDataset):
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        label = self.df[self.out_vars].iloc[idx].values
        label = np.nan_to_num(label.astype(float), nan=0).astype(np.int64)

        # Sagittal T1
        x1 = self.get_series_meanpos(row.study_id, 'Sagittal T1', img_num=self.img_num[0])
        x2 = self.get_series_meanpos(row.study_id, 'Sagittal T2/STIR', img_num=self.img_num[1])
        x3 = self.get_series_meanpos(row.study_id, 'Axial T2', img_num=self.img_num[2])

        if self.transform:
            x1 = self.transform(image=x1)['image']
            x2 = self.transform(image=x2)['image']
            x3 = self.transform(image=x3)['image']

        return x1, x2, x3, label


class MilSplitDataset(BaseDataset):
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        label = self.df[self.out_vars].iloc[idx].values
        label = np.nan_to_num(label.astype(float), nan=0).astype(np.int64)

        # Sagittal T1
        x1 = self.get_full_series(row.study_id, 'Sagittal T1')
        x2 = self.get_full_series(row.study_id, 'Sagittal T2/STIR')
        x3 = self.get_full_series(row.study_id, 'Axial T2')
        x1 = self.get_instances(x1, self.img_num[0], ch_num=5)
        x2 = self.get_instances(x2, self.img_num[1], ch_num=5)
        x3 = self.get_instances(x3, self.img_num[2], ch_num=5)

        if self.transform:
            x1 = self.transform(image=x1)['image']
            x2 = self.transform(image=x2)['image']
            x3 = self.transform(image=x3)['image']

        return x1, x2, x3, label

    def get_full_series(self, study_id, series_description):
        series_id = self.get_random_series_id(study_id, series_description)
        if series_id is None:
            return None

        series_dir = os.path.join(self.img_dir, study_id, series_id)
        file_list = natural_sort(os.listdir(series_dir))

        img_list = []
        for filename in file_list:
            ds = pydicom.dcmread(os.path.join(series_dir, filename))
            img = ds.pixel_array.astype(np.float32)
            img = cv2.resize(img, self.resolution, interpolation=cv2.INTER_CUBIC)

            img_list.append(img)
        x = np.stack(img_list, axis=2)

        # Standardize series
        x = (x - x.mean()) / x.std()

        return x

    def get_instances(self, x, img_num, ch_num):
        if x is None:
            return np.zeros((*self.resolution, img_num * ch_num), dtype=np.float32)
        offset = ch_num // 2
        slice_num = x.shape[-1]
        step = slice_num / (img_num + 1)
        st = step / 2.0
        end = slice_num - 1 - step / 2.0
        idx_list = np.linspace(st, end, img_num).astype(int)

        instances = []
        for i in idx_list:
            if i - offset < 0:
                start_idx = abs(i - offset)
            else:
                start_idx = 0
            data = x[..., max(0, i - offset) : min(slice_num, i + offset + 1)]
            end_idx = start_idx + data.shape[-1]
            instance = np.zeros((*x.shape[:-1], ch_num), dtype=np.float32)
            instance[..., start_idx:end_idx] = data
            instances.append(instance)
        return np.concatenate(instances, axis=2)
