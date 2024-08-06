import os
import random
from itertools import compress
import pandas as pd
import numpy as np
import pydicom
import cv2
import warnings

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
        elif slice_num < img_num:
            # pad with None symmetrically
            file_list = (
                [None] * ((img_num - slice_num) // 2)
                + file_list
                + [None] * ((img_num - slice_num) // 2 + (img_num - slice_num) % 2)
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
    def __init__(
        self,
        df,
        root_dir,
        data_dir,
        out_vars,
        img_num,
        resolution,
        heatmap_std,
        series_description='Sagittal T2/STIR',
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
        self.heatmap_std = heatmap_std
        self.series_description = series_description
        
        # TODO: future remove
        if not isinstance(self.img_num, int) and len(self.img_num) > 1:
            self.img_num = self.img_num[0]

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        x, coord, instance_num = self.get_series_with_coord(
            row.study_id, self.series_description, img_num=self.img_num
        )
        coord = np.array(coord, dtype=np.float32)

        heatmaps = []
        for i in range(5):
            if np.isnan(coord[i, :]).any():
                heatmaps.append(torch.zeros(*self.resolution))
            else:
                heatmaps.append(
                    self.gaussian_heatmap(*self.resolution, coord[i, :], std_dev=self.heatmap_std)
                )
        heatmaps = torch.stack(heatmaps, dim=-1).numpy()

        if self.transform:
            t = self.transform(image=x, mask=heatmaps)
            x, heatmaps = t['image'], t['mask']

        heatmaps = np.transpose(heatmaps, (2, 0, 1))

        return x, heatmaps

    def gaussian_heatmap(self, width, height, center, std_dev):
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

        if series_description == 'Sagittal T2/STIR':
            series_coord_padded['instance_number_norm'] = series_coord_padded.apply(fix_dir, axis=1)
            instance_num = series_coord_padded['instance_number_norm'].values.astype(np.float32)
        elif series_description == 'Sagittal T1':
            # Average left and right coordinates
            series_coord_padded = (
                series_coord_padded.groupby(['study_id', 'series_id', 'level'])
                .agg({'x_norm': 'mean', 'y_norm': 'mean', 'slice_dir': 'first'})
                .reset_index()
            )
            series_coord_padded['level'] = (
                series_coord_padded['level'].str.replace('/', '_').str.lower()
            )
            instance_num = None

        coordinates = list(
            zip(
                series_coord_padded['x_norm'].values * self.resolution[1],
                series_coord_padded['y_norm'].values * self.resolution[0],
            )
        )

        # Select slices
        if slice_num > img_num:
            if series_description == 'Sagittal T2/STIR':
                # centered around the middle slice
                start_index = (slice_num - img_num) // 2
                file_list = file_list[start_index : start_index + img_num]
            elif series_description == 'Sagittal T1':
                # two blocks of slices at 25% and 75%
                right_start = round(slice_num * 0.25) - img_num // 4
                lef_start = round(slice_num * 0.75) - img_num // 4
                file_list = (
                    file_list[right_start : right_start + img_num // 2]
                    + file_list[lef_start : lef_start + img_num // 2]
                )
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

        return x, coordinates, instance_num


class TilesSagt2Dataset:
    def __init__(
        self, df, root_dir, data_dir, img_num, resolution, proportion, labels, transform=None
    ):
        self.df = df
        self.data_dir = os.path.join(root_dir, data_dir)
        self.img_dir = os.path.join(
            self.data_dir,
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


class TilesSagt1Dataset:
    def __init__(
        self, df, root_dir, data_dir, img_num, resolution, proportion, labels, transform=None
    ):
        self.df = df
        self.data_dir = os.path.join(root_dir, data_dir)
        self.img_dir = os.path.join(
            self.data_dir,
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
        side = row.row_id.split('_')[0]
        filename = f'{row.study_id}_{row.row_id[-5:]}_{side}.npy'

        x = np.load(os.path.join(self.img_dir, filename))

        if self.transform:
            x = self.transform(image=x)['image']

        return x, label
    

class TilesAxiDataset:
    def __init__(
        self, df, root_dir, data_dir, img_num, resolution, proportion, labels, transform=None
    ):
        self.df = df
        self.data_dir = os.path.join(root_dir, data_dir)
        self.img_dir = os.path.join(
            self.data_dir,
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
        label = np.nan_to_num(row.label.astype(float), nan=0).astype(np.int64)

        filename = f'{row.study_id}_{row.row_id[-5:]}_{row.row_id.split('_')[0]}.npy'
        x = np.load(os.path.join(self.img_dir, filename))

        if self.transform:
            x = self.transform(image=x)['image']

        return x, label
    
class AxiCoordDataset:
    def __init__(
        self, df, root_dir, data_dir, img_num, resolution, heatmap_std, proportion, transform=None
    ):
        self.df = df
        self.data_dir = os.path.join(root_dir, data_dir)
        self.img_dir = os.path.join(
            self.data_dir,
            f'imgnum{img_num}_prop{int(proportion*100)}_res{resolution}',
        )
        df_tiles = pd.read_csv(
            os.path.join(self.img_dir, 'info.csv'), dtype={'study_id': 'str', 'series_id': 'str'}
        )
        self.df_tiles = df_tiles.merge(self.df[['study_id']], how='inner', on='study_id')
        self.heatmap_std = heatmap_std
        self.resolution = resolution
        self.transform = transform

    def __len__(self):
        return len(self.df_tiles)

    def __getitem__(self, idx):
        row = self.df_tiles.iloc[idx]
        
        filename = f'{row.study_id}_{row.row_id[-5:]}.npy'
        x = np.load(os.path.join(self.img_dir, filename))

        coord = np.array([[row.left_x_norm * self.resolution,
                           row.left_y_norm * self.resolution],
                          [row.right_x_norm * self.resolution,
                           row.right_y_norm * self.resolution]], dtype=np.float32)
        heatmaps = []
        for i in range(2):
            if np.isnan(coord[i, :]).any():
                heatmaps.append(torch.zeros(self.resolution, self.resolution))
            else:
                heatmaps.append(
                    self.gaussian_heatmap(self.resolution, self.resolution, coord[i, :], std_dev=self.heatmap_std)
                )
        heatmaps = torch.stack(heatmaps, dim=-1).numpy()

        if self.transform:
            t = self.transform(image=x, mask=heatmaps)
            x, heatmaps = t['image'], t['mask']

        heatmaps = np.transpose(heatmaps, (2, 0, 1))

        return x, heatmaps

    def gaussian_heatmap(self, width, height, center, std_dev):
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

class LevelROIDataset(Dataset):
    def __init__(
        self,
        df,
        root_dir,
        data_dir,
        img_num,
        resolution,
        roi_size,
        phase='train',
        transform=None,
    ):
        self.conditions = [
            'spinal_canal_stenosis',
            'left_neural_foraminal_narrowing',
            'right_neural_foraminal_narrowing',
            'left_subarticular_stenosis',
            'right_subarticular_stenosis',
        ]
        self.levels = ['l1_l2', 'l2_l3', 'l3_l4', 'l4_l5', 'l5_s1']
        self.df = pd.wide_to_long(
            df,
            self.conditions,
            i='study_id',
            j='level',
            sep='_',
            suffix=r'\w+',
        ).reset_index()
        self.data_dir = os.path.join(root_dir, data_dir)
        self.df_series = self.load_series_info()
        self.df_coordinates = self.load_coordinates_info(root_dir).merge(
            self.df_series, how='left', on=['study_id', 'series_id']
        )
        self.img_dir = os.path.join(self.data_dir, 'train_images')
        self.transform = transform
        self.img_num = img_num
        self.resolution = resolution
        self.roi_size = roi_size
        self.phase = phase

    def load_series_info(self):
        return pd.read_csv(
            os.path.join(self.data_dir, 'train_series_descriptions.csv'),
            dtype={'study_id': 'str', 'series_id': 'str'},
        )

    def load_coordinates_info(self, root_dir):
        return pd.read_csv(
            os.path.join(root_dir, 'data', 'processed', 'train_label_coordinates.csv'),
            dtype={'study_id': 'str', 'series_id': 'str'},
        )

    def __len__(self):
        return len(self.df)

    def get_series(self, study_id, series_description):
        series_list = self.df_series[
            (self.df_series['study_id'] == study_id)
            & (self.df_series['series_description'] == series_description)
        ]['series_id'].tolist()

        if len(series_list) == 0:
            # warnings.warn(f'No series found for study_id {study_id} and series_description {series_description}')
            return None

        if len(series_list) > 1:
            # warnings.warn(f'Multiple series found for study_id {study_id} and series_description {series_description}')
            pass

        return series_list

    def get_sagt2_coord(self, study_id, series_id, level):
        if self.phase == 'train':
            level = level.replace('_', '/').upper()
            coords = self.df_coordinates[
                (self.df_coordinates['study_id'] == study_id)
                & (self.df_coordinates['series_id'] == series_id)
                & (self.df_coordinates['level'] == level)
            ][['x_norm', 'y_norm', 'instance_number']].values

        if coords.shape[0] == 0:
            # warnings.warn(f'No coordinates found for study_id {study_id} and series_id {series_id} and level {level}')
            return (np.nan, np.nan, np.nan)

        if coords.shape[0] > 1:
            # warnings.warn(f'Multiple coordinates found for study_id {study_id} and series_id {series_id} and level {level}')
            pass

        x_norm, y_norm, instance_number = coords[0]
        return x_norm, y_norm, int(instance_number)

    def get_sagt1_coord(self, study_id, series_id, level, side):
        if self.phase == 'train':
            level = level.replace('_', '/').upper()
            df_coordinates_filtered = self.df_coordinates[
                (self.df_coordinates['study_id'] == study_id)
                & (self.df_coordinates['series_id'] == series_id)
                & (self.df_coordinates['level'] == level)
            ]
            coords = df_coordinates_filtered[
                df_coordinates_filtered['row_id'].str.startswith(side)
            ][['x_norm', 'y_norm', 'instance_number']].values

        if coords.shape[0] == 0:
            # warnings.warn(f'No coordinates found for study_id {study_id} and series_id {series_id} and level {level}')
            return np.nan, np.nan, np.nan

        if coords.shape[0] > 1:
            # warnings.warn(f'Multiple coordinates found for study_id {study_id} and series_id {series_id} and level {level}')
            pass

        x_norm, y_norm, instance_number = coords[0]
        return x_norm, y_norm, int(instance_number)
    
    def get_axi_coord(self, study_id, level, side):
        if self.phase == 'train':
            level = level.replace('_', '/').upper()
            df_coordinates_filtered = self.df_coordinates[
                (self.df_coordinates['study_id'] == study_id)
                & (self.df_coordinates['series_description'] == 'Axial T2')
                & (self.df_coordinates['level'] == level)
            ]
            coords = df_coordinates_filtered[
                df_coordinates_filtered['row_id'].str.startswith(side)
            ][['x_norm', 'y_norm', 'instance_number', 'series_id']].values

        if coords[0] == 0:
            # warnings.warn(f'No coordinates found for study_id {study_id} and series_id {series_id} and level {level}')
            return np.nan, np.nan, np.nan, ''

        if coords.shape[0] > 1:
            # warnings.warn(f'Multiple coordinates found for study_id {study_id} and series_id {series_id} and level {level}')
            pass

        x_norm, y_norm, instance_number, series_id = coords[0]
        return x_norm, y_norm, int(instance_number), str(series_id)

    def get_roi(
        self,
        study_id,
        series_id,
        img_num,
        resolution,
        roi_size,
        x_norm=np.nan,
        y_norm=np.nan,
        instance_number_type='middle',  # 'middle' 'index', 'filename', 'relative'
        instance_number=None,
        interpolation=cv2.INTER_CUBIC,
        standardize=True,
    ):
        x = np.zeros((resolution, resolution, img_num), dtype=np.float32)
        if np.isnan(x_norm) or np.isnan(y_norm):
            return x
        series_dir = os.path.join(self.img_dir, str(study_id), str(series_id))
        file_list = natural_sort(os.listdir(series_dir))
        slice_num = len(file_list)
        if slice_num == 0:
            return x

        # Fix direction
        ds_first = pydicom.dcmread(os.path.join(series_dir, file_list[0]))
        ds_last = pydicom.dcmread(os.path.join(series_dir, file_list[-1]))
        pos_diff = np.array(ds_last.ImagePositionPatient) - np.array(ds_first.ImagePositionPatient)
        pos_diff = pos_diff[np.abs(pos_diff).argmax()]
        if pos_diff < 0:
            file_list.reverse()
            if instance_number_type == 'index':
                instance_number = len(file_list) - instance_number - 1

        if instance_number_type == 'middle':
            start_index = (slice_num - img_num) // 2
        elif instance_number_type == 'index':
            start_index = max(instance_number - img_num // 2, 0)
        elif instance_number_type == 'filename':
            start_index = max(
                [int(file.rstrip('.dcm')) for file in file_list].index(instance_number)
                - img_num // 2,
                0,
            )
        elif instance_number_type == 'relative':
            start_index = max(int(instance_number * slice_num) - img_num // 2, 0)
        end_index = min(start_index + img_num, slice_num)
        file_list = file_list[start_index:end_index]

        for i, filename in enumerate(file_list):
            ds = pydicom.dcmread(os.path.join(series_dir, filename))
            img = ds.pixel_array.astype(np.float32)

            if i == 0:
                x_norm = x_norm * img.shape[1]
                y_norm = y_norm * img.shape[0]

            # Crop ROI
            size_x = roi_size / ds.PixelSpacing[1]
            size_y = roi_size / ds.PixelSpacing[0]
            x1 = round(x_norm - (size_x / 2))
            x2 = round(x_norm + (size_x / 2))
            y1 = round(y_norm - (size_y / 2))
            y2 = round(y_norm + (size_y / 2))
            if any([x1 < 0, x2 > img.shape[1], y1 < 0, y2 > img.shape[0]]):
                # warnings.warn(f'ROI out of bounds for study_id {study_id} and series_id {series_id}')
                break
            img = img[y1:y2, x1:x2]

            # Resize
            img = cv2.resize(img, (resolution, resolution), interpolation=interpolation)
            x[..., i] = img

        # Standardize image
        if standardize and x.std() != 0:
            x = (x - x.mean()) / x.std()

        return x

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        label = self.df[self.conditions].iloc[idx].values
        label = np.nan_to_num(label.astype(float), nan=0).astype(np.int64)
        level = np.zeros(len(self.levels), dtype=np.float32)
        level[self.levels.index(row.level)] = 1.0

        # Sagittal T2/STIR central ROI
        sagt2_series_id = self.get_series(row.study_id, 'Sagittal T2/STIR')
        sagt2_series_id = sagt2_series_id[0] if sagt2_series_id is not None else None
        sagt2_x_norm, sagt2_y_norm, sagt2_instance_number = self.get_sagt2_coord(
            row.study_id, sagt2_series_id, row.level
        )
        sagt2_roi = self.get_roi(
            study_id=row.study_id,
            series_id=sagt2_series_id,
            img_num=self.img_num,
            resolution=self.resolution,
            roi_size=self.roi_size,
            x_norm=sagt2_x_norm,
            y_norm=sagt2_y_norm,
            instance_number_type='middle',
        )
        
        # Axial T2 central ROI
        axi_left_x_norm, axi_left_y_norm, axi_left_instance_number, axi_left_series_id = self.get_axi_coord(row.study_id, row.level, 'left')
        axi_right_x_norm, axi_right_y_norm, axi_right_instance_number, axi_right_series_id = self.get_axi_coord(row.study_id, row.level, 'right')
        
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            axi_x_norm = np.nanmean([axi_left_x_norm, axi_right_x_norm])
            axi_y_norm = np.nanmean([axi_left_y_norm, axi_right_y_norm])
        if not np.isnan(axi_left_instance_number):
            axi_instance_number = axi_left_instance_number
            axi_series_id = axi_left_series_id
        else:
            axi_instance_number = axi_right_instance_number
            axi_series_id = axi_right_series_id
            
        axi_roi = self.get_roi(
            study_id=row.study_id,
            series_id=axi_series_id,
            img_num=self.img_num,
            resolution=self.resolution,
            roi_size=self.roi_size,
            x_norm=axi_x_norm,
            y_norm=axi_y_norm,
            instance_number_type='filename',
            instance_number=axi_instance_number,
        )

        if self.transform:
            sagt2_roi = self.transform(image=sagt2_roi)['image']
            # sagt1_roi = self.transform(image=sagt1_roi)['image']
            axi_roi = self.transform(image=axi_roi)['image']
            

        return sagt2_roi, axi_roi, level, label[0]


class LevelSideROIDataset(LevelROIDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sides = ['left', 'right']
        
        self.df.rename(
            columns={
                'left_neural_foraminal_narrowing': 'neural_foraminal_narrowing_left',
                'right_neural_foraminal_narrowing': 'neural_foraminal_narrowing_right',
                'left_subarticular_stenosis': 'subarticular_stenosis_left',
                'right_subarticular_stenosis': 'subarticular_stenosis_right',
            },
            inplace=True,
        )
        self.df = pd.wide_to_long(
            self.df,
            ['neural_foraminal_narrowing', 'subarticular_stenosis'],
            i=['study_id', 'level'],
            j='side',
            sep='_',
            suffix=r'\w+',
        ).reset_index()
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        label = self.df[self.conditions].iloc[idx].values
        label = np.nan_to_num(label.astype(float), nan=0).astype(np.int64)
        level = np.zeros(len(self.levels), dtype=np.float32)
        level[self.levels.index(row.level)] = 1.0
        side = np.zeros(2, dtype=np.float32)
        side[self.sides.index(row.side)] = 1.0

        # # Sagittal T2/STIR central ROI
        # sagt2_series_id = self.get_series(row.study_id, 'Sagittal T2/STIR')
        # sagt2_series_id = sagt2_series_id[0] if sagt2_series_id is not None else None
        # sagt2_x_norm, sagt2_y_norm, sagt2_instance_number = self.get_sagt2_coord(
        #     row.study_id, sagt2_series_id, row.level
        # )
        # sagt2_roi = self.get_roi(
        #     study_id=row.study_id,
        #     series_id=sagt2_series_id,
        #     img_num=self.img_num,
        #     resolution=self.resolution,
        #     roi_size=self.roi_size,
        #     x_norm=sagt2_x_norm,
        #     y_norm=sagt2_y_norm,
        #     instance_number_type='middle',
        # )

        # Sagittal T1 central ROI
        sagt1_series_id = self.get_series(row.study_id, 'Sagittal T1')[0]
        sagt1_series_id = sagt1_series_id[0] if sagt1_series_id is not None else None
        sagt1_x_norm, sagt1_y_norm, sagt1_instance_number = self.get_sagt1_coord(row.study_id, sagt1_series_id, row.level, row.side)
        
        sagt1_roi = self.get_roi(
            study_id=row.study_id,
            series_id=sagt1_series_id,
            img_num=self.img_num,
            resolution=self.resolution,
            roi_size=self.roi_size,
            x_norm=sagt1_x_norm,
            y_norm=sagt1_y_norm,
            instance_number_type='middle',
        )
        
        # # Axial T2 central ROI
        # axi_x_norm, axi_y_norm, axi_instance_number, axi_series_id = self.get_axi_coord(row.study_id, row.level, row.side)
            
        # axi_roi = self.get_roi(
        #     study_id=row.study_id,
        #     series_id=axi_series_id,
        #     img_num=self.img_num,
        #     resolution=self.resolution,
        #     roi_size=self.roi_size,
        #     x_norm=axi_x_norm,
        #     y_norm=axi_y_norm,
        #     instance_number_type='filename',
        #     instance_number=axi_instance_number,
        # )

        if self.transform:
            # sagt2_roi = self.transform(image=sagt2_roi)['image']
            sagt1_roi = self.transform(image=sagt1_roi)['image']
            # axi_roi = self.transform(image=axi_roi)['image']
            

        return sagt1_roi, level, label[1]