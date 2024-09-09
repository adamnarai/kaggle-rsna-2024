import os
import random
import warnings
import logging
import pandas as pd
import numpy as np
import pydicom
import cv2
import scipy

import torch
from torch.utils.data import Dataset

from rsna2024.utils import natural_sort


class CoordDataset(Dataset):
    def __init__(
        self,
        df,
        root_dir,
        data_dir,
        img_num,
        resolution,
        heatmap_std,
        phase='train',
        df_coordinates=None,
        cleaning_rule=None,
        transform=None,
    ):
        self.phase = phase
        if self.phase in ['train', 'valid', 'predict', 'faketest', 'valid_check']:
            self.img_subdir = 'train_images'
            self.series_filename = 'train_series_descriptions.csv'
        elif self.phase == 'test':
            self.img_subdir = 'test_images'
            self.series_filename = 'test_series_descriptions.csv'
        if self.phase == 'faketest':
            self.phase = 'test'

        self.df = df
        self.data_dir = os.path.join(root_dir, data_dir)
        self.df_series = self.load_series_info()
        if df_coordinates is None and self.phase != 'test':
            self.df_coordinates = self.load_coordinates_info()
        else:
            self.df_coordinates = df_coordinates
        if self.df_coordinates is not None:
            self.df_coordinates = self.df_coordinates.merge(
                self.df_series, how='left', on=['study_id', 'series_id']
            )
        self.img_dir = os.path.join(self.data_dir, self.img_subdir)
        self.transform = transform
        self.img_num = img_num
        self.resolution = resolution
        self.heatmap_std = heatmap_std
        self.cleaning_rule = cleaning_rule

        self.levels = ['l1_l2', 'l2_l3', 'l3_l4', 'l4_l5', 'l5_s1']
        self.sides = ['left', 'right']

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.ERROR)
        self.handler = logging.FileHandler('level_roi_dataset.log')
        self.logger.addHandler(self.handler)

    def load_series_info(self):
        return pd.read_csv(
            os.path.join(self.data_dir, self.series_filename),
            dtype={'study_id': 'str', 'series_id': 'str'},
        )

    def load_coordinates_info(self):
        return pd.read_csv(
            os.path.join(self.data_dir, '..', 'processed', 'train_label_coordinates.csv'),
            dtype={'study_id': 'str', 'series_id': 'str'},
        )

    def __len__(self):
        return len(self.df)

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

    def create_heatmaps(self, coords):
        heatmaps = []
        for i in range(coords.shape[0]):
            if np.isnan(coords[i, :]).any():
                heatmaps.append(torch.zeros(self.resolution, self.resolution))
            else:
                heatmaps.append(
                    self.gaussian_heatmap(
                        self.resolution, self.resolution, coords[i, :], std_dev=self.heatmap_std
                    )
                )
        return torch.stack(heatmaps, dim=-1).numpy()

    def most_frequent(self, l):
        l = [l_ for l_ in l if not (isinstance(l_, float) and np.isnan(l_))]
        if len(l) == 0:
            return None
        return max(l, key=l.count)

    def get_series_with_coords(self, study_id, series_description, level=None, side=None):
        # Get expected vars in standard order
        if series_description == 'Sagittal T2/STIR':
            expected_vars = [f'spinal_canal_stenosis_{lvl}' for lvl in self.levels]
        elif series_description == 'Sagittal T1':
            expected_vars = [f'{side}_neural_foraminal_narrowing_{lvl}' for lvl in self.levels]
        elif series_description == 'Axial T2':
            expected_vars = [f'{side}_subarticular_stenosis_{level}' for side in self.sides]

        # Get coords in standard order, padded with nans
        series_coords = self.df_coordinates[
            (self.df_coordinates['study_id'] == study_id)
            & (self.df_coordinates['series_description'] == series_description)
        ]
        series_coords = series_coords.merge(
            pd.DataFrame({'row_id': expected_vars}), on='row_id', how='right'
        )
        coords = np.array(
            [series_coords['x_norm'].values, series_coords['y_norm'].values], dtype=np.float32
        ).T

        series_list = series_coords['series_id'].unique().tolist()
        series_id = self.most_frequent(series_list)

        if series_id is None:
            instance_number = np.nan
        else:
            instance_number = self.most_frequent(
                series_coords[series_coords['series_id'] == series_id][
                    'instance_number'
                ].values.tolist()
            )
            if instance_number is None:
                instance_number = np.nan

        if len(series_list) == 0:
            self.logger.warning('%s %s not found', study_id, series_description)
            return None

        if len(series_list) > 1:
            self.logger.warning('%s %s multiple found', study_id, series_description)

        return series_id, coords, instance_number

    def get_series(self, study_id, series_description):
        series_list = self.df_series[
            (self.df_series['study_id'] == study_id)
            & (self.df_series['series_description'] == series_description)
        ]['series_id'].tolist()

        # Try to substitute T1 for T2 or vice versa
        if len(series_list) == 0:
            if series_description == 'Sagittal T2/STIR':
                series_description = 'Sagittal T1'
            elif series_description == 'Sagittal T1':
                series_description = 'Sagittal T2/STIR'
            series_list = self.df_series[
                (self.df_series['study_id'] == study_id)
                & (self.df_series['series_description'] == series_description)
            ]['series_id'].tolist()

        if len(series_list) == 0:
            self.logger.warning('%s %s not found', study_id, series_description)
            return None

        if len(series_list) > 1:
            self.logger.warning('%s %s multiple found', study_id, series_description)

        return series_list

    def get_image(
        self,
        study_id,
        series_id,
        instance_number_type='middle',  # 'middle' 'index', 'filename', 'relative'
        instance_number=None,
        interpolation=cv2.INTER_CUBIC,
        standardize=True,
    ):
        x = np.zeros((self.resolution, self.resolution, self.img_num), dtype=np.float32)
        if series_id is None:
            return x
        if instance_number_type != 'middle' and pd.isnull(instance_number):
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
            start_index = (slice_num - self.img_num) // 2
        elif instance_number_type == 'index':
            start_index = instance_number - self.img_num // 2
        elif instance_number_type == 'filename':
            start_index = [int(file.rstrip('.dcm')) for file in file_list].index(
                instance_number
            ) - self.img_num // 2
        elif instance_number_type == 'relative':
            start_index = int(instance_number * slice_num) - self.img_num // 2
        elif instance_number_type == 'centered_mm':
            try:
                start_index = int(
                    np.ceil(
                        (instance_number / float(ds_first.SpacingBetweenSlices) + slice_num / 2 - 1)
                        - self.img_num / 2
                    )
                )
            except:
                start_index = (slice_num - self.img_num) // 2
        elif instance_number_type == 'centered_mm_old':
            try:
                start_index = (
                    slice_num // 2
                    + round(instance_number / float(ds_first.SpacingBetweenSlices))
                    - self.img_num // 2
                )
            except:
                start_index = slice_num // 2 - self.img_num // 2
        start_index = min(max(start_index, 0), slice_num)
        end_index = min(start_index + self.img_num, slice_num)
        file_list = file_list[start_index:end_index]

        for i, filename in enumerate(file_list):
            ds = pydicom.dcmread(os.path.join(series_dir, filename))
            img = ds.pixel_array.astype(np.float32)

            # Resize
            img = cv2.resize(img, (self.resolution, self.resolution), interpolation=interpolation)
            x[..., i] = img

        # Standardize image
        if standardize and x.std() != 0:
            x = (x - x.mean()) / x.std()

        return x

    def __getitem__(self, idx):
        raise NotImplementedError('Please implement this method in a subclass')


class Sagt2CoordDataset(CoordDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Clean data
        if self.phase in ['train', 'valid', 'valid_check']:
            if self.cleaning_rule == 'keep_only_complete':
                coord_counts = (
                    self.df_coordinates[
                        self.df_coordinates['row_id'].str.contains('spinal_canal_stenosis')
                    ]
                    .groupby('study_id')
                    .count()['series_id']
                )
                good_study_ids = coord_counts[coord_counts == 5].index.astype(str).tolist()
                self.df = self.df[self.df['study_id'].isin(good_study_ids)]

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        if self.phase in ['train', 'valid', 'predict', 'valid_check']:
            series_id, coords, _ = self.get_series_with_coords(row.study_id, 'Sagittal T2/STIR')
            img = self.get_image(row.study_id, series_id, instance_number_type='middle')

            heatmaps = self.create_heatmaps(coords * self.resolution)
            if self.transform:
                t = self.transform(image=img, mask=heatmaps)
                img, heatmaps = t['image'], t['mask']
            heatmaps = np.transpose(heatmaps, (2, 0, 1))

            if self.phase == 'valid_check':
                if series_id is None:
                    series_id = ''
                return img, heatmaps, row.study_id, series_id

            return img, heatmaps

        elif self.phase in ['test']:
            series_id = self.get_series(row.study_id, 'Sagittal T2/STIR')
            if series_id is not None:
                series_id = series_id[0]  # get the first series_id
            img = self.get_image(row.study_id, series_id, instance_number_type='middle')
            if self.transform:
                img = self.transform(image=img)['image']
            if series_id is None:
                series_id = ''
            return img, row.study_id, series_id


class Sagt1CoordDataset(CoordDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Split for sides
        def side_at_end(colname):
            splits = colname.split('_')
            if splits[0] in self.sides:
                return '_'.join(splits[1:] + [splits[0]])
            else:
                return colname

        self.df = self.df.rename(columns=side_at_end)

        self.df = pd.wide_to_long(
            self.df,
            [
                f'{cond}_{level}'
                for cond in ['neural_foraminal_narrowing', 'subarticular_stenosis']
                for level in self.levels
            ],
            i='study_id',
            j='side',
            sep='_',
            suffix=r'\w+',
        ).reset_index()

        # Clean data
        if self.phase in ['train', 'valid', 'valid_check']:
            if self.cleaning_rule == 'keep_only_complete':
                coord_counts = (
                    self.df_coordinates[
                        self.df_coordinates['row_id'].str.contains('neural_foraminal_narrowing')
                    ]
                    .groupby('study_id')
                    .count()['series_id']
                )
                good_study_ids = coord_counts[coord_counts == 10].index.astype(str).tolist()
                self.df = self.df[self.df['study_id'].isin(good_study_ids)]

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Get series_id (and coords)
        if self.phase in ['train', 'valid', 'predict', 'valid_check']:
            series_id, coords, _ = self.get_series_with_coords(
                row.study_id, 'Sagittal T1', side=row.side
            )
        elif self.phase in ['test']:
            series_id = self.get_series(row.study_id, 'Sagittal T1')
            if series_id is not None:
                series_id = series_id[0]  # get the first series_id

        # Get image
        instance_number_type = 'centered_mm'
        instance_number = 16.8
        if row.side == 'right':
            instance_number = -1 * instance_number

        img = self.get_image(
            row.study_id,
            series_id,
            instance_number_type=instance_number_type,
            instance_number=instance_number,
        )

        # Return data (and heatmaps)
        if self.phase in ['train', 'valid', 'predict', 'valid_check']:
            heatmaps = self.create_heatmaps(coords * self.resolution)
            if self.transform:
                t = self.transform(image=img, mask=heatmaps)
                img, heatmaps = t['image'], t['mask']
            heatmaps = np.transpose(heatmaps, (2, 0, 1))

            if self.phase == 'valid_check':
                if series_id is None:
                    series_id = ''
                return img, heatmaps, row.study_id, series_id, row.side

            return img, heatmaps

        elif self.phase in ['test']:
            series_id = self.get_series(row.study_id, 'Sagittal T1')
            if series_id is not None:
                series_id = series_id[0]  # get the first series_id

            if self.transform:
                img = self.transform(image=img)['image']
            if series_id is None:
                series_id = ''
            return img, row.study_id, series_id, row.side


class AxiCoordDataset(CoordDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Need to split for levels
        self.df = pd.wide_to_long(
            self.df,
            [
                'spinal_canal_stenosis',
                'left_neural_foraminal_narrowing',
                'right_neural_foraminal_narrowing',
                'left_subarticular_stenosis',
                'right_subarticular_stenosis',
            ],
            i='study_id',
            j='level',
            sep='_',
            suffix=r'\w+',
        ).reset_index()

        # Clean data
        if self.phase in ['train', 'valid', 'valid_check']:
            if self.cleaning_rule == 'keep_only_complete':
                coord_counts = (
                    self.df_coordinates[
                        self.df_coordinates['row_id'].str.contains('subarticular_stenosis')
                    ]
                    .groupby('study_id')
                    .count()['series_id']
                )
                good_study_ids = coord_counts[coord_counts == 10].index.astype(str).tolist()
                self.df = self.df[self.df['study_id'].isin(good_study_ids)]

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        series_id, coords, instance_number = self.get_series_with_coords(
            row.study_id, 'Axial T2', level=row.level
        )

        img = self.get_image(
            row.study_id,
            series_id,
            instance_number_type='filename',
            instance_number=instance_number,
        )

        if self.phase in ['train', 'valid', 'predict', 'valid_check']:
            heatmaps = self.create_heatmaps(coords * self.resolution)
            if self.transform:
                t = self.transform(image=img, mask=heatmaps)
                img, heatmaps = t['image'], t['mask']
            heatmaps = np.transpose(heatmaps, (2, 0, 1))

            if self.phase == 'valid_check':
                if series_id is None:
                    series_id = ''
                return img, heatmaps, row.study_id, series_id

            return img, heatmaps

        elif self.phase in ['test']:
            if self.transform:
                img = self.transform(image=img)['image']
            if series_id is None:
                series_id = ''
            return img, row.study_id, series_id, row.level


class ROIDataset(Dataset):
    def __init__(
        self,
        df,
        root_dir,
        data_dir,
        img_num,
        resolution,
        roi_size,
        phase='train',
        df_coordinates=None,
        cleaning_rule=None,
        coord_inpute_file=None,
        resample_slice_spacing=None,
        interpolation='INTER_CUBIC',
        standardize=True,
        rand_instance_number_offsets=None,
        transform=None,
    ):
        self.phase = phase
        if self.phase in ['train', 'valid', 'predict', 'faketest', 'valid_check']:
            self.img_subdir = 'train_images'
            self.series_filename = 'train_series_descriptions.csv'
        elif self.phase == 'test':
            self.img_subdir = 'test_images'
            self.series_filename = 'test_series_descriptions.csv'
        if self.phase == 'faketest':
            self.phase = 'test'

        self.df = df
        self.data_dir = os.path.join(root_dir, data_dir)
        self.df_series = self.load_series_info()
        self.coord_inpute_file = coord_inpute_file
        if df_coordinates is None and self.phase != 'test':
            self.df_coordinates = self.load_coordinates_info()
        else:
            self.df_coordinates = df_coordinates
        if self.df_coordinates is not None:
            self.df_coordinates = self.df_coordinates.merge(
                self.df_series, how='left', on=['study_id', 'series_id']
            )
        self.img_dir = os.path.join(self.data_dir, self.img_subdir)

        self.df_orig = df
        self.df = pd.wide_to_long(
            df,
            [
                'spinal_canal_stenosis',
                'left_neural_foraminal_narrowing',
                'right_neural_foraminal_narrowing',
                'left_subarticular_stenosis',
                'right_subarticular_stenosis',
            ],
            i='study_id',
            j='level',
            sep='_',
            suffix=r'\w+',
        ).reset_index()
        self.transform = transform
        self.img_num = img_num
        self.resolution = resolution
        self.roi_size = roi_size
        self.cleaning_rule = cleaning_rule
        self.resample_slice_spacing = resample_slice_spacing
        self.interpolation = interpolation
        self.standardize = standardize
        self.rand_instance_number_offsets = rand_instance_number_offsets

        self.levels = ['l1_l2', 'l2_l3', 'l3_l4', 'l4_l5', 'l5_s1']
        self.sides = ['left', 'right']

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.ERROR)
        self.handler = logging.FileHandler('level_roi_dataset.log')
        self.logger.addHandler(self.handler)

    def load_series_info(self):
        return pd.read_csv(
            os.path.join(self.data_dir, self.series_filename),
            dtype={'study_id': 'str', 'series_id': 'str'},
        )

    def load_coordinates_info(self):
        df_coordinates = pd.read_csv(
            os.path.join(self.data_dir, '..', 'processed', 'train_label_coordinates.csv'),
            dtype={'study_id': 'str', 'series_id': 'str'},
        )
        if self.coord_inpute_file is not None:
            df_coordinates_pred = pd.read_csv(
                os.path.join(self.data_dir, '..', 'processed', self.coord_inpute_file + '.csv'),
                dtype={'study_id': 'str', 'series_id': 'str'},
            )
            bad_coordinates = pd.read_csv(
                os.path.join(
                    self.data_dir, '..', 'processed', 'bad_coords_spinal_canal_stenosis.csv'
                ),
                dtype={'study_id': 'str', 'series_id': 'str'},
            )

            # Remove bad coords
            df_coordinates.set_index(['study_id', 'series_id', 'row_id'], inplace=True)
            bad_coordinates.set_index(['study_id', 'series_id', 'row_id'], inplace=True)
            df_coordinates = df_coordinates.drop(
                df_coordinates.index.intersection(bad_coordinates.index)
            ).reset_index()

            # Inpute with predicted coords
            df_coordinates.set_index(['study_id', 'row_id'], inplace=True)
            df_coordinates_pred.set_index(['study_id', 'row_id'], inplace=True)
            df_coordinates = pd.concat(
                [
                    df_coordinates,
                    df_coordinates_pred.drop(df_coordinates.index).dropna(subset='series_id'),
                ],
                axis=0,
            ).reset_index()

        return df_coordinates

    def __len__(self):
        return len(self.df)

    def get_series(self, study_id, series_description):
        series_list = self.df_series[
            (self.df_series['study_id'] == study_id)
            & (self.df_series['series_description'] == series_description)
        ]['series_id'].tolist()

        # Try to substitute T1 for T2 or vice versa
        if len(series_list) == 0:
            if series_description == 'Sagittal T2/STIR':
                series_description = 'Sagittal T1'
            elif series_description == 'Sagittal T1':
                series_description = 'Sagittal T2/STIR'
            series_list = self.df_series[
                (self.df_series['study_id'] == study_id)
                & (self.df_series['series_description'] == series_description)
            ]['series_id'].tolist()

        if len(series_list) == 0:
            self.logger.warning('%s %s not found', study_id, series_description)
            return None

        if len(series_list) > 1:
            self.logger.warning('%s %s multiple found', study_id, series_description)

        return series_list

    def get_valid_fold_idx(self, model_dir, study_id):
        splits = pd.read_csv(os.path.join(model_dir, 'splits.csv'), dtype={'study_id': 'str'})
        fold_idx = (
            splits[(splits['study_id'] == study_id) & (splits['split'] == 'validation')][
                'fold'
            ].values[0]
            - 1
        )
        return fold_idx

    def get_sagt2_coord(self, study_id, level):
        coords = self.df_coordinates[
            (self.df_coordinates['study_id'] == study_id)
            & (self.df_coordinates['row_id'] == f'spinal_canal_stenosis_{level}')
        ][['x_norm', 'y_norm', 'instance_number', 'series_id']].values

        if coords.shape[0] == 0:
            self.logger.warning('SAGT2 %s %s no coordinates found', study_id, level)
            return (np.nan, np.nan, np.nan, None)

        if coords.shape[0] > 1:
            self.logger.warning('SAGT2 %s %s multiple coordinates found', study_id, level)

        x_norm, y_norm, instance_number, series_id = coords[0]
        if not pd.isnull(instance_number):
            instance_number = int(instance_number)
        return x_norm, y_norm, instance_number, series_id

    def get_sagt1_coord(self, study_id, level, side):
        coords = self.df_coordinates[
            (self.df_coordinates['study_id'] == study_id)
            & (self.df_coordinates['row_id'] == f'{side}_neural_foraminal_narrowing_{level}')
        ][['x_norm', 'y_norm', 'instance_number', 'series_id']].values

        if coords.shape[0] == 0:
            self.logger.warning('SAGT1 %s %s %s no coordinates found', study_id, level, side)
            return np.nan, np.nan, np.nan, None

        if coords.shape[0] > 1:
            self.logger.warning('SAGT1 %s %s %s multiple coordinates found', study_id, level, side)

        x_norm, y_norm, instance_number, series_id = coords[0]
        if not pd.isnull(instance_number):
            instance_number = int(instance_number)
        return x_norm, y_norm, instance_number, series_id

    def get_axi_coord(self, study_id, level, side):
        coords = self.df_coordinates[
            (self.df_coordinates['study_id'] == study_id)
            & (self.df_coordinates['row_id'] == f'{side}_subarticular_stenosis_{level}')
        ][['x_norm', 'y_norm', 'instance_number', 'series_id']].values

        if coords.shape[0] == 0:
            self.logger.warning('AXI %s %s %s no coordinates found', study_id, level, side)
            return np.nan, np.nan, np.nan, None

        if coords.shape[0] > 1:
            self.logger.warning('AXI %s %s %s multiple coordinates found', study_id, level, side)

        x_norm, y_norm, instance_number, series_id = coords[0]
        if not pd.isnull(instance_number):
            instance_number = int(instance_number)
        return x_norm, y_norm, instance_number, series_id

    def get_label(self, idx, label_name):
        label = self.df[label_name].iloc[idx]
        if isinstance(label_name, list):
            label = label.astype(float).values
        else:
            label = float(label)
        label = np.nan_to_num(label, nan=-100).astype(np.int64)
        return label

    def get_level_onehot(self, level_name):
        level = np.zeros(len(self.levels), dtype=np.float32)
        level[self.levels.index(level_name)] = 1.0
        return level

    def get_side_onehot(self, side_name):
        side = np.zeros(len(self.sides), dtype=np.float32)
        side[self.sides.index(side_name)] = 1.0
        return side

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
        resample_slice_spacing=None,
    ):
        x = np.zeros((resolution, resolution, img_num), dtype=np.float32)
        if series_id is None or np.isnan(x_norm) or np.isnan(y_norm):
            return x
        if instance_number_type != 'middle' and pd.isnull(instance_number):
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

        if resample_slice_spacing is not None:
            try:
                slice_spacing = float(ds_first.SpacingBetweenSlices)
            except Exception:
                slice_spacing = 4.5
            resample_factor = resample_slice_spacing / slice_spacing
            img_num_final = img_num
            img_num = int(img_num * resample_factor)
            resample_factor = img_num / img_num_final
            x = np.zeros((resolution, resolution, img_num), dtype=np.float32)

        if instance_number_type == 'middle':
            start_index = (slice_num - img_num) // 2
        elif instance_number_type == 'index':
            start_index = instance_number - img_num // 2
        elif instance_number_type == 'filename':
            start_index = [int(file.rstrip('.dcm')) for file in file_list].index(
                instance_number
            ) - img_num // 2
        elif instance_number_type == 'relative':
            start_index = int(instance_number * slice_num) - img_num // 2
        elif instance_number_type == 'centered_index':
            start_index = slice_num // 2 + instance_number - img_num // 2
        elif instance_number_type == 'centered_mm':
            try:
                start_index = int(
                    np.ceil(
                        (instance_number / float(ds_first.SpacingBetweenSlices) + slice_num / 2 - 1)
                        - img_num / 2
                    )
                )
            except Exception:
                start_index = (slice_num - img_num) // 2
        elif instance_number_type == 'centered_mm_old':
            try:
                start_index = (
                    slice_num // 2
                    + round(instance_number / float(ds_first.SpacingBetweenSlices))
                    - img_num // 2
                )
            except:
                start_index = slice_num // 2 - img_num // 2

        # Augment instance number during training
        if self.rand_instance_number_offsets is not None and self.phase in ['train']:
            start_index += random.sample(self.rand_instance_number_offsets, 1)[0]

        start_index = min(max(start_index, 0), slice_num)
        end_index = min(start_index + img_num, slice_num)
        file_list = file_list[start_index:end_index]

        interpolation = getattr(cv2, self.interpolation)
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
                self.logger.warning('%s %s ROI out of bounds', study_id, series_id)
                break
            img = img[y1:y2, x1:x2]

            # Resize
            img = cv2.resize(img, (resolution, resolution), interpolation=interpolation)
            x[..., i] = img

        if resample_slice_spacing is not None:
            x = scipy.ndimage.zoom(x, (1, 1, 1 / resample_factor), order=3)

        # Standardize image
        if self.standardize and x.std() != 0:
            x = (x - x.mean()) / x.std()

        return x

    def __getitem__(self, idx):
        raise NotImplementedError('Please implement this method in a subclass')


class SpinalROIDataset(ROIDataset):
    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Sagittal T2/STIR ROI
        sagt2_x_norm, sagt2_y_norm, _, sagt2_series_id = self.get_sagt2_coord(
            row.study_id, row.level
        )
        sagt2_roi = self.get_roi(
            study_id=row.study_id,
            series_id=sagt2_series_id,
            img_num=self.img_num[0],
            resolution=self.resolution,
            roi_size=self.roi_size,
            x_norm=sagt2_x_norm,
            y_norm=sagt2_y_norm,
            instance_number_type='middle',
            resample_slice_spacing=self.resample_slice_spacing,
        )

        # Axial T2 central ROI
        left_axi_x_norm, left_axi_y_norm, left_axi_instance_number, left_axi_series_id = (
            self.get_axi_coord(row.study_id, row.level, 'left')
        )
        right_axi_x_norm, right_axi_y_norm, right_axi_instance_number, right_axi_series_id = (
            self.get_axi_coord(row.study_id, row.level, 'right')
        )

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            axi_x_norm = np.nanmean([left_axi_x_norm, right_axi_x_norm])
            axi_y_norm = np.nanmean([left_axi_y_norm, right_axi_y_norm])
        if not pd.isnull(left_axi_instance_number):
            axi_instance_number = left_axi_instance_number
            axi_series_id = left_axi_series_id
        else:
            axi_instance_number = right_axi_instance_number
            axi_series_id = right_axi_series_id

        axi_roi = self.get_roi(
            study_id=row.study_id,
            series_id=axi_series_id,
            img_num=self.img_num[1],
            resolution=self.resolution,
            roi_size=self.roi_size,
            x_norm=axi_x_norm,
            y_norm=axi_y_norm,
            instance_number_type='filename',
            instance_number=axi_instance_number,
            resample_slice_spacing=self.resample_slice_spacing,
        )

        if self.transform:
            if isinstance(self.transform, list):
                print('multiple transforms used')
                sagt2_roi = self.transform[0](image=sagt2_roi)['image']
                axi_roi = self.transform[1](image=axi_roi)['image']
            else:
                sagt2_roi = self.transform(image=sagt2_roi)['image']
                axi_roi = self.transform(image=axi_roi)['image']

        level = self.get_level_onehot(row.level)
        if self.phase in ['train', 'valid', 'predict', 'valid_check']:
            label = self.get_label(idx, 'spinal_canal_stenosis')
            if self.phase == 'valid_check':
                sagt2_series_id = '' if sagt2_series_id is None else sagt2_series_id
                axi_series_id = '' if axi_series_id is None else axi_series_id
                return (
                    sagt2_roi,
                    axi_roi,
                    level,
                    label,
                    row.study_id,
                    sagt2_series_id,
                    axi_series_id,
                    row.level,
                )
            return sagt2_roi, axi_roi, level, label
        elif self.phase in ['test']:
            return sagt2_roi, axi_roi, level, 0


class ForaminalROIDataset(ROIDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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

        # Sagittal T1 ROI
        sagt1_x_norm, sagt1_y_norm, _, sagt1_series_id = self.get_sagt1_coord(
            row.study_id, row.level, row.side
        )

        instance_number_type = 'centered_mm'
        instance_number = 16.8
        if row.side == 'right':
            instance_number = -1 * instance_number

        sagt1_roi = self.get_roi(
            study_id=row.study_id,
            series_id=sagt1_series_id,
            img_num=self.img_num,
            resolution=self.resolution,
            roi_size=self.roi_size,
            x_norm=sagt1_x_norm,
            y_norm=sagt1_y_norm,
            instance_number_type=instance_number_type,
            instance_number=instance_number,
            resample_slice_spacing=self.resample_slice_spacing,
        )

        if self.transform:
            sagt1_roi = self.transform(image=sagt1_roi)['image']

        level = self.get_level_onehot(row.level)
        side = self.get_side_onehot(row.side)
        if self.phase in ['train', 'valid', 'predict', 'valid_check']:
            label = self.get_label(idx, 'neural_foraminal_narrowing')
            if self.phase == 'valid_check':
                sagt1_series_id = '' if sagt1_series_id is None else sagt1_series_id
                return sagt1_roi, level, side, label, row.study_id, sagt1_series_id, row.level
            return sagt1_roi, level, side, label
        elif self.phase in ['test']:
            return sagt1_roi, level, side, 0


class SubarticularROIDataset(ForaminalROIDataset):
    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Axial T2 central ROI
        axi_x_norm, axi_y_norm, axi_instance_number, axi_series_id = self.get_axi_coord(
            row.study_id, row.level, row.side
        )

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
            resample_slice_spacing=self.resample_slice_spacing,
        )

        # Flip if right side
        if row.side == 'right':
            axi_roi = np.flip(axi_roi, axis=1).copy()

        if self.transform:
            axi_roi = self.transform(image=axi_roi)['image']

        level = self.get_level_onehot(row.level)
        side = self.get_side_onehot(row.side)
        if self.phase in ['train', 'valid', 'predict', 'valid_check']:
            label = self.get_label(idx, 'subarticular_stenosis')
            if self.phase == 'valid_check':
                axi_series_id = '' if axi_series_id is None else axi_series_id
                return axi_roi, level, side, label, row.study_id, axi_series_id, row.level, row.side
            return axi_roi, level, side, label
        elif self.phase == 'test':
            return axi_roi, level, side, 0


class GlobalROIDataset(ROIDataset):
    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Sagittal T2/STIR ROI
        sagt2_x_norm, sagt2_y_norm, _, sagt2_series_id = self.get_sagt2_coord(
            row.study_id, row.level
        )
        sagt2_roi = self.get_roi(
            study_id=row.study_id,
            series_id=sagt2_series_id,
            img_num=self.img_num[0],
            resolution=self.resolution,
            roi_size=self.roi_size,
            x_norm=sagt2_x_norm,
            y_norm=sagt2_y_norm,
            instance_number_type='middle',
            resample_slice_spacing=self.resample_slice_spacing,
        )

        for side in ['left', 'right']:
            # Sagittal T1 ROI
            sagt1_x_norm, sagt1_y_norm, _, sagt1_series_id = self.get_sagt1_coord(
                row.study_id, row.level, side
            )

            sagt1_instance_number = 16.8
            if side == 'right':
                sagt1_instance_number = -1 * sagt1_instance_number

            sagt1_roi = self.get_roi(
                study_id=row.study_id,
                series_id=sagt1_series_id,
                img_num=self.img_num[1],
                resolution=self.resolution,
                roi_size=self.roi_size,
                x_norm=sagt1_x_norm,
                y_norm=sagt1_y_norm,
                instance_number_type='centered_mm',
                instance_number=sagt1_instance_number,
                resample_slice_spacing=self.resample_slice_spacing,
            )
            if side == 'left':
                sagt1_left_roi = sagt1_roi
            elif side == 'right':
                sagt1_right_roi = sagt1_roi

            # Axial T2 central ROI
            axi_x_norm, axi_y_norm, axi_instance_number, axi_series_id = self.get_axi_coord(
                row.study_id, row.level, side
            )

            axi_roi = self.get_roi(
                study_id=row.study_id,
                series_id=axi_series_id,
                img_num=self.img_num[2],
                resolution=self.resolution,
                roi_size=self.roi_size,
                x_norm=axi_x_norm,
                y_norm=axi_y_norm,
                instance_number_type='filename',
                instance_number=axi_instance_number,
                resample_slice_spacing=self.resample_slice_spacing,
            )

            # Flip if right side
            if side == 'left':
                axi_left_roi = axi_roi
            elif side == 'right':
                axi_right_roi = np.flip(axi_roi, axis=1).copy()

        if self.transform:
            sagt2_roi = self.transform[0](image=sagt2_roi)['image']
            sagt1_left_roi = self.transform[1](image=sagt1_left_roi)['image']
            sagt1_right_roi = self.transform[1](image=sagt1_right_roi)['image']
            axi_left_roi = self.transform[2](image=axi_left_roi)['image']
            axi_right_roi = self.transform[2](image=axi_right_roi)['image']

        level = self.get_level_onehot(row.level)
        if self.phase in ['train', 'valid', 'predict', 'valid_check']:
            label = self.get_label(
                idx,
                [
                    'spinal_canal_stenosis',
                    'left_neural_foraminal_narrowing',
                    'right_neural_foraminal_narrowing',
                    'left_subarticular_stenosis',
                    'right_subarticular_stenosis',
                ],
            )
            return (
                sagt2_roi,
                sagt1_left_roi,
                sagt1_right_roi,
                axi_left_roi,
                axi_right_roi,
                level,
                label,
            )
        elif self.phase in ['test']:
            return sagt2_roi, sagt1_left_roi, sagt1_right_roi, axi_left_roi, axi_right_roi, level, 0


class SpinalROIDatasetV2(SpinalROIDataset):
    # TODO: future remove
    pass


class ForaminalROIDatasetV2(ForaminalROIDataset):
    # TODO: future remove
    pass
