import os
import random
from itertools import compress
import pandas as pd
import numpy as np
import pydicom
import cv2
import warnings
import logging
from albumentations.pytorch import ToTensorV2

import torch
from torch.utils.data import Dataset

from rsna2024.utils import natural_sort, load_config, sagi_coord_to_axi_instance_number
from rsna2024 import model as module_model
import rsna2024.data_loader as module_data


class Sagt2CoordDataset(Dataset):
    def __init__(
        self,
        df,
        root_dir,
        data_dir,
        img_num,
        resolution,
        heatmap_std,
        phase='train',
        transform=None,
    ):
        if phase == 'test':
            self.img_subdir = 'test_images'
            self.series_filename = 'test_series_descriptions.csv'
        else:
            self.img_subdir = 'train_images'
            self.series_filename = 'train_series_descriptions.csv'

        self.levels = ['l1_l2', 'l2_l3', 'l3_l4', 'l4_l5', 'l5_s1']
        self.sides = ['left', 'right']

        self.df = df
        self.data_dir = os.path.join(root_dir, data_dir)
        self.df_series = self.load_series_info()
        self.df_coordinates = self.load_coordinates_info(root_dir).merge(
            self.df_series, how='left', on=['study_id', 'series_id']
        )
        self.img_dir = os.path.join(self.data_dir, 'train_images')
        self.transform = transform
        self.img_num = img_num
        self.resolution = resolution
        self.heatmap_std = heatmap_std

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.ERROR)
        self.handler = logging.FileHandler('level_roi_dataset.log')
        self.logger.addHandler(self.handler)

    def load_series_info(self):
        return pd.read_csv(
            os.path.join(self.data_dir, self.series_filename),
            dtype={'study_id': 'str', 'series_id': 'str'},
        )

    def load_coordinates_info(self, root_dir):
        return pd.read_csv(
            os.path.join(root_dir, 'data', 'processed', 'train_label_coordinates.csv'),
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
        for i in range(5):
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
        res = max(set(l), key=l.count)
        if isinstance(res, float) and np.isnan(res):
            return None
        return res

    def get_series_with_coords(self, study_id, series_description, level=None):
        # Get expected vars in standard order
        if series_description == 'Sagittal T2/STIR':
            expected_vars = [f'spinal_canal_stenosis_{lvl}' for lvl in self.levels]
        elif series_description == 'Sagittal T1':
            expected_vars = [
                f'{side}_neural_foraminal_narrowing_{lvl}'
                for lvl in self.levels
                for side in self.sides
            ]
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

        if len(series_list) == 0:
            self.logger.warning('%s %s not found', study_id, series_description)
            return None

        if len(series_list) > 1:
            self.logger.warning('%s %s multiple found', study_id, series_description)

        instance_number = self.most_frequent(series_coords['instance_number'].values)

        return series_id, coords, instance_number

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
            start_index = max(instance_number - self.img_num // 2, 0)
        elif instance_number_type == 'filename':
            start_index = max(
                [int(file.rstrip('.dcm')) for file in file_list].index(instance_number)
                - self.img_num // 2,
                0,
            )
        elif instance_number_type == 'relative':
            start_index = max(int(instance_number * slice_num) - self.img_num // 2, 0)
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
        row = self.df.iloc[idx]

        series_id, coords, instance_number = self.get_series_with_coords(
            row.study_id, 'Sagittal T2/STIR'
        )
        heatmaps = self.create_heatmaps(coords * self.resolution)
        img = self.get_image(row.study_id, series_id, instance_number_type='middle')

        if self.transform:
            t = self.transform(image=img, mask=heatmaps)
            img, heatmaps = t['image'], t['mask']
        heatmaps = np.transpose(heatmaps, (2, 0, 1))

        return img, heatmaps


class Sagt1CoordDataset(Sagt2CoordDataset):
    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        series_id, coords, instance_number = self.get_series_with_coords(
            row.study_id, 'Sagittal T1'
        )
        # Average left/right coordinates
        coords = np.nanmean(coords.reshape(2, -1, 2, order='F'), axis=0)
        heatmaps = self.create_heatmaps(coords * self.resolution)
        img_left = self.get_image(
            row.study_id, series_id, instance_number_type='relative', instance_number=0.70
        )
        img_right = self.get_image(
            row.study_id, series_id, instance_number_type='relative', instance_number=0.26
        )
        img = np.concatenate([img_left, img_right], axis=-1)

        if self.transform:
            t = self.transform(image=img, mask=heatmaps)
            img, heatmaps = t['image'], t['mask']
        heatmaps = np.transpose(heatmaps, (2, 0, 1))

        return img, heatmaps


class AxiCoordDataset(Sagt2CoordDataset):
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

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        series_id, coords, instance_number = self.get_series_with_coords(
            row.study_id, 'Axial T2', level=row.level
        )
        # Average left/right coordinates
        coords = np.nanmean(coords.reshape(2, -1, 2, order='F'), axis=0)
        heatmaps = self.create_heatmaps(coords * self.resolution)
        img = self.get_image(
            row.study_id,
            series_id,
            instance_number_type='filename',
            instance_number=instance_number,
        )

        if self.transform:
            t = self.transform(image=img, mask=heatmaps)
            img, heatmaps = t['image'], t['mask']
        heatmaps = np.transpose(heatmaps, (2, 0, 1))

        return img, heatmaps


class SpinalROIDataset(Dataset):
    def __init__(
        self,
        df,
        root_dir,
        data_dir,
        img_num,
        resolution,
        roi_size,
        phase='train',
        coord_model_names={},
        transform=None,
    ):
        if phase == 'train':
            self.img_subdir = 'train_images'
            self.series_filename = 'train_series_descriptions.csv'
        elif phase == 'test':
            self.img_subdir = 'test_images'
            self.series_filename = 'test_series_descriptions.csv'

        self.levels = ['l1_l2', 'l2_l3', 'l3_l4', 'l4_l5', 'l5_s1']
        self.sides = ['left', 'right']
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
        self.data_dir = os.path.join(root_dir, data_dir)
        self.df_series = self.load_series_info()
        self.df_coordinates = self.load_coordinates_info(root_dir).merge(
            self.df_series, how='left', on=['study_id', 'series_id']
        )
        self.img_dir = os.path.join(self.data_dir, self.img_subdir)
        self.transform = transform
        self.img_num = img_num
        self.resolution = resolution
        self.roi_size = roi_size
        self.phase = phase

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.ERROR)
        self.handler = logging.FileHandler('level_roi_dataset.log')
        self.logger.addHandler(self.handler)
        
        if phase != 'train':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.load_coord_models(root_dir, coord_model_names, phase)

    def load_coord_models(self, root_dir, coord_model_names, phase):
        self.coord_model_dirs = {}
        self.coord_models = {}
        self.coord_datasets = {}
        for k, v in coord_model_names.items():
            self.coord_model_dirs[k] = os.path.join(root_dir, 'models', 'rsna-2024-' + v)
            cfg = load_config(os.path.join(self.coord_model_dirs[k], 'config.json'))
            model_path_list = sorted(
                [
                    os.path.join(self.coord_model_dirs[k], x)
                    for x in os.listdir(self.coord_model_dirs[k])
                    if x.endswith('_best.pt')
                ]
            )

            self.coord_models[k] = []
            for path in model_path_list:
                model = getattr(module_model, cfg['model']['type'])(**cfg['model']['args'])
                model.load_state_dict(torch.load(path))
                model.to(self.device)
                model.eval()
                self.coord_models[k].append(model)

            self.coord_datasets[k] = getattr(module_data, cfg['dataset']['type'])(
                df=self.df_orig, root_dir=root_dir, **cfg['model']['args']
            )
                
    def load_series_info(self):
        return pd.read_csv(
            os.path.join(self.data_dir, self.series_filename),
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
            self.logger.warning('%s %s not found', study_id, series_description)
            return None

        if len(series_list) > 1:
            self.logger.warning('%s %s multiple found', study_id, series_description)

        return series_list
    
    def get_valid_fold_idx(self, model_dir, study_id):
        splits = pd.read_csv(os.path.join(model_dir, 'splits.csv'), dtype={'study_id': 'str'})
        fold_idx = splits[(splits['study_id'] == study_id) & (splits['split'] == 'validation')]['fold'].values[0] - 1
        return fold_idx

    def get_sagt2_coord(self, study_id, series_id, level):
        if self.phase == 'train':
            level = level.replace('_', '/').upper()
            coords = self.df_coordinates[
                (self.df_coordinates['study_id'] == study_id)
                & (self.df_coordinates['series_id'] == series_id)
                & (self.df_coordinates['level'] == level)
            ][['x_norm', 'y_norm', 'instance_number']].values

            if coords.shape[0] == 0:
                self.logger.warning(
                    'SAGT2 %s %s %s no coordinates found', study_id, series_id, level
                )
                return (np.nan, np.nan, np.nan)

            if coords.shape[0] > 1:
                self.logger.warning(
                    'SAGT2 %s %s %s multiple coordinates found', study_id, series_id, level
                )

            x_norm, y_norm, instance_number = coords[0]
            return x_norm, y_norm, int(instance_number)
        else:
            transform = ToTensorV2()

            # Get data
            img = self.coord_datasets['sagt2'].get_image(
                study_id, series_id, instance_number_type='middle'
            )
            img = transform(image=img)['image']
            img = img.unsqueeze(0)

            # Run keypoint detection model
            if self.phase == 'valid':
                # Load single model for OOF coordinate prediction
                fold_idx = self.get_valid_fold_idx(self.coord_model_dirs['sagt2'], study_id)
                model_list = [self.coord_models['sagt2'][fold_idx]]
            else:
                model_list = self.coord_models['sagt2']
            preds = []
            with torch.no_grad():
                for model in model_list:
                    preds.append(model(img.to(self.device)))
            pred = torch.stack(preds, dim=0).mean(dim=0)

            # Get coordinates
            level_idx = self.levels.index(level)
            pred = pred[level_idx]
            y_coord, x_coord = np.unravel_index(pred.argmax(), pred.shape)
            x_norm = x_coord / pred.shape[1]
            y_norm = y_coord / pred.shape[0]

            return x_norm, y_norm, np.nan

    def get_sagt1_coord(self, study_id, series_id, level, side):
        if self.phase == 'train':
            df_coordinates_filtered = self.df_coordinates[
                (self.df_coordinates['study_id'] == study_id)
                & (self.df_coordinates['series_id'] == series_id)
                & (self.df_coordinates['level'] == level.replace('_', '/').upper())
            ]
            coords = df_coordinates_filtered[
                df_coordinates_filtered['row_id'].str.startswith(side)
            ][['x_norm', 'y_norm', 'instance_number']].values

            if coords.shape[0] == 0:
                self.logger.warning(
                    'SAGT1 %s %s %s %s no coordinates found', study_id, series_id, level, side
                )
                return np.nan, np.nan, np.nan

            if coords.shape[0] > 1:
                self.logger.warning(
                    'SAGT1 %s %s %s %s multiple coordinates found', study_id, series_id, level, side
                )

            x_norm, y_norm, instance_number = coords[0]
            return x_norm, y_norm, int(instance_number)
        else:
            transform = ToTensorV2()

            # Get data
            img_left = self.coord_datasets['sagt1'].get_image(
                study_id, series_id, instance_number_type='relative', instance_number=0.70
            )
            img_right = self.coord_datasets['sagt1'].get_image(
                study_id, series_id, instance_number_type='relative', instance_number=0.26
            )
            img = np.concatenate([img_left, img_right], axis=-1)
            img = transform(image=img)['image']
            img = img.unsqueeze(0)

            # Run keypoint detection model
            if self.phase == 'valid':
                # Load single model for OOF coordinate prediction
                fold_idx = self.get_valid_fold_idx(self.coord_model_dirs['sagt1'], study_id)
                model_list = [self.coord_models['sagt1'][fold_idx]]
            else:
                model_list = self.coord_models['sagt1']
            preds = []
            with torch.no_grad():
                for model in model_list:
                    preds.append(model(img.to(self.device)))
            pred = torch.stack(preds, dim=0).mean(dim=0)

            # Get coordinates
            level_idx = self.levels.index(level)
            pred = pred[level_idx]
            y_coord, x_coord = np.unravel_index(pred.argmax(), pred.shape)
            x_norm = x_coord / pred.shape[1]
            y_norm = y_coord / pred.shape[0]

            return x_norm, y_norm, np.nan

    def get_axi_coord(self, study_id, level, side):
        if self.phase == 'train':
            df_coordinates_filtered = self.df_coordinates[
                (self.df_coordinates['study_id'] == study_id)
                & (self.df_coordinates['series_description'] == 'Axial T2')
                & (self.df_coordinates['level'] == level.replace('_', '/').upper())
            ]
            coords = df_coordinates_filtered[
                df_coordinates_filtered['row_id'].str.startswith(side)
            ][['x_norm', 'y_norm', 'instance_number', 'series_id']].values

            if coords.shape[0] == 0:
                self.logger.warning('AXI %s %s %s no coordinates found', study_id, level, side)
                return np.nan, np.nan, np.nan, ''

            if coords.shape[0] > 1:
                self.logger.warning(
                    'AXI %s %s %s multiple coordinates found', study_id, level, side
                )

            x_norm, y_norm, instance_number, series_id = coords[0]
            return x_norm, y_norm, int(instance_number), str(series_id)
        else:
            sagt2_series_id = self.get_series(study_id, 'Sagittal T2/STIR')
            sagt2_series_id = sagt2_series_id[0] if sagt2_series_id is not None else None
            sag_x_norm, sag_y_norm, sag_instance_number = self.get_sagt2_coord(study_id, sagt2_series_id, level)
            sag_dir = os.path.join(self.img_dir, str(study_id), str(sagt2_series_id))
            axi_dir_list = self.get_series(study_id, 'Axial T2')
            axi_slice_idx = sagi_coord_to_axi_instance_number(sag_x_norm, sag_y_norm, sag_dir, axi_dir_list)
            
            transform = ToTensorV2()

            # Get data
            img = self.coord_datasets['axi'].get_image(
                study_id, series_id, instance_number_type='index', instance_number=axi_slice_idx
            )
            img = transform(image=img)['image']
            img = img.unsqueeze(0)

            # Run keypoint detection model
            if self.phase == 'valid':
                # Load single model for OOF coordinate prediction
                fold_idx = self.get_valid_fold_idx(self.coord_model_dirs['axi'], study_id)
                model_list = [self.coord_models['axi'][fold_idx]]
            else:
                model_list = self.coord_models['axi']
            preds = []
            with torch.no_grad():
                for model in model_list:
                    preds.append(model(img.to(self.device)))
            pred = torch.stack(preds, dim=0).mean(dim=0)

            # Get coordinates
            side_idx = self.sides.index(side)
            pred = pred[side_idx]
            y_coord, x_coord = np.unravel_index(pred.argmax(), pred.shape)
            x_norm = x_coord / pred.shape[1]
            y_norm = y_coord / pred.shape[0]

            return x_norm, y_norm, axi_slice_idx

    def get_labels(self, idx, label_name, level_name, side_name=None):
        label = self.df[label_name].iloc[idx]
        label = np.nan_to_num(float(label), nan=0).astype(np.int64)

        level = np.zeros(len(self.levels), dtype=np.float32)
        level[self.levels.index(level_name)] = 1.0

        if side_name is None:
            return label, level
        side = np.zeros(2, dtype=np.float32)
        side[self.sides.index(side_name)] = 1.0

        return label, level, side

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
                self.logger.warning('%s %s ROI out of bounds', study_id, series_id)
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
        label, level = self.get_labels(idx, 'spinal_canal_stenosis', row.level)

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

        if self.transform:
            sagt2_roi = self.transform(image=sagt2_roi)['image']

        return sagt2_roi, level, label


class ForaminalROIDataset(SpinalROIDataset):
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
        label, level, side = self.get_labels(idx, 'neural_foraminal_narrowing', row.level, row.side)

        # Sagittal T1 central ROI
        sagt1_series_id = self.get_series(row.study_id, 'Sagittal T1')
        sagt1_series_id = sagt1_series_id[0] if sagt1_series_id is not None else None
        sagt1_x_norm, sagt1_y_norm, sagt1_instance_number = self.get_sagt1_coord(
            row.study_id, sagt1_series_id, row.level, row.side
        )
        if row.side == 'left':
            instance_number = 0.70
        elif row.side == 'right':
            instance_number = 0.26

        sagt1_roi = self.get_roi(
            study_id=row.study_id,
            series_id=sagt1_series_id,
            img_num=self.img_num,
            resolution=self.resolution,
            roi_size=self.roi_size,
            x_norm=sagt1_x_norm,
            y_norm=sagt1_y_norm,
            instance_number_type='relative',
            instance_number=instance_number,
        )

        if self.transform:
            sagt1_roi = self.transform(image=sagt1_roi)['image']

        return sagt1_roi, level, label


class SubarticularROIDataset(ForaminalROIDataset):
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        label, level, side = self.get_labels(idx, 'subarticular_stenosis', row.level, row.side)

        # Sagittal T1 central ROI
        sagt1_series_id = self.get_series(row.study_id, 'Sagittal T1')
        sagt1_series_id = sagt1_series_id[0] if sagt1_series_id is not None else None
        sagt1_x_norm, sagt1_y_norm, sagt1_instance_number = self.get_sagt1_coord(
            row.study_id, sagt1_series_id, row.level, row.side
        )
        if row.side == 'left':
            instance_number = 0.70
        elif row.side == 'right':
            instance_number = 0.26

        sagt1_roi = self.get_roi(
            study_id=row.study_id,
            series_id=sagt1_series_id,
            img_num=self.img_num,
            resolution=self.resolution,
            roi_size=self.roi_size,
            x_norm=sagt1_x_norm,
            y_norm=sagt1_y_norm,
            instance_number_type='relative',
            instance_number=instance_number,
        )

        # Axial T2 central ROI
        axi_x_norm, axi_y_norm, axi_instance_number, axi_series_id = self.get_axi_coord(
            row.study_id, row.level, row.side
        )
        
        if self.phase == 'train':
            instance_number_type = 'filename'
        else:    
            instance_number_type = 'index'

        axi_roi = self.get_roi(
            study_id=row.study_id,
            series_id=axi_series_id,
            img_num=self.img_num,
            resolution=self.resolution,
            roi_size=self.roi_size,
            x_norm=axi_x_norm,
            y_norm=axi_y_norm,
            instance_number_type=instance_number_type,
            instance_number=axi_instance_number,
        )

        if self.transform:
            # sagt2_roi = self.transform(image=sagt2_roi)['image']
            sagt1_roi = self.transform(image=sagt1_roi)['image']
            axi_roi = self.transform(image=axi_roi)['image']

        return axi_roi, sagt1_roi, level, label


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
