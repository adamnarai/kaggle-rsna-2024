import os
import json
import pandas as pd
import numpy as np
import pydicom
import torch
from tqdm import tqdm
import logging
import cv2

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from rsna2024.runner import Runner
from rsna2024.utils import natural_sort, sagi_coord_to_axi_instance_number, get_series

# Params
sagt2_model_name = 'glad-moon-593'
sagt1_model_name = 'skilled-totem-601'
axi_model_name = 'scarlet-feather-603'

root_dir = '/media/latlab/MR/projects/kaggle-rsna-2024'
data_dir = os.path.join(root_dir, 'data', 'raw')
img_dir = os.path.join(data_dir, 'train_images')
df_series = pd.read_csv(
    os.path.join(data_dir, 'train_series_descriptions.csv'),
    dtype={'study_id': 'str', 'series_id': 'str'},
)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

levels = ['L1/L2', 'L2/L3', 'L3/L4', 'L4/L5', 'L5/S1']
sides = ['left', 'right']


def load_config(config_path):
    with open(config_path, encoding='utf-8') as f:
        return json.load(f)


def get_sagi2axi_data(study_id, series_id):
    sag_dir = os.path.join(img_dir, str(study_id), str(series_id))
    axi_dir_list = [
        os.path.join(img_dir, str(study_id), str(axi_series_id))
        for axi_series_id in get_series(df_series, study_id, 'Axial T2')
    ]
    if os.path.isdir(sag_dir) and len(axi_dir_list) != 0:
        sag_file_list = natural_sort(os.listdir(sag_dir))
        mid_sag_ds = pydicom.dcmread(os.path.join(sag_dir, sag_file_list[len(sag_file_list) // 2]))

        axi_ds_list = []
        for axi_dir in axi_dir_list:
            axi_file_list = natural_sort(os.listdir(axi_dir))
            axi_ds_list += [pydicom.dcmread(os.path.join(axi_dir, file)) for file in axi_file_list]
        return mid_sag_ds, axi_ds_list
    else:
        return None, None


def get_coord_from_heatmap_pred(pred, i, idx):
    predi = pred[i, idx, ...].squeeze()
    y_coord, x_coord = np.unravel_index(predi.argmax(), predi.shape)
    x_norm = x_coord / predi.shape[1]
    y_norm = y_coord / predi.shape[0]
    return x_norm, y_norm


# Sagittal T2/STIR coordinates
print('\nPredicting Sagittal T2/STIR coordinates and Axial T2 instance numbers...')
cfg = load_config(os.path.join(root_dir, 'models', 'rsna-2024-' + sagt2_model_name, 'config.json'))
kp_sagt2_preds, kp_sagt2_ys, kp_sagt2_data = Runner(
    cfg, model_name='rsna-2024-' + sagt2_model_name
).predict()

kp_sagt2_data = (
    kp_sagt2_data[['study_id']]
    .merge(
        df_series[df_series['series_description'] == 'Sagittal T2/STIR'], on='study_id', how='left'
    )
    .drop_duplicates('study_id')
)

coord_df_sagt2_list = []
coord_df_axi_list = []
for i in tqdm(range(len(kp_sagt2_data))):
    study_id = kp_sagt2_data.iloc[i]['study_id']
    series_id = kp_sagt2_data.iloc[i]['series_id']

    # Get data for sagi to axi coord projection
    mid_sag_ds, axi_ds_list = get_sagi2axi_data(study_id, series_id)

    for level_idx, level in enumerate(levels):
        # Get sagt2 coordinates
        x_norm, y_norm = get_coord_from_heatmap_pred(kp_sagt2_preds, i, level_idx)
        coord_df_sagt2_list.append(
            {
                'study_id': study_id,
                'series_id': series_id,
                'condition': 'Spinal Canal Stenosis',
                'level': level,
                'x_norm': x_norm,
                'y_norm': y_norm,
                'instance_number': np.nan,
                'row_id': 'spinal_canal_stenosis_' + level.replace('/', '_').lower(),
            }
        )

        # Generate axial instance numbers
        if mid_sag_ds is None or axi_ds_list is None or np.isnan(x_norm) or np.isnan(y_norm):
            axi_series_id, axi_instance_number = np.nan, np.nan
        else:
            axi_series_id, axi_instance_number = sagi_coord_to_axi_instance_number(
                x_norm, y_norm, mid_sag_ds, axi_ds_list
            )
        for side in sides:
            coord_df_axi_list.append(
                {
                    'study_id': kp_sagt2_data.iloc[i]['study_id'],
                    'series_id': axi_series_id,
                    'condition': side.capitalize() + ' Subarticular Stenosis',
                    'level': level,
                    'x_norm': np.nan,
                    'y_norm': np.nan,
                    'instance_number': axi_instance_number,
                    'row_id': side + '_subarticular_stenosis_' + level.replace('/', '_').lower(),
                }
            )
coord_df = pd.concat((pd.DataFrame(coord_df_sagt2_list), pd.DataFrame(coord_df_axi_list)))
del kp_sagt2_preds, kp_sagt2_ys, kp_sagt2_data

# Sagittal T1 coordinates
print('\nPredicting Sagittal T1 coordinates...')
cfg = load_config(os.path.join(root_dir, 'models', 'rsna-2024-' + sagt1_model_name, 'config.json'))
kp_sagt1_preds, kp_sagt1_ys, kp_sagt1_data = Runner(
    cfg, model_name='rsna-2024-' + sagt1_model_name
).predict()

kp_sagt1_data = (
    kp_sagt1_data[['study_id']]
    .merge(df_series[df_series['series_description'] == 'Sagittal T1'], on='study_id', how='left')
    .drop_duplicates('study_id')
)

coord_df_list = []
for i in tqdm(range(len(kp_sagt1_data))):
    for side_idx, level in enumerate(levels):
        for side in sides:
            # Get coordinates
            x_norm, y_norm = get_coord_from_heatmap_pred(kp_sagt1_preds, i, level_idx)
            coord_df_list.append(
                {
                    'study_id': kp_sagt1_data.iloc[i]['study_id'],
                    'series_id': kp_sagt1_data.iloc[i]['series_id'],
                    'condition': side.capitalize() + ' Neural Foraminal Narrowing',
                    'level': level,
                    'x_norm': x_norm,
                    'y_norm': y_norm,
                    'instance_number': np.nan,
                    'row_id': side
                    + '_neural_foraminal_narrowing_'
                    + level.replace('/', '_').lower(),
                }
            )
coord_df = pd.concat((coord_df, pd.DataFrame(coord_df_list)))
del kp_sagt1_preds, kp_sagt1_ys, kp_sagt1_data


# Axial T2 coordinates
print('\nPredicting Axial T2 coordinates...')
cfg = load_config(os.path.join(root_dir, 'models', 'rsna-2024-' + axi_model_name, 'config.json'))
kp_axi_preds, kp_axi_ys, kp_axi_data = Runner(
    cfg, model_name='rsna-2024-' + axi_model_name
).predict(df_coordinates=coord_df)

for i in tqdm(range(len(kp_axi_data))):
    study_id = kp_axi_data.iloc[i]['study_id']
    for level in levels:
        level = level.replace('/', '_').lower()
        for side_idx, side in enumerate(sides):
            # Get series_id from coord_df
            series_id = coord_df.loc[
                (coord_df['study_id'] == study_id)
                & (coord_df['row_id'] == side + '_subarticular_stenosis_' + level),
                'series_id',
            ].values[0]

            # Get coordinates
            x_norm, y_norm = get_coord_from_heatmap_pred(kp_axi_preds, i, side_idx)

            # Insert into coord_df
            coord_df.loc[
                (coord_df['study_id'] == study_id)
                & (coord_df['series_id'] == series_id)
                & (coord_df['row_id'] == side + '_subarticular_stenosis_' + level),
                ['x_norm', 'y_norm'],
            ] = [x_norm, y_norm]

# Save coords
coord_df.to_csv(
    os.path.join(root_dir, 'data', 'processed', 'train_label_coordinates_predicted.csv'),
    index=False,
)
print('Done.')