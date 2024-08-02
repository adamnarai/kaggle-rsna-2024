import os
import numpy as np
import pandas as pd
import pydicom
import cv2
from tqdm import tqdm

from rsna2024.utils import natural_sort


def get_tile(
    study_id,
    series_id,
    x_coord,
    y_coord,
    img_dir,
    img_num,
    prop,
    resolution,
    norm_coords=False,
    block_position='middle',
):
    x = np.zeros((resolution, resolution, img_num), dtype=np.float32)
    series_dir = os.path.join(img_dir, str(study_id), str(series_id))
    file_list = natural_sort(os.listdir(series_dir))

    # Fix direction
    ds_first = pydicom.dcmread(os.path.join(series_dir, file_list[0]))
    ds_last = pydicom.dcmread(os.path.join(series_dir, file_list[-1]))
    pos_diff = np.array(ds_last.ImagePositionPatient) - np.array(ds_first.ImagePositionPatient)
    pos_diff = pos_diff[np.abs(pos_diff).argmax()]
    if pos_diff < 0:
        file_list.reverse()

    slice_num = len(file_list)
    if block_position == 'middle':
        start_index = (slice_num - img_num) // 2
    elif block_position == 'right':
        start_index = round(slice_num * 0.26) - img_num // 2
    elif block_position == 'left':
        start_index = round(slice_num * 0.70) - img_num // 2
    file_list = file_list[start_index : start_index + img_num]

    for i, filename in enumerate(file_list):
        ds = pydicom.dcmread(os.path.join(series_dir, filename))
        img = ds.pixel_array.astype(np.float32)

        if norm_coords and i == 0:
            x_coord = x_coord * img.shape[1]
            y_coord = y_coord * img.shape[0]

        # Crop ROI
        size = min(*img.shape) * prop
        x1 = round(x_coord - (size / 2))
        x2 = round(x_coord + (size / 2))
        y1 = round(y_coord - (size / 2))
        y2 = round(y_coord + (size / 2))
        if any([x1 < 0, x2 > img.shape[1], y1 < 0, y2 > img.shape[0]]):
            break
        img = img[y1:y2, x1:x2]

        # Resize
        img = cv2.resize(img, (resolution, resolution), interpolation=cv2.INTER_CUBIC)
        x[..., i] = img

    # Standardize image
    if x.std() != 0:
        x = (x - x.mean()) / x.std()

    return x

# def get_axi_block(study_id, series_id, img_dir, slice_idx, img_num, resolution, prop, x_coord, y_coord):
#     x = np.zeros((resolution, resolution, img_num), dtype=np.float32)
#     series_dir = os.path.join(img_dir, str(study_id), str(series_id))
#     file_list = natural_sort(os.listdir(series_dir))
#     slice_num = len(file_list)

#     # Fix direction
#     ds_first = pydicom.dcmread(os.path.join(series_dir, file_list[0]))
#     ds_last = pydicom.dcmread(os.path.join(series_dir, file_list[-1]))
#     pos_diff = np.array(ds_last.ImagePositionPatient) - np.array(ds_first.ImagePositionPatient)
#     pos_diff = pos_diff[np.abs(pos_diff).argmax()]
#     if pos_diff < 0:
#         file_list.reverse()
#         slice_idx = slice_num - 1 - slice_idx


#     start_index = slice_idx - img_num // 2
#     file_list = file_list[start_index : start_index + img_num]

#     for i, filename in enumerate(file_list):
#         ds = pydicom.dcmread(os.path.join(series_dir, filename))
#         img = ds.pixel_array.astype(np.float32)
        
#         # Crop ROI
#         size = min(*img.shape) * prop
#         x1 = round(x_coord - (size / 2))
#         x2 = round(x_coord + (size / 2))
#         y1 = round(y_coord - (size / 2))
#         y2 = round(y_coord + (size / 2))
#         if any([x1 < 0, x2 > img.shape[1], y1 < 0, y2 > img.shape[0]]):
#             break
#         img = img[y1:y2, x1:x2]

#         # Resize
#         img = cv2.resize(img, (resolution, resolution), interpolation=cv2.INTER_CUBIC)
#         x[..., i] = img

#     # Standardize image
#     if x.std() != 0:
#         x = (x - x.mean()) / x.std()
        
#     return x

def get_axi_block(study_id, series_id, img_dir, slice_idx, img_num, resolution, prop, x_coord, y_coord):
    x = np.zeros((resolution, resolution, img_num), dtype=np.float32)
    series_dir = os.path.join(img_dir, str(study_id), str(series_id))
    
    file_list = [str(x) + '.dcm' for x in [slice_idx - 1, slice_idx, slice_idx + 1]]

    for i, filename in enumerate(file_list):
        ds_path = os.path.join(series_dir, filename)
        if not os.path.exists(ds_path):
            break
        ds = pydicom.dcmread(ds_path)
        img = ds.pixel_array.astype(np.float32)
        
        # Crop ROI
        size = min(*img.shape) * prop
        x1 = round(x_coord - (size / 2))
        x2 = round(x_coord + (size / 2))
        y1 = round(y_coord - (size / 2))
        y2 = round(y_coord + (size / 2))
        if any([x1 < 0, x2 > img.shape[1], y1 < 0, y2 > img.shape[0]]):
            break
        img = img[y1:y2, x1:x2]

        # Resize
        img = cv2.resize(img, (resolution, resolution), interpolation=cv2.INTER_CUBIC)
        x[..., i] = img

    # Standardize image
    if x.std() != 0:
        x = (x - x.mean()) / x.std()
        
    return x

def get_ds_list(study_id, series_id, img_dir):
    file_list = natural_sort(os.listdir(os.path.join(img_dir, str(study_id), str(series_id))))
    ds_list = [pydicom.dcmread(os.path.join(img_dir, str(study_id), str(series_id), file)) for file in file_list]
    return ds_list   

def get_middle_img(study_id, series_id, img_dir):
    file_list = natural_sort(os.listdir(os.path.join(img_dir, str(study_id), str(series_id))))
    ds = pydicom.dcmread(os.path.join(img_dir, str(study_id), str(series_id), file_list[len(file_list) // 2]))
    return ds

def sagi2axi(sag_y, sag_ds, axi_ds_list):
    axi_z = sag_ds.ImagePositionPatient[2] - sag_y * sag_ds.PixelSpacing[1]
    z_coords = np.array([ds.ImagePositionPatient[2] for ds in axi_ds_list])
    axi_slice = np.argmin(abs(z_coords - axi_z))
    return axi_slice


if __name__ == '__main__':
    # TODO: Remove random sampling from sagt2
    do_sagt2 = False
    do_sagt1 = False
    do_axi = True
    do_axi_from_sagt2 = False

    root = '/media/latlab/MR/projects/kaggle-rsna-2024'
    data_dir = os.path.join(root, 'data')
    raw_data_dir = os.path.join(data_dir, 'raw')
    img_dir = os.path.join(raw_data_dir, 'train_images')
    processed_data_dir = os.path.join(data_dir, 'processed')
    os.makedirs(processed_data_dir, exist_ok=True)

    df_train = pd.read_csv(os.path.join(raw_data_dir, 'train.csv'))
    df_series = pd.read_csv(os.path.join(raw_data_dir, 'train_series_descriptions.csv'))
    df_coordinates = pd.read_csv(os.path.join(processed_data_dir, 'train_label_coordinates.csv'))
    df_coordinates = df_coordinates.merge(df_series, how='left', on=['study_id', 'series_id'])

    # Sagittal T1
    if do_sagt1:
        img_num = 5
        prop = 0.25
        resolution = 128

        out_dir = os.path.join(
            processed_data_dir,
            'tiles_sagt1',
            f'imgnum{img_num}_prop{int(prop*100)}_res{resolution}',
        )
        os.makedirs(out_dir, exist_ok=True)
        df_coordinates_sagt1 = df_coordinates[df_coordinates['series_description'] == 'Sagittal T1']

        # Average left and right coordinates
        # TODO: Remove samples too far away from eachother
        df_coordinates_sagt1 = (
            df_coordinates_sagt1.groupby(['study_id', 'series_id', 'level'])[['x', 'y']]
            .mean()
            .reset_index()
        )
        df_coordinates_sagt1['level'] = (
            df_coordinates_sagt1['level'].str.replace('/', '_').str.lower()
        )

        img_info_list = []
        for row in tqdm(df_coordinates_sagt1.itertuples(), total=len(df_coordinates_sagt1)):
            for side in ['left', 'right']:
                x = get_tile(
                    study_id=row.study_id,
                    series_id=row.series_id,
                    x_coord=row.x,
                    y_coord=row.y,
                    img_dir=img_dir,
                    img_num=img_num,
                    prop=prop,
                    resolution=resolution,
                    block_position=side,
                )
                if np.all(x == 0):
                    continue

                filename = f'{row.study_id}_{row.level}_{side}.npy'
                row_id = f'{side}_neural_foraminal_narrowing_{row.level}'
                label = df_train[df_train['study_id'] == row.study_id][row_id].values[0]
                img_info_list.append((row.study_id, row.series_id, row_id, filename, label))

                # Save image
                out_file = os.path.join(out_dir, filename)
                np.save(out_file, x)

        out_df = pd.DataFrame(
            img_info_list, columns=['study_id', 'series_id', 'row_id', 'filename', 'label']
        )
        out_df.to_csv(os.path.join(out_dir, 'info.csv'), index=False)

    # Sagittal T2/STIR
    if do_sagt2:
        img_num = 5
        prop = 0.25
        resolution = 128

        out_dir = os.path.join(
            processed_data_dir,
            'tiles_sagt2',
            f'imgnum{img_num}_prop{int(prop*100)}_res{resolution}',
        )
        os.makedirs(out_dir, exist_ok=True)
        df_coordinates_sagt2 = df_coordinates[
            df_coordinates['series_description'] == 'Sagittal T2/STIR'
        ].sample(frac=1, random_state=42)

        img_info_list = []
        for row in tqdm(df_coordinates_sagt2.itertuples(), total=len(df_coordinates_sagt2)):
            x = get_tile(
                study_id=row.study_id,
                series_id=row.series_id,
                x_coord=row.x,
                y_coord=row.y,
                img_dir=img_dir,
                img_num=img_num,
                prop=prop,
                resolution=resolution,
            )
            if np.all(x == 0):
                continue

            filename = f'{row.study_id}_{row.row_id[-5:]}.npy'
            label = df_train[df_train['study_id'] == row.study_id][row.row_id].values[0]
            img_info_list.append((row.study_id, row.series_id, row.row_id, filename, label))

            # Save image
            out_file = os.path.join(out_dir, filename)
            np.save(out_file, x)

        out_df = pd.DataFrame(
            img_info_list, columns=['study_id', 'series_id', 'row_id', 'filename', 'label']
        )
        out_df.to_csv(os.path.join(out_dir, 'info.csv'), index=False)

    if do_axi:
        img_num = 3
        prop = 0.5
        resolution = 256

        out_dir = os.path.join(
            processed_data_dir,
            'tiles_axi',
            f'imgnum{img_num}_prop{int(prop*100)}_res{resolution}',
        )
        os.makedirs(out_dir, exist_ok=True)
        df_coordinates_axi = df_coordinates[df_coordinates['series_description'] == 'Axial T2'].sample(frac=1, random_state=42)

        # Use only one image for the two sides
        df_coordinates_axi = (
            df_coordinates_axi.groupby(['study_id', 'series_id', 'level'])
            .agg({'x': 'mean', 'y': 'mean', 'study_id': 'first', 'series_id': 'first', 'row_id': 'first', 'level': 'first', 'instance_number': 'first'})
        )

        img_info_list = []
        for row in tqdm(df_coordinates_axi.itertuples(), total=len(df_coordinates_axi)):
            level = row.row_id[-5:]
            axi_ds_list = get_ds_list(row.study_id, row.series_id, img_dir)

            # Get matching sagit2
            sagi_row = df_coordinates[
                (df_coordinates['study_id'] == row.study_id)
                & (df_coordinates['series_description'] == 'Sagittal T2/STIR')
                & (df_coordinates['level'] == row.level)
            ]
            if len(sagi_row) == 0:
                continue
            sagi_row = sagi_row.iloc[0]
            sagi_ds = get_middle_img(row.study_id, sagi_row.series_id, img_dir)

            # Get axial slice index
            axi_slice_idx = sagi2axi(sagi_row.y, sagi_ds, axi_ds_list)

            x = get_axi_block(
                study_id=row.study_id,
                series_id=row.series_id,
                img_dir=img_dir,
                slice_idx=axi_slice_idx,
                img_num=img_num,
                resolution=resolution,
                prop=prop,
                x_coord=row.x,
                y_coord=row.y,
            )
            if np.all(x == 0):
                continue

            filename = f'{row.study_id}_{level}.npy'
            right_row_id = '_'.join(['right'] + row.row_id.split('_')[1:])
            left_row_id = '_'.join(['left'] + row.row_id.split('_')[1:])
            right_label = df_train[df_train['study_id'] == row.study_id][right_row_id].values[0]
            left_label = df_train[df_train['study_id'] == row.study_id][left_row_id].values[0]
            img_info_list.append((row.study_id, row.series_id, row.row_id, filename, left_label, right_label))

            # Save image
            out_file = os.path.join(out_dir, filename)
            np.save(out_file, x)

        out_df = pd.DataFrame(
            img_info_list, columns=['study_id', 'series_id', 'row_id', 'filename', 'left_label', 'right_label']
        )
        out_df.to_csv(os.path.join(out_dir, 'info.csv'), index=False)

    if do_axi_from_sagt2:
        img_num = 5
        prop = 0.25
        resolution = 128

        out_dir = os.path.join(
            processed_data_dir,
            'tiles_axi_from_sagt2',
            f'imgnum{img_num}_prop{int(prop*100)}_res{resolution}',
        )
        os.makedirs(out_dir, exist_ok=True)
        df_coordinates_sagt2 = df_coordinates[
            df_coordinates['series_description'] == 'Sagittal T2/STIR'
        ].sample(frac=1, random_state=42)

        img_info_list = []
        for row in tqdm(df_coordinates_sagt2.itertuples(), total=len(df_coordinates_sagt2)):
            x = get_tile(
                study_id=row.study_id,
                series_id=row.series_id,
                x_coord=row.x,
                y_coord=row.y,
                img_dir=img_dir,
                img_num=img_num,
                prop=prop,
                resolution=resolution,
            )
            if np.all(x == 0):
                continue

            filename = f'{row.study_id}_{row.row_id[-5:]}.npy'
            left_row_id = 'left_' + row.row_id.replace('spinal_canal_stenosis', 'subarticular_stenosis')
            right_row_id = 'right_' + row.row_id.replace('spinal_canal_stenosis', 'subarticular_stenosis')
            left_label = df_train[df_train['study_id'] == row.study_id][left_row_id].values[0]
            right_label = df_train[df_train['study_id'] == row.study_id][right_row_id].values[0]

            img_info_list.append((row.study_id, row.series_id, row.row_id, filename, left_label, right_label))

            # Save image
            out_file = os.path.join(out_dir, filename)
            np.save(out_file, x)

        out_df = pd.DataFrame(
            img_info_list, columns=['study_id', 'series_id', 'row_id', 'filename', 'left_label', 'right_label']
        )
        out_df.to_csv(os.path.join(out_dir, 'info.csv'), index=False)
