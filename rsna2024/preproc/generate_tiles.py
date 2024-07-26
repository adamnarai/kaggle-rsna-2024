import os
import numpy as np
import pandas as pd
import pydicom
import cv2
from tqdm import tqdm

from rsna2024.utils import natural_sort

# Params
img_num = 5
prop = 0.25
resolution = 64

root = '/media/latlab/MR/projects/kaggle-rsna-2024'
data_dir = os.path.join(root, 'data')
raw_data_dir = os.path.join(data_dir, 'raw')
processed_data_dir = os.path.join(data_dir, 'processed')
os.makedirs(processed_data_dir, exist_ok=True)

df_train = pd.read_csv(os.path.join(raw_data_dir, 'train.csv'))
df_series = pd.read_csv(os.path.join(raw_data_dir, 'train_series_descriptions.csv'))
df_coordinates = pd.read_csv(os.path.join(processed_data_dir, 'train_label_coordinates.csv'))
df_coordinates = df_coordinates.merge(df_series, how='left', on=['study_id', 'series_id'])

out_dir = os.path.join(
    processed_data_dir, 'tiles_sagt2', f'imgnum{img_num}_prop{int(prop*100)}_res{resolution}'
)
os.makedirs(out_dir, exist_ok=True)

df_coordinates_sagt2 = df_coordinates[
    df_coordinates['series_description'] == 'Sagittal T2/STIR'
].sample(frac=1, random_state=42)

img_info_list = []
for row in tqdm(df_coordinates_sagt2.itertuples(), total=len(df_coordinates_sagt2)):
    x = np.zeros((resolution, resolution, img_num), dtype=np.float32)
    series_dir = os.path.join(raw_data_dir, 'train_images', str(row.study_id), str(row.series_id))
    file_list = natural_sort(os.listdir(series_dir))

    # Fix direction
    ds_first = pydicom.dcmread(os.path.join(series_dir, file_list[0]))
    ds_last = pydicom.dcmread(os.path.join(series_dir, file_list[-1]))
    pos_diff = np.array(ds_last.ImagePositionPatient) - np.array(ds_first.ImagePositionPatient)
    pos_diff = pos_diff[np.abs(pos_diff).argmax()]
    if pos_diff < 0:
        file_list.reverse()

    slice_num = len(file_list)
    start_index = (slice_num - img_num) // 2
    file_list = file_list[start_index : start_index + img_num]

    for i, filename in enumerate(file_list):
        ds = pydicom.dcmread(os.path.join(series_dir, filename))
        img = ds.pixel_array.astype(np.float32)

        # Crop ROI
        size = min(*img.shape) * prop
        x1 = round(row.x - (size / 2))
        x2 = round(row.x + (size / 2))
        y1 = round(row.y - (size / 2))
        y2 = round(row.y + (size / 2))
        if any([x1 < 0, x2 > img.shape[1], y1 < 0, y2 > img.shape[0]]):
            break
        img = img[y1:y2, x1:x2]

        # Resize
        img = cv2.resize(img, (resolution, resolution), interpolation=cv2.INTER_CUBIC)
        x[..., i] = img
    if np.all(x == 0):
        continue

    # Standardize image
    x = (x - x.mean()) / x.std()

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
