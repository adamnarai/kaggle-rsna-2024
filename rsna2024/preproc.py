import os
import numpy as np
import pandas as pd
import pydicom

root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
data_dir = os.path.join(root, 'data')
raw_data_dir = os.path.join(data_dir, 'raw')
processed_data_dir = os.path.join(data_dir, 'processed')
in_path = os.path.join(raw_data_dir, 'train_label_coordinates.csv')
out_path = os.path.join(processed_data_dir, 'train_label_coordinates.csv')
os.makedirs(processed_data_dir, exist_ok=True)

df = pd.read_csv(in_path, dtype={'study_id': 'str', 'series_id': 'str'})
condition = df['condition'].str.replace(' ', '_').str.lower()
level = df['level'].str.replace('/', '_').str.lower()
df['row_id'] = condition + '_' + level

def normalize(s):
    dcm_folder = os.path.join(raw_data_dir, 'train_images', s.study_id, s.series_id)
    dcm_files = os.listdir(dcm_folder)
    if len(dcm_files) == 0:
        return [np.nan, np.nan]
    ds = pydicom.dcmread(os.path.join(dcm_folder, dcm_files[0]))
    return [s.x/ds.Columns, s.y/ds.Rows]

df[['x_norm', 'y_norm']] = df.apply(normalize, axis=1, result_type='expand')

df.to_csv(out_path, index=False)
