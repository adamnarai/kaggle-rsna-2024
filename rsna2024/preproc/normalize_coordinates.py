import os
import numpy as np
import pandas as pd
import pydicom

from rsna2024.utils import natural_sort

root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
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
    dcm_files = natural_sort(os.listdir(dcm_folder))
    file_num = len(dcm_files)
    filename = str(s.instance_number) + '.dcm'
    ds = pydicom.dcmread(os.path.join(dcm_folder, filename))
    ds_first = pydicom.dcmread(os.path.join(dcm_folder, dcm_files[0]))
    ds_last = pydicom.dcmread(os.path.join(dcm_folder, dcm_files[-1]))
    pos_diff = np.array(ds_last.ImagePositionPatient) - np.array(ds_first.ImagePositionPatient)
    pos_diff = pos_diff[np.abs(pos_diff).argmax()]
    if pos_diff >= 0:
        slice_dir = 0
    else:
        slice_dir = 1
    return [
        s.x / ds.Columns,
        s.y / ds.Rows,
        dcm_files.index(filename) / file_num,
        file_num,
        int(ds.InstanceNumber),
        slice_dir,
    ]


df[['x_norm', 'y_norm', 'instance_number_norm', 'file_num', 'dcm_instance_number', 'slice_dir']] = (
    df.apply(normalize, axis=1, result_type='expand')
)

df.to_csv(out_path, index=False)
