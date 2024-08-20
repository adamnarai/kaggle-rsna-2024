import os
import re
import numpy as np
import pydicom


def natural_sort(l):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)


def get_series(df_series, study_id, series_description):
    series_list = df_series[
        (df_series['study_id'] == study_id)
        & (df_series['series_description'] == series_description)
    ]['series_id'].tolist()
    if len(series_list) == 0:
        return None
    return series_list


def sagi_coord_to_axi_instance_number(sag_x_norm, sag_y_norm, mid_sag_ds, axi_ds_list):
    # Calculate sagittal world coordinates based on middle slice
    sag_affine = dcm_affine(mid_sag_ds)
    sag_coord = np.array([sag_y_norm * mid_sag_ds.Rows, sag_x_norm * mid_sag_ds.Columns, 0, 1])
    sag_world_coord = (sag_affine @ sag_coord)[:-1]

    # Get closest axial slice
    dist_list = []
    for ds in axi_ds_list:
        normal = np.cross(ds.ImageOrientationPatient[:3], ds.ImageOrientationPatient[3:])
        normal /= np.linalg.norm(normal)
        dist = np.abs(np.dot(sag_world_coord - ds.ImagePositionPatient, normal))
        dist_list.append(dist)
    axi_slice_idx = np.argmin(dist_list)
    min_dist = dist_list[axi_slice_idx]
    if min_dist > 5:
        return None, np.nan
    axi_series_id, axi_instance_number = axi_ds_list[axi_slice_idx].filename.split('/')[-2:]

    return axi_series_id, int(axi_instance_number.replace('.dcm', ''))


def dcm_affine(ds):
    F11, F21, F31 = ds.ImageOrientationPatient[3:]
    F12, F22, F32 = ds.ImageOrientationPatient[:3]
    dr, dc = ds.PixelSpacing
    Sx, Sy, Sz = ds.ImagePositionPatient

    return np.array(
        [
            [F11 * dr, F12 * dc, 0, Sx],
            [F21 * dr, F22 * dc, 0, Sy],
            [F31 * dr, F32 * dc, 0, Sz],
            [0, 0, 0, 1],
        ]
    )
