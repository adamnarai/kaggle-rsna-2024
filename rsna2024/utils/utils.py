import os
import re
import numpy as np
import pydicom


def natural_sort(l):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)


def sagi_coord_to_axi_instance_number(sag_x_norm, sag_y_norm, sag_dir, axi_dir_list):

    # Calculate sagittal world coordinates based on middle slice
    sag_file_list = natural_sort(os.listdir(sag_dir))
    mid_sag_ds = pydicom.dcmread(os.path.join(sag_dir, sag_file_list[len(sag_file_list) // 2]))
    sag_affine = dcm_affine(mid_sag_ds)
    sag_coord = np.array([sag_y_norm * mid_sag_ds.Rows, sag_x_norm * mid_sag_ds.Columns, 0, 1])
    sag_world_coord = sag_affine @ sag_coord

    # Get all axi ds
    axi_ds_list = []
    for axi_dir in axi_dir_list:
        axi_file_list = natural_sort(os.listdir(axi_dir))
        axi_ds_list += [pydicom.dcmread(os.path.join(axi_dir, file)) for file in axi_file_list]
    
    # Get closest axial slice
    axi_coord_list = []
    for ds in axi_ds_list:
        affine = dcm_affine(ds)
        axi_coord_list.append(affine @ np.array([ds.Rows // 2, ds.Columns // 2, 0, 1]))
    axi_slice_idx = np.argmin(
        [sum((axi_coord - sag_world_coord) ** 2) for axi_coord in axi_coord_list]
    )

    return int(axi_slice_idx)


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
