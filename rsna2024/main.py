import os
import cv2

from rsna2024.runner import Runner
from rsna2024.utils import load_config
from rsna2024.preproc import normalize_coordinates

# cv2 parallel fix
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

seg_model_type_list = [
    'coord_sagt2',
    'coord_sagt1',
    'coord_axi',
]

roi_model_type_list = [
    'spinal_roi',
    'foraminal_roi',
    'subarticular_roi',
    'global_roi',
]

base_model_dict = {
    'resnet18': {'resolution': 128, 'lr': 1e-3},
    'swin_tiny_patch4_window7_224': {'resolution': 224, 'lr': 3e-5},
    'convnext_nano': {'resolution': 224, 'lr': 3e-5},
}

if __name__ == '__main__':
    # Preprocessing: normalize coordinates
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    print('Normalizing coordinates...')
    normalize_coordinates(root_dir)
    
    # Train segmentation models
    for model_type in seg_model_type_list:
        cfg = load_config(filename=f'config_{model_type}')
        Runner(cfg).train()

    # Train ROI models
    for base_model, params in base_model_dict.items():
        for model_type in roi_model_type_list:
            cfg = load_config(filename=f'config_{model_type}')
            cfg['model']['args']['base_model'] = base_model
            cfg['dataset']['args']['resolution'] = params['resolution']
            cfg['optimizer']['args']['lr'] = params['lr']
            Runner(cfg).train()