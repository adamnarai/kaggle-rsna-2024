import os
import cv2

from rsna2024.runner import Runner
from rsna2024.utils import load_config

# cv2 parallel fix
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

if __name__ == '__main__':
    model_type_list = [
        'coord_sagt2',
        'coord_sagt1',
        'coord_axi',
        'spinal_roi',
        'foraminal_roi',
        'subarticular_roi',
        'global_roi',
    ]
    
    for model_type in model_type_list:
        cfg = load_config(filename=f'config_{model_type}')
        Runner(cfg).train()
