import os
import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from rsna2024.runner import Runner
from rsna2024.utils import load_config

if __name__ == '__main__':
    cfg = load_config(filename='config_split')
    Runner(cfg).train()