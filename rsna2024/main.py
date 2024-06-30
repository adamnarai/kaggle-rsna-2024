import os

from rsna2024.runner import Runner
from rsna2024.utils import load_config

# Fix for random DataLoader hangs
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

if __name__ == "__main__":
    cfg = load_config(filename='config')
    Runner(cfg).train()