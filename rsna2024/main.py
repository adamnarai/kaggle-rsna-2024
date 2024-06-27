from rsna2024.runner import Runner
from rsna2024.utils import load_config

if __name__ == "__main__":
    cfg = load_config(filename='config')
    runner = Runner(cfg)
    runner.train()