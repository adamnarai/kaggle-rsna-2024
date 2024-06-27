import os
import yaml


def load_config(filename: str = 'config') -> dict:
    """
    Load a YAML configuration file.
    """
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    cfg_path = os.path.join(root, 'experiments', filename + '.yml')
    with open(cfg_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    cfg.update({'root': root})
    return cfg
