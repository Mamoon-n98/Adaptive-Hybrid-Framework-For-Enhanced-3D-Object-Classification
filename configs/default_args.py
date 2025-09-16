# Process starts: Imports.
import yaml
import os
# Process ends: Imports complete.

# Process starts: Load configuration function.
def get_default_args():
    """
    Load configuration from modelnet40_config.yaml.
    :return: Configuration dictionary.
    """
    config_path = '/root/autodl-tmp/Adaptive-Hybrid-Framework-For-Enhanced-3D-Object-Classification/configs/modelnet40_config.yaml'
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config
# Process ends: Load configuration function complete.
