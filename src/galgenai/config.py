"""to load config"""

import os
import yaml
from pathlib import Path
from typing import Dict, Optional

# Repository root: go up from src/galgenai/ to repository root
REPO_ROOT = Path(__file__).parent.parent.parent

def load_config(config_path: Optional[str] = None) -> Dict:
    """
    Load configuration from YAML file and resolve relative paths.

    Relative paths in the config are resolved relative to the repository root.
    Absolute paths are kept as-is.

    Parameters
    ----------
    config_path : str, optional
        Explicit path to config file. If None, searches default locations.

    Returns
    -------
    dict
        Configuration dictionary with resolved paths.
        Returns empty dict if no config found.
    """
    if config_path is None:
        config_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "galgenai_config.yaml",
        )
        print(config_path)
    config_path = Path(config_path)

    if config_path.exists() and config_path.is_file():
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                if config is None:
                    config = {}

                # Resolve relative paths in cosmos section
                if 'cosmos' in config:
                    cosmos = config['cosmos']
                    # Resolve catalog_path if relative
                    if 'catalog_path' in cosmos:
                        path = Path(cosmos['catalog_path'])
                        if not path.is_absolute():
                            cosmos['catalog_path'] = str(REPO_ROOT / path)
                    # Resolve hf_dataset_path if relative
                    if 'hf_dataset_path' in cosmos:
                        path = Path(cosmos['hf_dataset_path'])
                        if not path.is_absolute():
                            cosmos['hf_dataset_path'] = str(REPO_ROOT / path)

                # Resolve relative paths in training section
                if 'training' in config and 'output_dir' in config['training']:
                    path = Path(config['training']['output_dir'])
                    if not path.is_absolute():
                        config['training']['output_dir'] = str(REPO_ROOT / path)

                return config
        except Exception as e:
            raise ValueError(f"Failed to load config from {config_path}: {e}")

    return {}
