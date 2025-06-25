# quantumflake/utils/io.py

import yaml
from pathlib import Path

def get_project_root() -> Path:
    """Returns the project root folder."""
    return Path(__file__).parent.parent.parent

def resolve_path(path_str: str) -> Path:
    """
    Resolves a path string. If it's absolute, return it.
    If it's relative, resolve it from the project root.
    """
    path = Path(path_str)
    if path.is_absolute():
        return path
    return get_project_root() / path

def load_config(path: str) -> dict:
    """Loads a YAML configuration file."""
    config_path = resolve_path(path)
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def merge_configs(base, overrides):
    """Merges CLI overrides into a base config dict."""
    # Simplified parser for key=value pairs
    for opt in overrides:
        key, value = opt.split('=', 1)
        # Attempt to cast value to a more specific type
        if value.lower() == 'true':
            value = True
        elif value.lower() == 'false':
            value = False
        else:
            try:
                value = int(value)
            except ValueError:
                try:
                    value = float(value)
                except ValueError:
                    pass # Keep as string
        
        # Navigate nested keys if necessary
        keys = key.split('.')
        d = base
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        d[keys[-1]] = value
    return base
