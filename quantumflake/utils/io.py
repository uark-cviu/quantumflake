import yaml
from pathlib import Path

def get_project_root() -> Path:
    return Path(__file__).parent.parent.parent

def resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return get_project_root() / path

def load_config(path: str) -> dict:
    config_path = resolve_path(path)
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def merge_configs(base, overrides):
    for opt in overrides:
        key, value = opt.split('=', 1)
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
                    pass

        keys = key.split('.')
        d = base
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        d[keys[-1]] = value
    return base