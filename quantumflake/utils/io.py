import yaml
from pathlib import Path

def resolve_path(path_str: str, base_dir: Path = None) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    
    base = base_dir if base_dir is not None else Path.cwd()
    return base / path

def load_config(path_str: str) -> dict:
    config_path = resolve_path(path_str)

    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found at: '{config_path}'")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    config_dir = config_path.parent
    
    if config.get('models', {}).get('detector', {}).get('weights'):
        config['models']['detector']['weights'] = str(resolve_path(config['models']['detector']['weights'], config_dir))
    
    if config.get('models', {}).get('classifier', {}).get('weights'):
        config['models']['classifier']['weights'] = str(resolve_path(config['models']['classifier']['weights'], config_dir))

    if config.get('calibration_ref_path'):
        config['calibration_ref_path'] = str(resolve_path(config['calibration_ref_path'], config_dir))

    return config

def merge_configs(base, overrides):
    for opt in overrides:
        key, value = opt.split('=', 1)
        if value.lower() == 'true': value = True
        elif value.lower() == 'false': value = False
        else:
            try: value = int(value)
            except ValueError:
                try: value = float(value)
                except ValueError: pass
        
        keys = key.split('.')
        d = base
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        d[keys[-1]] = value
    return base
