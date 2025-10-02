import os
import sys
import io
import tarfile
import tempfile
import urllib.request
from pathlib import Path

_DEFAULT_BRANCH = "main"
_TARBALL_URL = f"https://codeload.github.com/facebookresearch/detectron2/tar.gz/refs/heads/{{branch}}"

def _cache_root():
    return Path(os.path.expanduser("~")) / ".cache" / "quantumflake" / "vitdet"

def _project_root(cache_dir: Path, branch: str):
    return cache_dir / f"detectron2-{branch}" / "projects" / "ViTDet"

def ensure_vitdet_available(cache_dir: str = None, branch: str = _DEFAULT_BRANCH) -> Path:
    cache_dir = Path(cache_dir) if cache_dir else _cache_root()
    proj = _project_root(cache_dir, branch)
    if proj.exists():
        if str(proj) not in sys.path:
            sys.path.append(str(proj))
        return proj

    cache_dir.mkdir(parents=True, exist_ok=True)
    url = _TARBALL_URL.format(branch=branch)
    print(f"[vitdet-bootstrap] downloading {url} ...")
    with urllib.request.urlopen(url) as resp:
        data = resp.read()

    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tf:
        members = [m for m in tf.getmembers() if "/projects/ViTDet/" in m.name]
        tf.extractall(path=cache_dir, members=members)

    if str(proj) not in sys.path:
        sys.path.append(str(proj))
    return proj

def resolve_vitdet_config_path(architecture: str, proj_root: Path) -> str:
    assert architecture.startswith("vitdet://"), "expected vitdet:// URI"
    rel = architecture.replace("vitdet://", "", 1).lstrip("/")

    cfg_path = proj_root / "configs" / rel
    if cfg_path.is_file():
        return str(cfg_path)

    stem = rel.replace(".yaml", "").replace(".yml", "").replace(".py", "")
    candidates = [
        proj_root / "configs" / f"{stem}.py",
        proj_root / "configs" / f"{stem.lower()}.py",
        proj_root / "configs" / f"{stem.upper()}.py",
    ]
    for p in candidates:
        if p.is_file():
            return str(p)

    raise FileNotFoundError(f"ViTDet config not found under: {proj_root / 'configs'} for '{rel}'")
