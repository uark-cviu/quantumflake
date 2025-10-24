import io
import os
import sys
import tarfile
import urllib.request
from pathlib import Path
from typing import Optional

_DEFAULT_BRANCH = "main"
_TARBALL_URL = "https://codeload.github.com/Jaluus/MaskTerial/tar.gz/refs/heads/{branch}"

def _cache_root() -> Path:
    return Path.home() / ".cache" / "quantumflake" / "maskterial"

def _find_project_root(base: Path) -> Optional[Path]:
    for p in base.iterdir():
        if not p.is_dir():
            continue
        if (p / "maskterial").is_dir() and (p / "configs").is_dir():
            return p
        for q in p.iterdir():
            if q.is_dir() and (q / "maskterial").is_dir() and (q / "configs").is_dir():
                return q
    return None

def ensure_maskterial_available(cache_dir: Optional[str] = None, branch: str = _DEFAULT_BRANCH) -> Path:
    cache_root = Path(cache_dir) if cache_dir else _cache_root()
    cache_root.mkdir(parents=True, exist_ok=True)
    proj = _find_project_root(cache_root)
    if proj:
        if str(proj) not in sys.path:
            sys.path.append(str(proj))
        return proj

    url = _TARBALL_URL.format(branch=branch)
    print(f"[maskterial-bootstrap] downloading {url} ...")
    with urllib.request.urlopen(url) as resp:
        data = resp.read()

    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tf:
        members = []
        for m in tf.getmembers():
            parts = m.name.split("/")
            if "maskterial" in parts or "configs" in parts:
                members.append(m)
        tf.extractall(path=cache_root, members=members)

    proj = _find_project_root(cache_root)
    if not proj:
        raise FileNotFoundError("Failed to bootstrap MaskTerial: couldn't locate 'maskterial' + 'configs' in cache.")

    if str(proj) not in sys.path:
        sys.path.append(str(proj))
    return proj

def resolve_maskterial_config_path(architecture: str, proj_root: Path) -> str:
    if architecture.startswith("maskterial://"):
        rel = architecture.replace("maskterial://", "", 1).lstrip("/")
        base = proj_root / "configs"
        candidates = [
            base / rel,
            base / rel.lower(),
            base / rel.upper(),
        ]
        for p in candidates:
            if p.is_file():
                return str(p)
        stem = rel.replace(".yaml", "").replace(".yml", "").replace(".py", "")
        for p in [
            base / f"{stem}.py",
            base / f"{stem.lower()}.py",
            base / f"{stem.upper()}.py",
        ]:
            if p.is_file():
                return str(p)
        raise FileNotFoundError(f"MaskTerial config not found under {base} for '{rel}'")
    p = Path(architecture)
    if p.is_file():
        return str(p)
    raise FileNotFoundError(f"Config file not found: {architecture}")
