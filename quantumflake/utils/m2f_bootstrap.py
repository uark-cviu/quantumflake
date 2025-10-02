import io
import sys
import tarfile
import urllib.request
from pathlib import Path
from typing import Optional

_TARBALLS = [
    "https://codeload.github.com/facebookresearch/Mask2Former/tar.gz/refs/heads/main",
    "https://codeload.github.com/facebookresearch/Mask2Former/tar.gz/main",
]

def _cache_root() -> Path:
    return Path.home() / ".cache" / "quantumflake" / "mask2former"

def _find_parent_with_pkg(base: Path) -> Optional[Path]:
    for p in base.rglob("mask2former"):
        if (p / "__init__.py").exists():
            return p.parent
    return None

def ensure_mask2former_available(cache_dir: Optional[str] = None) -> Path:
    cache_root = Path(cache_dir) if cache_dir else _cache_root()
    cache_root.mkdir(parents=True, exist_ok=True)

    parent = _find_parent_with_pkg(cache_root)
    if parent:
        if str(parent) not in sys.path:
            sys.path.append(str(parent))
        return parent

    data = None
    last_err = None
    for url in _TARBALLS:
        try:
            print(f"[mask2former-bootstrap] downloading {url} ...")
            with urllib.request.urlopen(url) as resp:
                data = resp.read()
                break
        except Exception as e:
            last_err = e
            continue
    if data is None:
        raise RuntimeError(f"Failed to fetch Mask2Former tarball (last error: {last_err})")

    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tf:
        members = [
            m for m in tf.getmembers()
            if "/mask2former/" in m.name or m.name.endswith("/mask2former")
        ]
        tf.extractall(path=cache_root, members=members)

    parent = _find_parent_with_pkg(cache_root)
    if not parent:
        raise FileNotFoundError("mask2former package not found after extraction")
    if str(parent) not in sys.path:
        sys.path.append(str(parent))
    return parent
