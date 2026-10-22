"""Released wormOCR weights and the explicit download command.

Weights are published as GitHub release assets, never stored in git, and
never fetched implicitly: only `wormocr download` / `download()` touches the
network. Every file is checked against a SHA-256 recorded here.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from wormocr.exceptions import CheckpointError

REPOSITORY = "arjun988/wormOCR"
DEFAULT_MODEL = "wormocr-2M"


@dataclass(frozen=True)
class Release:
    name: str
    tag: str
    files: dict[str, str]
    description: str

    def url(self, filename: str) -> str:
        return f"https://github.com/{REPOSITORY}/releases/download/{self.tag}/{filename}"


RELEASES: dict[str, Release] = {
    "wormocr-2M": Release(
        name="wormocr-2M",
        tag="wormocr-2M",
        files={
            "wormocr-2M.safetensors": "eee974be652164b485954dd181940c981172bb49c4c2f04dee02fcc06e5ae8e1",
            "wormocr-2M.json": "b2665906e913d8b8496ece94c029f5a01fc588fce4857462e87798d38e20f7ad",
            "wormocr-2M.onnx": "8b7a5bbd92f0f6dd2945b6db2d7530eeca407d07df684b9942a0a5dfcef1cdea",
        },
        description="First wormOCR release: 2,174,880 parameters, printed English (ASCII) text lines.",
    ),
}


def cache_dir() -> Path:
    override = os.environ.get("WORMOCR_HOME")
    if override:
        return Path(override).expanduser()
    base = os.environ.get("XDG_CACHE_HOME") or (Path.home() / ".cache")
    return Path(base) / "wormocr"


def get_release(name: str) -> Release:
    try:
        return RELEASES[name]
    except KeyError:
        available = ", ".join(sorted(RELEASES))
        raise CheckpointError(f"unknown wormOCR model {name!r}. Released models: {available}") from None


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def model_dir(name: str, root: str | Path | None = None) -> Path:
    return (Path(root) if root else cache_dir()) / name


def local_weights(name: str, root: str | Path | None = None) -> Path | None:
    """Path to the downloaded safetensors file, or None if it is not there."""
    release = get_release(name)
    folder = model_dir(name, root)
    weights = folder / f"{release.name}.safetensors"
    config = folder / f"{release.name}.json"
    return weights if weights.exists() and config.exists() else None


def download(
    name: str = DEFAULT_MODEL,
    root: str | Path | None = None,
    force: bool = False,
    include_onnx: bool = True,
    log=print,
) -> Path:
    """Download a released model into the cache and verify every file."""
    release = get_release(name)
    folder = model_dir(name, root)
    folder.mkdir(parents=True, exist_ok=True)
    for filename, expected in release.files.items():
        if filename.endswith(".onnx") and not include_onnx:
            continue
        target = folder / filename
        if target.exists() and not force and sha256(target) == expected:
            log(f"{filename}: already downloaded and verified")
            continue
        url = release.url(filename)
        log(f"{filename}: downloading {url}")
        handle, temporary = tempfile.mkstemp(dir=folder, suffix=".part")
        os.close(handle)
        try:
            with urllib.request.urlopen(url, timeout=60) as response, open(temporary, "wb") as out:
                shutil.copyfileobj(response, out)
            actual = sha256(temporary)
            if actual != expected:
                raise CheckpointError(f"{filename}: SHA-256 mismatch (expected {expected}, got {actual})")
            os.replace(temporary, target)
        except OSError as exc:
            raise CheckpointError(f"{filename}: download failed: {exc}") from exc
        finally:
            if os.path.exists(temporary):
                os.remove(temporary)
        log(f"{filename}: {target.stat().st_size / 1e6:.1f} MB, SHA-256 verified")
    return folder / f"{release.name}.safetensors"


def load_safetensors(path: str | Path, map_location: Any = "cpu") -> dict[str, Any]:
    """Read released weights (`<name>.safetensors` + `<name>.json`) as a checkpoint payload."""
    try:
        from safetensors.torch import load_file
    except ImportError as exc:
        raise CheckpointError("loading .safetensors weights needs: pip install safetensors") from exc
    weights = Path(path)
    meta_path = weights.with_suffix(".json")
    if not meta_path.exists():
        raise CheckpointError(f"{weights} needs its metadata file {meta_path.name} next to it")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    for key in ("config", "charset"):
        if key not in meta:
            raise CheckpointError(f"{meta_path} is missing {key!r}")
    device = str(map_location) if map_location is not None else "cpu"
    return {
        "model_state": load_file(str(weights), device=device),
        "config": meta["config"],
        "charset": meta["charset"],
        "maturity": meta.get("maturity", "untrained"),
        "name": meta.get("name"),
    }
