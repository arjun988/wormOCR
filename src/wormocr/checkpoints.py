"""Checkpoint format.

Checkpoints are PyTorch pickle files. Load only files you trust.
They store the model, optimizer, charset, config, and a maturity label.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from wormocr import __version__
from wormocr.exceptions import CheckpointError

FORMAT_VERSION = 1


def public_config(config: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in config.items() if not str(key).startswith("_")}


def save_checkpoint(path: str | Path, payload: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    body = dict(payload)
    body["wormocr_format"] = FORMAT_VERSION
    body["wormocr_version"] = __version__
    torch.save(body, destination)


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    source = Path(path)
    if not source.exists():
        raise CheckpointError(
            f"checkpoint not found: {source}. Download the released model with "
            "`wormocr download`, train one with `wormocr train --config configs/tiny.yaml`, "
            "or pass an explicit --checkpoint."
        )
    if source.suffix == ".safetensors":
        from wormocr.hub import load_safetensors

        return load_safetensors(source, map_location=map_location)
    try:
        payload = torch.load(source, map_location=map_location, weights_only=False)
    except TypeError:
        payload = torch.load(source, map_location=map_location)
    except Exception as exc:
        raise CheckpointError(f"cannot read checkpoint {source}: {exc}") from exc
    if not isinstance(payload, dict) or "model_state" not in payload:
        raise CheckpointError(f"{source} is not a wormOCR checkpoint")
    if "charset" not in payload or "config" not in payload:
        raise CheckpointError(f"{source} is missing its charset or config")
    return payload
