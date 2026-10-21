"""YAML configuration, defaults, and dotted command-line overrides."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from wormocr.exceptions import ConfigError

DEFAULT_CONFIG: dict[str, Any] = {
    "seed": 1337,
    "device": "auto",
    "model": {
        "name": "tiny",
        "in_channels": 1,
        "image_height": 32,
        "channels": [32, 64, 128, 192],
        "separable": True,
        "rnn_hidden": 320,
        "rnn_layers": 2,
        "rnn_dropout": 0.1,
        "proj_size": 256,
    },
    "vocab": {"charset": None},
    "data": {
        "train_samples": 4000,
        "val_samples": 400,
        "val_seed": None,
        "font_size_min": 16,
        "font_size_max": 32,
        "max_chars": 40,
        "augment": True,
        "font_dir": None,
        "labels": None,
        "val_labels": None,
        "image_root": None,
        "val_image_root": None,
        "max_width": 1024,
    },
    "train": {
        "epochs": 10,
        "batch_size": 32,
        "lr": 0.001,
        "min_lr_ratio": 0.1,
        "weight_decay": 0.01,
        "grad_clip": 5.0,
        "warmup_steps": 100,
        "max_steps": None,
        "max_minutes": None,
        "early_stopping_patience": 5,
        "checkpoint_dir": "artifacts/checkpoints/tiny",
        "log_every": 20,
        "save_every": 5,
        "amp": False,
        "num_workers": 0,
    },
    "eval": {"batch_size": 32, "beam_width": 0, "synthetic_samples": 200},
    "export": {"opset": 17},
}

MODE_PRESETS: dict[str, dict[str, Any]] = {
    "smoke": {
        "train": {
            "epochs": 1,
            "max_steps": 2,
            "batch_size": 4,
            "log_every": 1,
            "early_stopping_patience": 0,
            "warmup_steps": 0,
            "save_every": 0,
            "max_minutes": 5,
        },
        "data": {"train_samples": 8, "val_samples": 4, "augment": False},
    },
    "debug": {
        "train": {
            "epochs": 1,
            "max_steps": 40,
            "log_every": 5,
            "early_stopping_patience": 0,
            "warmup_steps": 0,
            "max_minutes": 15,
        },
        "data": {"train_samples": 128, "val_samples": 32},
    },
    "baseline": {},
    "resume": {},
}


def repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").exists() and (parent / "configs").exists():
            return parent
    return Path.cwd()


def config_path(name: str) -> Path:
    candidate = Path(name)
    if candidate.exists():
        return candidate
    rooted = repo_root() / name
    if rooted.exists():
        return rooted
    by_stem = repo_root() / "configs" / f"{name}.yaml"
    if by_stem.exists():
        return by_stem
    raise ConfigError(f"config not found: {name}")


def deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged


def parse_override_value(raw: str) -> Any:
    try:
        return yaml.safe_load(raw)
    except yaml.YAMLError:
        return raw


def set_dotted(config: dict[str, Any], dotted: str, value: Any) -> None:
    if not dotted or dotted.startswith(".") or dotted.endswith("."):
        raise ConfigError(f"invalid override key: {dotted!r}")
    cursor = config
    keys = dotted.split(".")
    for key in keys[:-1]:
        child = cursor.get(key)
        if not isinstance(child, dict):
            child = {}
            cursor[key] = child
        cursor = child
    cursor[keys[-1]] = value


def apply_overrides(config: dict[str, Any], overrides: list[str] | None) -> dict[str, Any]:
    updated = config
    for item in overrides or []:
        if "=" not in item:
            raise ConfigError(f"override must look like key=value, got {item!r}")
        key, raw = item.split("=", 1)
        set_dotted(updated, key.strip(), parse_override_value(raw))
    return updated


def apply_mode(config: dict[str, Any], mode: str, labels_set: bool = False) -> dict[str, Any]:
    if mode not in MODE_PRESETS:
        known = ", ".join(sorted(MODE_PRESETS))
        raise ConfigError(f"unknown train mode {mode!r}. Expected one of: {known}")
    preset = MODE_PRESETS[mode]
    if mode == "smoke" and labels_set:
        preset = {"train": preset.get("train", {})}
    return deep_update(config, preset)


def validate_config(config: dict[str, Any]) -> None:
    channels = config["model"]["channels"]
    if not isinstance(channels, list) or len(channels) != 4:
        raise ConfigError("model.channels must be a list of four positive integers")
    if any(int(channel) < 4 for channel in channels):
        raise ConfigError("each model channel count must be at least 4")
    height = int(config["model"]["image_height"])
    if height % 16 != 0 or height < 16:
        raise ConfigError("model.image_height must be a multiple of 16 and at least 16")
    if int(config["model"]["rnn_hidden"]) < 8:
        raise ConfigError("model.rnn_hidden must be at least 8")
    if int(config["model"]["rnn_layers"]) < 1:
        raise ConfigError("model.rnn_layers must be at least 1")
    if int(config["train"]["batch_size"]) < 1:
        raise ConfigError("train.batch_size must be at least 1")
    if int(config["data"]["train_samples"]) < 2:
        raise ConfigError("data.train_samples must be at least 2")
    if float(config["train"]["lr"]) <= 0:
        raise ConfigError("train.lr must be positive")


def load_config(
    path: str | Path,
    overrides: list[str] | None = None,
    mode: str | None = None,
) -> dict[str, Any]:
    source = config_path(str(path))
    try:
        loaded = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    except OSError as exc:
        raise ConfigError(f"cannot read config {source}") from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML in {source}: {exc}") from exc
    if not isinstance(loaded, dict):
        raise ConfigError(f"config root must be a mapping: {source}")
    config = deep_update(DEFAULT_CONFIG, loaded)
    config = apply_overrides(config, overrides)
    if mode:
        config = apply_mode(config, mode, labels_set=bool(config["data"].get("labels")))
        config = apply_overrides(config, overrides)
    validate_config(config)
    config["_config_path"] = str(source)
    return config


def dump_config(config: dict[str, Any]) -> str:
    public = {key: value for key, value in config.items() if not str(key).startswith("_")}
    return yaml.safe_dump(public, sort_keys=False)
