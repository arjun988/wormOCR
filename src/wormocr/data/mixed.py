"""Weighted mixtures of synthetic printed, synthetic handwritten, and real lines.

`data.sources` in a config lists the parts of the mixture:

    sources:
      - {kind: synthetic, weight: 0.35}
      - {kind: handwriting, weight: 0.35}
      - {kind: labels, weight: 0.2, labels: path/train.tsv}

Which source item i comes from is a pure function of (seed, i), so a run is
reproducible regardless of worker count. Real lines get light, random
augmentation; synthetic items keep their own deterministic augmentation.
"""

from __future__ import annotations

import bisect
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter
from torch.utils.data import ConcatDataset, Dataset, Subset

from wormocr.data.dataset import LabelDataset, OnlineSynthDataset, _read_labels
from wormocr.data.handwriting import handwriting_fonts, render_handwriting, sample_hw_text
from wormocr.data.synthetic import _jpeg, _noise, fit_label_width
from wormocr.exceptions import ConfigError, DataError
from wormocr.preprocessing import prepare_line
from wormocr.vocabulary import Vocabulary


def _finish(image: Image.Image, text: str, config: dict) -> dict:
    model_cfg = config["model"]
    data_cfg = config["data"]
    image = fit_label_width(
        image,
        text,
        int(model_cfg["image_height"]),
        list(model_cfg["channels"]),
        int(data_cfg["max_width"]),
        int(model_cfg.get("width_stride", 2)),
    )
    tensor, pad_value = prepare_line(
        image,
        height=int(model_cfg["image_height"]),
        align=2,
        max_width=max(int(data_cfg["max_width"]), image.width),
        min_width=8,
    )
    return {"image": tensor, "text": text, "pad_value": pad_value, "width": int(tensor.shape[-1])}


class SynthHandwritingDataset(Dataset):
    def __init__(self, config: dict, samples: int, seed: int, augment: bool, font_dir: str = "") -> None:
        self.config = config
        self.samples = int(samples)
        self.seed = int(seed)
        self.augment = bool(augment)
        self.fonts = handwriting_fonts(font_dir or config["data"].get("handwriting_font_dir") or "")
        self.vocabulary = Vocabulary((config.get("vocab") or {}).get("charset"))

    def __len__(self) -> int:
        return self.samples

    def __getitem__(self, index: int) -> dict:
        seed = self.seed + int(index) * 10007
        rng = random.Random(seed)
        np_rng = np.random.default_rng(seed)
        text = sample_hw_text(rng, int(self.config["data"]["max_chars"]))
        self.vocabulary.encode(text)
        image = render_handwriting(text, rng, np_rng, self.fonts, augment=self.augment)
        return _finish(image, text, self.config)


def _augment_real(image: Image.Image, rng: random.Random) -> Image.Image:
    """Mild changes only: real crops already carry real noise and their labels must stay true."""
    image = image.convert("L")
    if rng.random() < 0.3:
        angle = rng.uniform(-2.0, 2.0)
        fill = int(np.median(np.asarray(image)))
        image = image.rotate(angle, resample=Image.Resampling.BILINEAR, expand=True, fillcolor=fill)
    if rng.random() < 0.3:
        shear = rng.uniform(-0.15, 0.15)
        fill = int(np.median(np.asarray(image)))
        image = image.transform(
            image.size, Image.Transform.AFFINE, (1, shear, -shear * image.height / 2, 0, 1, 0),
            resample=Image.Resampling.BILINEAR, fillcolor=fill,
        )
    if rng.random() < 0.3:
        scale = rng.uniform(0.85, 1.15)
        image = image.resize((max(8, int(image.width * scale)), image.height), Image.Resampling.BILINEAR)
    if rng.random() < 0.25:
        array = np.asarray(image).astype(np.float32)
        mean = array.mean()
        array = (array - mean) * rng.uniform(0.7, 1.2) + mean + rng.uniform(-20, 20)
        image = Image.fromarray(np.clip(array, 0, 255).astype(np.uint8), mode="L")
    if rng.random() < 0.15:
        image = image.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.3, 0.9)))
    if rng.random() < 0.15:
        image = _noise(image, np.random.default_rng(rng.getrandbits(32)), rng.uniform(2, 8))
    if rng.random() < 0.15:
        image = _jpeg(image, rng.randint(40, 85))
    return image


class RealLinesDataset(Dataset):
    """Prepared line crops (`relative<TAB>text`), sized for CTC like synthetic lines."""

    def __init__(self, labels: str | Path, config: dict, augment: bool, image_root: str | Path | None = None) -> None:
        self.labels = Path(labels)
        self.root = Path(image_root) if image_root else self.labels.parent
        self.config = config
        self.augment = bool(augment)
        vocabulary = Vocabulary((config.get("vocab") or {}).get("charset"))
        self.records = []
        for relative, text in _read_labels(self.labels):
            try:
                vocabulary.encode(text)
            except ValueError:
                continue
            if text.strip():
                self.records.append((relative, text))
        if not self.records:
            raise DataError(f"no usable labels in {self.labels}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        relative, text = self.records[int(index)]
        with Image.open(self.root / relative) as handle:
            image = handle.convert("L")
        if self.augment:
            image = _augment_real(image, random.Random())
        item = _finish(image, text, self.config)
        item["path"] = str(self.root / relative)
        return item


class MixedDataset(Dataset):
    def __init__(self, parts: list[tuple[Dataset, float]], samples: int, seed: int) -> None:
        if not parts:
            raise ConfigError("data.sources is empty")
        weights = [float(weight) for _dataset, weight in parts]
        if any(weight <= 0 for weight in weights):
            raise ConfigError("every data.sources weight must be positive")
        total = sum(weights)
        self.datasets = [dataset for dataset, _weight in parts]
        self.cumulative = []
        running = 0.0
        for weight in weights:
            running += weight / total
            self.cumulative.append(running)
        self.samples = int(samples)
        self.seed = int(seed)

    def __len__(self) -> int:
        return self.samples

    def __getitem__(self, index: int) -> dict:
        rng = random.Random(self.seed * 1_000_003 + int(index))
        choice = min(bisect.bisect_left(self.cumulative, rng.random()), len(self.datasets) - 1)
        dataset = self.datasets[choice]
        return dataset[rng.randrange(len(dataset))]


def _resolve(path: str, base: Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else base / candidate


def build_source(spec: dict, config: dict, train: bool, base: Path, seed: int, samples: int) -> Dataset:
    kind = spec.get("kind")
    augment = bool(spec.get("augment", True)) and train
    if kind == "synthetic":
        return OnlineSynthDataset(config, samples, seed, augment)
    if kind == "handwriting":
        return SynthHandwritingDataset(config, samples, seed + 7, augment, spec.get("font_dir") or "")
    if kind == "labels":
        key = "labels" if train else "val_labels"
        if not spec.get(key):
            raise ConfigError(f"data.sources entry {spec.get('name') or kind} needs '{key}'")
        root = spec.get("image_root")
        return RealLinesDataset(_resolve(spec[key], base), config, augment, _resolve(root, base) if root else None)
    raise ConfigError(f"unknown data.sources kind {kind!r}; use synthetic, handwriting, or labels")


def build_mixed(config: dict, base: Path) -> tuple[Dataset, Dataset | None]:
    data = config["data"]
    seed = int(config["seed"])
    train_samples = int(data["train_samples"])
    parts = [
        (build_source(spec, config, True, base, seed + 31 * index, train_samples), float(spec.get("weight", 1.0)))
        for index, spec in enumerate(data["sources"])
    ]
    train = MixedDataset(parts, train_samples, seed)
    per_source = int(data.get("val_per_source") or 0)
    if per_source <= 0:
        return train, None
    val_seed = int(data.get("val_seed") or seed + 1_000_003)
    val_parts: list[Dataset] = []
    for index, spec in enumerate(data["sources"]):
        if spec.get("kind") == "labels" and not spec.get("val_labels"):
            continue
        source = build_source(spec, config, False, base, val_seed + 31 * index, per_source)
        if len(source) > per_source:
            stride = len(source) / per_source
            source = Subset(source, [int(position * stride) for position in range(per_source)])
        val_parts.append(source)
    return train, ConcatDataset(val_parts) if val_parts else None
