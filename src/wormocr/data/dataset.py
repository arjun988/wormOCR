"""Training datasets: on-the-fly synthetic lines, manifests, and label files."""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset

from wormocr.data.fonts import discover_fonts
from wormocr.data.synthetic import make_sample
from wormocr.exceptions import DataError
from wormocr.preprocessing import prepare_line
from wormocr.vocabulary import Vocabulary


class OnlineSynthDataset(Dataset):
    """Deterministic synthetic lines. Item i depends only on the seed and index."""

    def __init__(self, config: dict, samples: int, seed: int, augment: bool) -> None:
        self.config = config
        self.samples = int(samples)
        self.seed = int(seed)
        self.augment = bool(augment)
        self.fonts = list(discover_fonts(config["data"].get("font_dir") or ""))

    def __len__(self) -> int:
        return self.samples

    def __getitem__(self, index: int) -> dict:
        return make_sample(int(index), self.seed, self.config, self.fonts, self.augment)


class LabelDataset(Dataset):
    """Real or exported lines described by `relative-path<TAB>text`."""

    def __init__(self, labels_path: str | Path, image_root: str | Path, config: dict, vocabulary: Vocabulary) -> None:
        self.root = Path(image_root)
        self.config = config
        self.vocabulary = vocabulary
        self.records = _read_labels(labels_path)
        if not self.records:
            raise DataError(f"no labels found in {labels_path}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        relative, text = self.records[index]
        path = self.root / relative
        if not path.exists():
            raise DataError(f"missing image for label {relative!r}: {path}")
        try:
            self.vocabulary.encode(text)
        except ValueError as exc:
            raise DataError(f"{relative}: {exc}") from exc
        if not text:
            raise DataError(f"{relative} has an empty label")
        with Image.open(path) as handle:
            image = handle.convert("RGB")
        tensor, pad_value = prepare_line(
            image,
            height=int(self.config["model"]["image_height"]),
            align=2,
            max_width=int(self.config["data"]["max_width"]),
        )
        return {
            "image": tensor,
            "text": text,
            "pad_value": pad_value,
            "width": int(tensor.shape[-1]),
            "path": str(path),
        }


def _read_labels(path: str | Path) -> list[tuple[str, str]]:
    records: list[tuple[str, str]] = []
    source = Path(path)
    if not source.exists():
        raise DataError(f"labels file not found: {source}")
    for line_number, raw in enumerate(source.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if "\t" not in stripped:
            raise DataError(f"{source}:{line_number} must be image<TAB>text")
        relative, text = stripped.split("\t", 1)
        records.append((relative.strip(), text))
    return records


def collate_lines(samples: list[dict]) -> dict:
    if not samples:
        raise DataError("empty batch")
    height = samples[0]["image"].shape[-2]
    widths = [int(sample["image"].shape[-1]) for sample in samples]
    max_width = max(widths)
    images = samples[0]["image"].new_empty(len(samples), 1, height, max_width)
    for index, sample in enumerate(samples):
        image = sample["image"]
        width = image.shape[-1]
        images[index, :, :, :width] = image
        if width < max_width:
            images[index, :, :, width:] = float(sample["pad_value"])
    return {
        "images": images,
        "texts": [sample["text"] for sample in samples],
        "widths": torch.tensor(widths, dtype=torch.long),
    }
