import torch

from wormocr.config import load_config
from wormocr.data.dataset import OnlineSynthDataset, collate_lines
from wormocr.data.fonts import discover_fonts
from wormocr.models.crnn import build_model, build_specs, output_width
from wormocr.vocabulary import Vocabulary


def test_synthetic_samples_are_reproducible_and_ctc_safe():
    assert len(discover_fonts("")) >= 3
    config = load_config("configs/debug.yaml")
    dataset = OnlineSynthDataset(config, samples=6, seed=11, augment=False)
    first = dataset[1]
    second = dataset[1]
    assert first["text"] == second["text"]
    assert torch.equal(first["image"], second["image"])
    specs = build_specs(config["model"]["channels"])
    assert output_width(first["width"], specs) >= len(first["text"])
    Vocabulary().encode(first["text"])
    batch = collate_lines([dataset[0], dataset[1], dataset[2]])
    assert batch["images"].shape[0] == 3
    assert int(batch["widths"].max()) == batch["images"].shape[-1]


def test_augmentation_keeps_the_same_label():
    config = load_config("configs/debug.yaml")
    clean = OnlineSynthDataset(config, samples=2, seed=3, augment=False)
    dirty = OnlineSynthDataset(config, samples=2, seed=3, augment=True)
    assert clean[0]["text"] == dirty[0]["text"]
