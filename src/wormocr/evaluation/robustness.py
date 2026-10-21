"""Controlled corruptions for robustness measurements."""

from __future__ import annotations

import random

from PIL import Image

from wormocr.data.fonts import discover_fonts
from wormocr.data.synthetic import _tensor_to_pil, make_sample, sample_text
from wormocr.evaluation.metrics import MetricTotals
from wormocr.inference.engine import LineEngine

CORRUPTIONS = ("clean", "blur", "noise", "jpeg", "lowres", "rotate", "shadow", "small")


def measure_robustness(engine: LineEngine, samples: int, seed: int) -> dict[str, dict]:
    config = engine.config
    fonts = list(discover_fonts(config["data"].get("font_dir") or ""))
    texts = [
        sample_text(random.Random(seed + index * 17), int(config["data"]["max_chars"]))
        for index in range(samples)
    ]
    report: dict[str, dict] = {}
    for kind in CORRUPTIONS:
        images: list[Image.Image] = []
        for index, text in enumerate(texts):
            sample = make_sample(
                index,
                seed + 99,
                config,
                fonts,
                augment=False,
                text=text,
                corruption=kind,
            )
            images.append(_tensor_to_pil(sample["image"]))
        hypotheses = engine.recognize(images)
        totals = MetricTotals()
        for text, hypothesis in zip(texts, hypotheses):
            totals.update(text, str(hypothesis["text"]))
        report[kind] = totals.to_dict()
    return report
