"""Load a checkpoint and recognize text lines."""

from __future__ import annotations

import warnings
from pathlib import Path

import torch
from PIL import Image

from wormocr.checkpoints import load_checkpoint
from wormocr.inference.decode import ctc_beam_search, greedy_confidence, greedy_decode
from wormocr.models.crnn import build_model
from wormocr.preprocessing import prepare_line
from wormocr.runtime import resolve_device
from wormocr.vocabulary import Vocabulary


class LineEngine:
    def __init__(
        self,
        model,
        vocabulary: Vocabulary,
        device: torch.device,
        config: dict,
        maturity: str,
        beam_width: int = 0,
        max_width: int = 4096,
        batch_size: int = 8,
    ) -> None:
        self.model = model
        self.vocabulary = vocabulary
        self.device = device
        self.config = config
        self.maturity = maturity
        self.beam_width = int(beam_width)
        self.max_width = int(max_width)
        self.batch_size = max(1, int(batch_size))
        self.image_height = int(config["model"]["image_height"])
        self.model.to(device)
        self.model.eval()

    @classmethod
    def load(
        cls,
        path: str | Path,
        device: str | None = None,
        beam_width: int = 0,
        batch_size: int = 8,
    ) -> "LineEngine":
        torch_device = resolve_device(device)
        payload = load_checkpoint(path, map_location=torch_device)
        vocabulary = Vocabulary(payload["charset"])
        model = build_model(payload["config"], vocabulary.size)
        model.load_state_dict(payload["model_state"])
        maturity = str(payload.get("maturity") or "untrained")
        if maturity in {"untrained", "smoke-tested"}:
            warnings.warn(
                f"This checkpoint is {maturity} and is not a quality OCR model.",
                UserWarning,
                stacklevel=2,
            )
        return cls(
            model=model,
            vocabulary=vocabulary,
            device=torch_device,
            config=payload["config"],
            maturity=maturity,
            beam_width=beam_width,
            max_width=int(payload["config"].get("data", {}).get("max_width") or 1024) * 2,
            batch_size=batch_size,
        )

    def recognize(self, images: list[Image.Image]) -> list[dict[str, float | str]]:
        if not images:
            return []
        prepared: list[tuple[torch.Tensor, float, int]] = []
        for image in images:
            tensor, pad_value = prepare_line(
                image,
                height=self.image_height,
                align=2,
                max_width=self.max_width,
            )
            prepared.append((tensor, pad_value, int(tensor.shape[-1])))
        order = sorted(range(len(prepared)), key=lambda index: prepared[index][2])
        outputs: list[dict[str, float | str] | None] = [None] * len(prepared)
        for start in range(0, len(order), self.batch_size):
            chunk = order[start : start + self.batch_size]
            widths = [prepared[index][2] for index in chunk]
            max_width = max(widths)
            batch = prepared[chunk[0]][0].new_empty(len(chunk), 1, self.image_height, max_width)
            for row, index in enumerate(chunk):
                tensor, pad_value, width = prepared[index]
                batch[row, :, :, :width] = tensor
                if width < max_width:
                    batch[row, :, :, width:] = pad_value
            with torch.no_grad():
                log_probs = self.model(batch.to(self.device))
            length_tensor = self.model.output_lengths(torch.tensor(widths, dtype=torch.long))
            length_tensor = torch.minimum(length_tensor, torch.tensor(log_probs.shape[1]))
            if self.beam_width > 0:
                sequences = [
                    ctc_beam_search(
                        log_probs[row, : int(length_tensor[row])].cpu(),
                        self.beam_width,
                        self.vocabulary.blank,
                    )
                    for row in range(len(chunk))
                ]
            else:
                sequences = greedy_decode(log_probs, length_tensor, self.vocabulary.blank)
            for row, index in enumerate(chunk):
                outputs[index] = {
                    "text": self.vocabulary.decode(sequences[row]),
                    "confidence": greedy_confidence(
                        log_probs[row].cpu(),
                        int(length_tensor[row]),
                        self.vocabulary.blank,
                    ),
                }
        return [item for item in outputs if item is not None]
