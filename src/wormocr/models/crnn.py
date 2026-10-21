"""Depthwise-separable CNN + BiGRU + CTC log-probabilities.

Width is downsampled only once (stride 2). Height 32 is reduced to a short
feature column and flattened, so narrow glyphs keep a usable time axis.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from wormocr.exceptions import ConfigError


@dataclass(frozen=True)
class Spec:
    kind: str
    out_channels: int
    kernel: tuple[int, int]
    stride: tuple[int, int]
    padding: tuple[int, int]


def build_specs(channels: list[int]) -> list[Spec]:
    if len(channels) != 4:
        raise ConfigError("the recognizer expects four channel stages")
    first, second, third, fourth = (int(value) for value in channels)
    return [
        Spec("conv", first, (3, 3), (1, 1), (1, 1)),
        Spec("pool", 0, (2, 2), (2, 2), (0, 0)),
        Spec("conv", second, (3, 3), (1, 1), (1, 1)),
        Spec("pool", 0, (2, 1), (2, 1), (0, 0)),
        Spec("conv", third, (3, 3), (1, 1), (1, 1)),
        Spec("pool", 0, (2, 1), (2, 1), (0, 0)),
        Spec("conv", fourth, (3, 3), (1, 1), (1, 1)),
        Spec("pool", 0, (2, 1), (2, 1), (0, 0)),
    ]


def _spatial(size: int, kernel: int, stride: int, padding: int) -> int:
    return (int(size) + 2 * padding - kernel) // stride + 1


def output_width(width: int, specs: list[Spec]) -> int:
    size = int(width)
    for spec in specs:
        size = _spatial(size, spec.kernel[1], spec.stride[1], spec.padding[1])
        if size <= 0:
            return 0
    return size


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel: tuple[int, int],
        stride: tuple[int, int],
        padding: tuple[int, int],
        separable: bool,
    ) -> None:
        super().__init__()
        if separable:
            self.conv: nn.Module = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    in_channels,
                    kernel,
                    stride=stride,
                    padding=padding,
                    groups=in_channels,
                    bias=False,
                ),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            )
        else:
            self.conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel,
                stride=stride,
                padding=padding,
                bias=False,
            )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=False)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(images)))


class Encoder(nn.Module):
    def __init__(self, in_channels: int, specs: list[Spec], separable: bool) -> None:
        super().__init__()
        self.specs = specs
        blocks: list[nn.Module] = []
        channels = in_channels
        for spec in specs:
            if spec.kind == "conv":
                blocks.append(
                    ConvBlock(
                        channels,
                        spec.out_channels,
                        spec.kernel,
                        spec.stride,
                        spec.padding,
                        separable=separable and channels > 1,
                    )
                )
                channels = spec.out_channels
            elif spec.kind == "pool":
                blocks.append(
                    nn.MaxPool2d(
                        kernel_size=spec.kernel,
                        stride=spec.stride,
                        padding=spec.padding,
                    )
                )
            else:
                raise ConfigError(f"unknown encoder spec {spec.kind}")
        self.blocks = nn.ModuleList(blocks)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = images
        for block in self.blocks:
            features = block(features)
        return features

    def output_width(self, width: int) -> int:
        return output_width(width, self.specs)


class CRNN(nn.Module):
    """Variable-width line recognizer. Forward returns log-probabilities (B, T, C)."""

    def __init__(
        self,
        num_classes: int,
        in_channels: int = 1,
        image_height: int = 32,
        channels: list[int] | None = None,
        separable: bool = True,
        rnn_hidden: int = 256,
        rnn_layers: int = 2,
        rnn_dropout: float = 0.1,
        proj_size: int = 256,
    ) -> None:
        super().__init__()
        if num_classes < 2:
            raise ConfigError("num_classes must include the blank and at least one character")
        stage_channels = list(channels or [32, 64, 128, 192])
        self.image_height = int(image_height)
        self.num_classes = int(num_classes)
        self.encoder = Encoder(in_channels, build_specs(stage_channels), separable)
        with torch.no_grad():
            probe = torch.zeros(1, in_channels, self.image_height, 32)
            encoded = self.encoder(probe)
        if encoded.shape[-2] < 1 or encoded.shape[-1] < 1:
            raise ConfigError(
                f"encoder collapsed a {self.image_height}px-tall input; "
                "choose a taller image_height that is a multiple of 16"
            )
        feature_height = int(encoded.shape[-2])
        feature_channels = int(encoded.shape[1])
        feature_size = feature_channels * feature_height
        self.feature_size = feature_size
        self.proj = nn.Sequential(
            nn.Linear(feature_size, proj_size),
            nn.ReLU(inplace=False),
        )
        dropout = float(rnn_dropout) if rnn_layers > 1 else 0.0
        self.rnn_dropout = nn.Dropout(dropout if dropout > 0 else 0.0)
        self.rnn = nn.GRU(
            input_size=proj_size,
            hidden_size=int(rnn_hidden),
            num_layers=int(rnn_layers),
            batch_first=True,
            bidirectional=True,
            dropout=dropout if rnn_layers > 1 else 0.0,
        )
        self.classifier = nn.Linear(int(rnn_hidden) * 2, self.num_classes)

    def output_lengths(self, widths: torch.Tensor) -> torch.Tensor:
        lengths = [self.encoder.output_width(int(width)) for width in widths.detach().cpu().tolist()]
        return torch.tensor(lengths, dtype=torch.long)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4:
            raise ValueError("images must have shape (batch, channels, height, width)")
        if images.shape[2] != self.image_height:
            raise ValueError(
                f"expected image height {self.image_height}, got {images.shape[2]}"
            )
        features = self.encoder(images)
        batch, channels, height, time = features.shape
        sequence = features.permute(0, 3, 1, 2).contiguous().view(batch, time, channels * height)
        sequence = self.rnn_dropout(self.proj(sequence))
        sequence, _hidden = self.rnn(sequence)
        return self.classifier(sequence).float().log_softmax(dim=-1)


def build_model(config: dict, num_classes: int) -> CRNN:
    model_cfg = config["model"]
    return CRNN(
        num_classes=num_classes,
        in_channels=int(model_cfg.get("in_channels", 1)),
        image_height=int(model_cfg["image_height"]),
        channels=[int(value) for value in model_cfg["channels"]],
        separable=bool(model_cfg["separable"]),
        rnn_hidden=int(model_cfg["rnn_hidden"]),
        rnn_layers=int(model_cfg["rnn_layers"]),
        rnn_dropout=float(model_cfg.get("rnn_dropout", 0.0)),
        proj_size=int(model_cfg.get("proj_size", model_cfg["rnn_hidden"])),
    )


def count_parameters(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def state_dict_num_bytes(module: nn.Module) -> int:
    return sum(parameter.numel() * parameter.element_size() for parameter in module.parameters())
