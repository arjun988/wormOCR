"""Image loading and line normalization.

The recognizer consumes grayscale tensors of a fixed height and variable width,
normalized to [-1, 1]. Padding on the right uses the image border so a short
line in a batch is not given a false edge.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError

from wormocr.exceptions import RecognitionError


def to_grayscale(image: Image.Image) -> Image.Image:
    if image.mode == "RGBA":
        background = Image.new("RGBA", image.size, (255, 255, 255, 255))
        image = Image.alpha_composite(background, image)
        return image.convert("L")
    if image.mode != "L":
        return image.convert("L")
    return image


def border_color(image: Image.Image) -> int:
    array = np.asarray(image.convert("L"))
    if array.size == 0:
        return 255
    border = np.concatenate(
        [array[:, 0], array[:, -1], array[0, :], array[-1, :]]
    )
    return int(np.median(border))


def resize_to_height(image: Image.Image, height: int) -> Image.Image:
    if height < 8:
        raise ValueError("image height must be at least 8")
    width, source_height = image.size
    if width < 1 or source_height < 1:
        raise ValueError("image is empty")
    if source_height == height:
        return image
    new_width = max(1, int(round(width * (height / float(source_height)))))
    return image.resize((new_width, height), Image.Resampling.BILINEAR)


def _normalize_pixel(value: float) -> float:
    return (value / 255.0 - 0.5) / 0.5


def prepare_line(
    image: Image.Image,
    height: int = 32,
    align: int = 2,
    max_width: int = 2048,
    min_width: int = 8,
) -> tuple[torch.Tensor, float]:
    """Return a float tensor of shape (1, height, width) and its pad value."""
    if align < 1:
        raise ValueError("align must be positive")
    gray = to_grayscale(image)
    if gray.width < 1 or gray.height < 1:
        raise ValueError("image is empty")
    resized = resize_to_height(gray, height)
    if resized.width > max_width:
        resized = resized.resize((max_width, height), Image.Resampling.BILINEAR)
    color = border_color(resized)
    if resized.width < min_width:
        canvas = Image.new("L", (min_width, height), color=color)
        canvas.paste(resized, (0, 0))
        resized = canvas
    if resized.width % align != 0:
        padded_width = resized.width + (align - resized.width % align)
        canvas = Image.new("L", (padded_width, height), color=color)
        canvas.paste(resized, (0, 0))
        resized = canvas
        color = border_color(resized)
    array = np.asarray(resized, dtype=np.float32)
    tensor = torch.from_numpy(array).unsqueeze(0)
    tensor = (tensor / 255.0 - 0.5) / 0.5
    return tensor, float(_normalize_pixel(color))


def image_from_array(array: np.ndarray) -> Image.Image:
    if array.ndim not in (2, 3):
        raise RecognitionError("expected an HxW or HxWx3 image array")
    values = array
    if np.issubdtype(values.dtype, np.floating):
        values = np.clip(values * 255.0, 0, 255).astype(np.uint8)
    else:
        values = values.astype(np.uint8)
    if values.ndim == 2:
        return Image.fromarray(values, mode="L")
    channels = values.shape[2]
    if channels == 3:
        return Image.fromarray(values, mode="RGB")
    if channels == 4:
        return Image.fromarray(values, mode="RGBA")
    raise RecognitionError(f"unsupported channel count: {channels}")


def load_image(source: str | Path) -> Image.Image:
    path = Path(source)
    if not path.exists():
        raise RecognitionError(f"image not found: {path}")
    if path.suffix.lower() == ".pdf":
        raise RecognitionError("use the document loader for PDF files")
    try:
        with Image.open(path) as handle:
            return handle.convert("RGBA") if handle.mode == "RGBA" else handle.convert("RGB")
    except UnidentifiedImageError as exc:
        raise RecognitionError(f"cannot read image: {path}") from exc
    except OSError as exc:
        raise RecognitionError(f"cannot read image: {path}") from exc


def coerce_image(source: str | Path | Image.Image | np.ndarray) -> Image.Image:
    if isinstance(source, Image.Image):
        return source
    if isinstance(source, np.ndarray):
        return image_from_array(source)
    if isinstance(source, (str, Path)):
        return load_image(source)
    raise RecognitionError(
        "image must be a file path, a PIL image, or a numpy array"
    )


def looks_like_line(image: Image.Image, max_height: int = 96) -> bool:
    width, height = image.size
    return height <= max_height and width >= height
