"""Reproducible synthetic text-line images.

The label is chosen first and is never rewritten by augmentation. Geometric
changes stay mild so the pixels still contain the same characters.
"""

from __future__ import annotations

import json
import random
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont

from wormocr.data.fonts import discover_fonts
from wormocr.data.lexicon import PANGRAM, REPEAT_WORDS, WORDS
from wormocr.exceptions import DataError
from wormocr.models.crnn import build_specs, output_width
from wormocr.preprocessing import prepare_line, resize_to_height
from wormocr.vocabulary import Vocabulary, ctc_time_needed, default_charset

CHARSET = default_charset()
LETTERS = [char for char in CHARSET if char.isalnum() or char in " .,!?'-:/$%"]


def sample_text(rng: random.Random, max_chars: int) -> str:
    limit = max(1, int(max_chars))
    kind = rng.choices(
        ["word", "words", "number", "code", "repeat", "random", "punct", "pangram"],
        weights=[30, 16, 14, 12, 10, 10, 6, 2],
    )[0]
    if kind == "word":
        text = rng.choice(WORDS)
    elif kind == "words":
        count = 2 if limit < 16 else rng.randint(2, 3)
        text = " ".join(rng.choice(WORDS) for _ in range(count))
    elif kind == "number":
        text = _number(rng)
    elif kind == "code":
        text = _code(rng)
    elif kind == "repeat":
        text = rng.choice(REPEAT_WORDS)
        if rng.random() < 0.4 and limit >= len(text) + 4:
            text = f"{text} {rng.choice(REPEAT_WORDS)}"
    elif kind == "punct":
        word = rng.choice(WORDS)
        text = rng.choice(
            [
                f"{word}, {rng.choice(WORDS)}.",
                f"{word}: {rng.choice(WORDS)}!",
                f"({word})",
                f"'{word}'",
                f"{word}?",
            ]
        )
    elif kind == "pangram":
        text = PANGRAM if rng.random() < 0.5 else PANGRAM.lower()
        if len(text) > limit:
            text = rng.choice(WORDS)
    else:
        length = rng.randint(1, min(12, limit))
        pool = [char for char in CHARSET if char != " "]
        chars = [rng.choice(pool) for _ in range(length)]
        if length > 4:
            chars[length // 2] = " "
        text = "".join(chars)
    text = _vary_case(rng, text, kind)
    text = text[:limit].strip()
    if not text:
        text = "a"
    return text


def _vary_case(rng: random.Random, text: str, kind: str) -> str:
    if kind in {"number", "code", "random"}:
        return text
    roll = rng.random()
    if roll < 0.55:
        return text.lower()
    if roll < 0.75:
        return text.upper()
    if roll < 0.92:
        return " ".join(part.capitalize() for part in text.split(" "))
    return text


def _number(rng: random.Random) -> str:
    choice = rng.randrange(4)
    if choice == 0:
        return str(rng.randint(0, 99999))
    if choice == 1:
        return f"{rng.randint(0, 999)}.{rng.randint(0, 99):02d}"
    if choice == 2:
        return f"{rng.randint(1, 28):02d}/{rng.randint(1, 12):02d}/{rng.randint(1990, 2026)}"
    return f"{rng.randint(1, 100)}%"


def _code(rng: random.Random) -> str:
    choice = rng.randrange(9)
    if choice == 0:
        return f"INV-{rng.randint(1000, 9999)}"
    if choice == 1:
        return f"${rng.randint(1, 999)}.{rng.randint(0, 99):02d}"
    if choice == 2:
        return f"ID {rng.choice('ABCDEFGHJKLMNPQRSTUVWXYZ')}{rng.randint(100, 999)}"
    if choice == 3:
        return f"${rng.randint(1, 99)},{rng.randint(0, 999):03d}.{rng.randint(0, 99):02d}"
    if choice == 4:
        return f"({rng.randint(200, 999)}) {rng.randint(200, 999)}-{rng.randint(0, 9999):04d}"
    if choice == 5:
        return f"#{rng.randint(1, 99999)}"
    if choice == 6:
        return f"{rng.choice(WORDS).capitalize()}:"
    if choice == 7:
        return f"{rng.randint(1, 999):,}".replace(",", rng.choice([",", ""])) + rng.choice(["", "%", ".00"])
    return f"555-01{rng.randint(0, 99):02d}"


def render_text(
    text: str,
    rng: random.Random,
    np_rng: np.random.Generator,
    fonts: list[str],
    font_size_min: int,
    font_size_max: int,
    augment: bool,
    corruption: str | None = None,
) -> Image.Image:
    if not fonts:
        raise DataError("no fonts are available for synthetic rendering")
    last_error = "renderer produced no image"
    for _attempt in range(10):
        try:
            image, background = _draw(text, rng, fonts, font_size_min, font_size_max)
        except OSError as exc:
            last_error = str(exc)
            continue
        if corruption:
            image = apply_corruption(image, corruption, rng, np_rng, background)
        elif augment:
            image = apply_augmentations(image, rng, np_rng, background)
        array = np.asarray(image)
        if int(np.abs(array.astype(np.int16) - int(np.median(array))).sum()) < 8:
            last_error = "rendered text had no visible ink"
            continue
        return image
    raise DataError(f"failed to render {text!r}: {last_error}")


def _draw(
    text: str,
    rng: random.Random,
    fonts: list[str],
    font_size_min: int,
    font_size_max: int,
) -> tuple[Image.Image, int]:
    font = ImageFont.truetype(rng.choice(fonts), rng.randint(font_size_min, font_size_max))
    if rng.random() < 0.82:
        background = rng.randint(214, 255)
        fill = rng.randint(0, 45)
    else:
        background = rng.randint(0, 36)
        fill = rng.randint(215, 255)
    if abs(background - fill) < 80:
        fill = 0 if background > 128 else 255
    probe = ImageDraw.Draw(Image.new("L", (1, 1)))
    box = probe.textbbox((0, 0), text, font=font)
    text_width = max(1, box[2] - box[0])
    text_height = max(1, box[3] - box[1])
    if rng.random() < 0.4:
        pad_x = rng.randint(1, 4)
        pad_y = rng.randint(1, 3)
        slack = rng.randint(0, 2)
    else:
        pad_x = rng.randint(4, 18)
        pad_y = rng.randint(3, 12)
        slack = rng.randint(0, 8)
    width = text_width + pad_x * 2 + slack
    height = text_height + pad_y * 2 + slack
    image = Image.new("L", (width, height), color=background)
    draw = ImageDraw.Draw(image)
    origin_x = pad_x + rng.randint(0, max(1, slack)) - box[0]
    origin_y = pad_y + rng.randint(0, max(1, slack)) - box[1]
    stroke = 1 if rng.random() < 0.12 else 0
    draw.text(
        (origin_x, origin_y),
        text,
        font=font,
        fill=fill,
        stroke_width=stroke,
        stroke_fill=fill,
    )
    return image, background


def apply_augmentations(
    image: Image.Image,
    rng: random.Random,
    np_rng: np.random.Generator,
    background: int,
) -> Image.Image:
    """Apply up to two mild corruptions, including scan-like degradations."""
    if rng.random() < 0.3:
        return image
    count = 1 if rng.random() < 0.6 else 2
    for _ in range(count):
        image = _one_augmentation(image, rng, np_rng, background)
    return image


def _one_augmentation(
    image: Image.Image,
    rng: random.Random,
    np_rng: np.random.Generator,
    background: int,
) -> Image.Image:
    kind = rng.choices(
        ["blur", "noise", "jpeg", "tone", "light", "lowres", "rotate", "shear", "scan", "bold", "thin", "binary"],
        weights=[8, 8, 8, 8, 5, 9, 5, 5, 12, 5, 4, 5],
    )[0]
    if kind == "scan":
        return _scan(image, rng, np_rng)
    if kind == "bold":
        return image.filter(ImageFilter.MinFilter(3) if background > 128 else ImageFilter.MaxFilter(3))
    if kind == "thin":
        if image.height < 24:
            return image
        return image.filter(ImageFilter.MaxFilter(3) if background > 128 else ImageFilter.MinFilter(3))
    if kind == "binary":
        array = np.asarray(image).astype(np.int16)
        threshold = int((array.min() + array.max()) / 2) + rng.randint(-20, 20)
        return Image.fromarray(np.where(array > threshold, 255, 0).astype(np.uint8), mode="L")
    if kind == "blur":
        return image.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.3, 0.9)))
    if kind == "noise":
        return _noise(image, np_rng, rng.uniform(3, 10))
    if kind == "jpeg":
        return _jpeg(image, rng.randint(45, 85))
    if kind == "tone":
        image = ImageEnhance.Brightness(image).enhance(rng.uniform(0.8, 1.2))
        return ImageEnhance.Contrast(image).enhance(rng.uniform(0.8, 1.25))
    if kind == "light":
        return _lighting(image, np_rng, strength=0.25)
    if kind == "lowres":
        return _down_up(image, rng)
    if kind == "rotate":
        return image.rotate(
            rng.uniform(-2.5, 2.5),
            resample=Image.Resampling.BILINEAR,
            expand=True,
            fillcolor=background,
        )
    return _shear(image, rng.uniform(-0.08, 0.08), background)


def apply_corruption(
    image: Image.Image,
    kind: str,
    rng: random.Random,
    np_rng: np.random.Generator,
    background: int,
) -> Image.Image:
    if kind == "clean":
        return image
    if kind == "blur":
        return image.filter(ImageFilter.GaussianBlur(radius=1.3))
    if kind == "noise":
        return _noise(image, np_rng, 18)
    if kind == "jpeg":
        return _jpeg(image, 28)
    if kind == "lowres":
        return _down_up(image, rng, factor=3)
    if kind == "rotate":
        return image.rotate(2.5, resample=Image.Resampling.BILINEAR, expand=True, fillcolor=background)
    if kind == "shadow":
        return _lighting(image, np_rng, strength=0.55)
    if kind == "small":
        return image
    raise DataError(f"unknown corruption {kind!r}")


def _scan(image: Image.Image, rng: random.Random, np_rng: np.random.Generator) -> Image.Image:
    """Approximate a low-resolution office scan: shrink, add sensor noise, re-enlarge."""
    width, height = image.size
    target = rng.randint(10, 18)
    if height <= target:
        return _noise(image, np_rng, rng.uniform(4, 12))
    scale = target / float(height)
    small = image.resize((max(1, int(width * scale)), target), Image.Resampling.BOX)
    small = _noise(small, np_rng, rng.uniform(4, 16))
    if rng.random() < 0.5:
        small = _jpeg(small, rng.randint(35, 75))
    return small.resize((width, height), Image.Resampling.BILINEAR)


def _noise(image: Image.Image, np_rng: np.random.Generator, sigma: float) -> Image.Image:
    array = np.asarray(image).astype(np.float32)
    array += np_rng.normal(0.0, sigma, size=array.shape)
    return Image.fromarray(np.clip(array, 0, 255).astype(np.uint8), mode="L")


def _jpeg(image: Image.Image, quality: int) -> Image.Image:
    buffer = BytesIO()
    image.convert("L").save(buffer, format="JPEG", quality=int(quality))
    buffer.seek(0)
    with Image.open(buffer) as handle:
        return handle.convert("L")


def _lighting(image: Image.Image, np_rng: np.random.Generator, strength: float = 0.35) -> Image.Image:
    array = np.asarray(image).astype(np.float32)
    height, width = array.shape
    direction = float(np_rng.random())
    ramp = np.linspace(1.0 - strength, 1.0, width if direction < 0.5 else height, dtype=np.float32)
    if direction < 0.5:
        array *= ramp[None, :]
    else:
        array *= ramp[:, None]
    return Image.fromarray(np.clip(array, 0, 255).astype(np.uint8), mode="L")


def _down_up(image: Image.Image, rng: random.Random, factor: int | None = None) -> Image.Image:
    scale = factor or rng.choice([2, 2, 3])
    width, height = image.size
    small = image.resize((max(1, width // scale), max(1, height // scale)), Image.Resampling.BILINEAR)
    return small.resize((width, height), Image.Resampling.BILINEAR)


def _shear(image: Image.Image, amount: float, background: int) -> Image.Image:
    return image.transform(
        image.size,
        Image.Transform.AFFINE,
        (1, amount, 0, 0, 1, 0),
        resample=Image.Resampling.BILINEAR,
        fillcolor=background,
    )


def fit_label_width(
    image: Image.Image,
    text: str,
    height: int,
    channels: list[int],
    max_width: int,
    width_stride: int = 2,
) -> Image.Image:
    """Resize to the model height and widen until CTC can align the label, including repeats."""
    image = resize_to_height(image.convert("L"), height)
    specs = build_specs([int(value) for value in channels], int(width_stride))
    needed = max(len(text), ctc_time_needed(text))
    width = image.width
    guard = 0
    while output_width(max(width, 8), specs) < needed and guard < 8:
        width = max(width + 2, int(width * 1.25) + len(text))
        guard += 1
        if width > max(max_width * 2, 4096):
            raise DataError(
                f"label {text!r} needs more than {width} pixels. Raise data.max_width or lower data.max_chars."
            )
    if width != image.width:
        image = image.resize((width, height), Image.Resampling.BILINEAR)
    if output_width(image.width, specs) < needed:
        raise DataError(f"label {text!r} does not fit in the image width")
    return image


def make_sample(
    index: int,
    seed: int,
    config: dict,
    fonts: list[str],
    augment: bool,
    text: str | None = None,
    corruption: str | None = None,
) -> dict:
    data_cfg = config["data"]
    model_cfg = config["model"]
    rng = random.Random(seed + index * 10007)
    np_rng = np.random.default_rng(seed + index * 10007)
    label = text if text is not None else sample_text(rng, int(data_cfg["max_chars"]))
    charset = (config.get("vocab") or {}).get("charset") or default_charset()
    try:
        Vocabulary(charset).encode(label)
    except ValueError as exc:
        raise DataError(str(exc)) from exc
    image = render_text(
        label,
        rng,
        np_rng,
        fonts,
        int(data_cfg["font_size_min"]),
        int(data_cfg["font_size_max"]),
        augment=augment,
        corruption=corruption,
    )
    if corruption == "small":
        image = image.resize((max(1, image.width // 2), max(1, image.height // 2)), Image.Resampling.BILINEAR)
    image = fit_label_width(
        image,
        label,
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
    return {
        "image": tensor,
        "text": label,
        "pad_value": pad_value,
        "width": int(tensor.shape[-1]),
    }


def generate_dataset(config: dict, samples: int, output_dir: str | Path, seed: int, augment: bool = False) -> Path:
    """Write PNG lines and a JSONL manifest. Used for inspection, not required to train."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    fonts = list(discover_fonts(config["data"].get("font_dir") or ""))
    manifest = destination / "manifest.jsonl"
    with manifest.open("w", encoding="utf-8") as handle:
        for index in range(int(samples)):
            sample = make_sample(index, seed, config, fonts, augment=augment)
            image = _tensor_to_pil(sample["image"])
            name = f"{index:06d}.png"
            image.save(destination / name)
            handle.write(
                json.dumps({"image": name, "text": sample["text"], "width": sample["width"]}, ensure_ascii=True)
                + "\n"
            )
    return manifest


def _tensor_to_pil(tensor) -> Image.Image:
    array = tensor.squeeze(0).detach().cpu().numpy()
    array = np.clip((array * 0.5 + 0.5) * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(array, mode="L")
