"""Synthetic handwritten lines from open-licensed handwriting fonts.

Fonts are downloaded by scripts/download_fonts.py (SIL OFL / Apache 2.0) and
are never committed. Each sample is a pure function of (seed, index), like the
printed generator. Font regularity is broken up with per-word or per-glyph
jitter, slant, a wavy baseline, elastic warping, stroke-width and pressure
changes, paper textures (ruled lines, grids, stains), and camera effects.
"""

from __future__ import annotations

import functools
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from wormocr.config import repo_root
from wormocr.data.lexicon import WORDS
from wormocr.data.lexicon_medical import DURATIONS, FORMS, FREQUENCIES, GENERICS, NOTES, UNITS
from wormocr.data.synthetic import _down_up, _jpeg, _lighting, _noise, _number
from wormocr.exceptions import DataError
from wormocr.vocabulary import default_charset

DEFAULT_FONT_DIR = "artifacts/fonts/handwriting"
LETTERS = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
CHARSET = set(default_charset())
FIRST_NAMES = "james mary john linda robert sarah michael priya arjun fatima chen maria david anna ali omar sofia luca emma noah".split()
SURNAMES = "smith garcia shukla khan nguyen patel muller rossi kim silva brown wilson ahmed sato lopez martin".split()


@functools.lru_cache(maxsize=4)
def handwriting_fonts(font_dir: str = "") -> tuple[tuple[str, frozenset], ...]:
    """(path, characters it can draw) for every usable handwriting font."""
    from fontTools.ttLib import TTFont

    root = Path(font_dir or DEFAULT_FONT_DIR)
    if not root.is_absolute():
        root = repo_root() / root
    found: list[tuple[str, frozenset]] = []
    for path in sorted(root.rglob("*.ttf")):
        try:
            cmap = TTFont(str(path), lazy=True).getBestCmap() or {}
            ImageFont.truetype(str(path), 24)
        except Exception:
            continue
        coverage = frozenset(chr(code) for code in cmap if chr(code) in CHARSET)
        if LETTERS <= coverage and len(coverage) >= 62:
            found.append((str(path), coverage))
    if not found:
        raise DataError(
            f"no handwriting fonts found in {root}. Run: python scripts/download_fonts.py"
        )
    return tuple(found)


def sample_hw_text(rng: random.Random, max_chars: int) -> str:
    kind = rng.choices(
        ["phrase", "sentence", "name", "number", "medical", "note", "word", "mixed"],
        weights=[22, 20, 10, 10, 16, 8, 10, 4],
    )[0]
    if kind == "phrase":
        text = " ".join(rng.choice(WORDS) for _ in range(rng.randint(2, 4)))
    elif kind == "sentence":
        words = [rng.choice(WORDS) for _ in range(rng.randint(4, 8))]
        words[0] = words[0].capitalize()
        text = " ".join(words) + rng.choice([".", ".", ",", "", "!", "?", ":"])
    elif kind == "name":
        first, last = rng.choice(FIRST_NAMES), rng.choice(SURNAMES)
        style = rng.random()
        if style < 0.35:
            text = f"{first.upper()} {last.upper()}"
        elif style < 0.7:
            text = f"{first.capitalize()} {last.capitalize()}"
        else:
            text = f"{rng.choice(['Dr.', 'Mr.', 'Ms.', 'Mrs.'])} {last.capitalize()}"
    elif kind == "number":
        text = rng.choice(
            [
                _number(rng),
                f"{rng.randint(1, 12)}:{rng.randint(0, 59):02d} {rng.choice(['am', 'pm', 'AM', 'PM'])}",
                f"Rs. {rng.randint(10, 9999)}",
                f"${rng.randint(1, 999)}.{rng.randint(0, 99):02d}",
                f"+91 {rng.randint(70000, 99999)} {rng.randint(10000, 99999)}",
                f"{rng.randint(1, 28)}/{rng.randint(1, 12)}/{rng.randint(20, 26)}",
            ]
        )
    elif kind == "medical":
        drug = rng.choice(GENERICS)
        drug = drug.capitalize() if rng.random() < 0.6 else (drug.upper() if rng.random() < 0.5 else drug)
        dose = f"{rng.choice([5, 10, 20, 25, 40, 50, 100, 250, 500, 650, 1])}{rng.choice(['', ' '])}{rng.choice(UNITS)}"
        parts = [rng.choice(FORMS), drug, dose]
        if rng.random() < 0.7:
            parts.append(rng.choice(FREQUENCIES))
        if rng.random() < 0.4:
            parts.append(rng.choice(DURATIONS))
        text = " ".join(parts[: rng.randint(2, len(parts))])
    elif kind == "note":
        text = rng.choice(NOTES)
        if rng.random() < 0.5:
            text += rng.choice([f" {rng.randint(90, 180)}/{rng.randint(50, 110)}", f" {rng.randint(1, 30)} days", ": " + rng.choice(WORDS)])
    elif kind == "word":
        text = rng.choice(WORDS + GENERICS)
        roll = rng.random()
        text = text.capitalize() if roll < 0.35 else (text.upper() if roll < 0.5 else text)
    else:
        text = f"{rng.choice(WORDS)} {rng.randint(1, 99)} {rng.choice(WORDS)}"
    text = text[: max(1, int(max_chars))].strip()
    return text or "a"


def _pick_font(text: str, rng: random.Random, fonts: tuple[tuple[str, frozenset], ...]) -> str | None:
    needed = set(text) - {" "}
    for _ in range(12):
        path, coverage = rng.choice(fonts)
        if needed <= coverage:
            return path
    usable = [path for path, coverage in fonts if needed <= coverage]
    return rng.choice(usable) if usable else None


def _paste_max(canvas: np.ndarray, piece: np.ndarray, x: int, y: int) -> None:
    height, width = piece.shape
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(canvas.shape[1], x + width), min(canvas.shape[0], y + height)
    if x1 <= x0 or y1 <= y0:
        return
    region = canvas[y0:y1, x0:x1]
    np.maximum(region, piece[y0 - y : y1 - y, x0 - x : x1 - x], out=region)


def _render_piece(text: str, font: ImageFont.FreeTypeFont, angle: float) -> np.ndarray:
    left, top, right, bottom = font.getbbox(text)
    pad = 4
    image = Image.new("L", (right - left + 2 * pad, bottom - top + 2 * pad), 0)
    ImageDraw.Draw(image).text((pad - left, pad - top), text, font=font, fill=255)
    if abs(angle) > 0.2:
        image = image.rotate(angle, resample=Image.Resampling.BICUBIC, expand=True, fillcolor=0)
    return np.asarray(image, dtype=np.uint8), top - pad


def _ink_mask(text: str, font_path: str, size: int, rng: random.Random) -> np.ndarray:
    font = ImageFont.truetype(font_path, size)
    ascent, descent = font.getmetrics()
    space = max(font.getlength(" "), 0.28 * size)
    glyph_mode = rng.random() < 0.35
    canvas = np.zeros((int((ascent + descent) * 2.2) + 8, int(font.getlength(text) * 1.6 + size * 2 + 16)), np.uint8)
    baseline = int(ascent * 1.4) + 4
    x = size // 3 + 2
    drift = 0.0
    drift_step = rng.uniform(-0.04, 0.04) * size
    for word_index, word in enumerate(text.split(" ")):
        if word_index:
            x += int(space * rng.uniform(0.85, 1.8))
        if not word:
            continue
        drift += drift_step * rng.uniform(0.0, 1.0)
        if glyph_mode:
            for char in word:
                scale = rng.uniform(0.88, 1.12)
                glyph_font = ImageFont.truetype(font_path, max(8, int(size * scale)))
                piece, top = _render_piece(char, glyph_font, rng.uniform(-6, 6))
                offset = int(drift + rng.uniform(-0.06, 0.06) * size)
                _paste_max(canvas, piece, x, baseline - glyph_font.getmetrics()[0] + top + offset)
                x += int(glyph_font.getlength(char) * rng.uniform(0.82, 1.08))
        else:
            word_font = ImageFont.truetype(font_path, max(8, int(size * rng.uniform(0.9, 1.1))))
            piece, top = _render_piece(word, word_font, rng.uniform(-3.5, 3.5))
            offset = int(drift + rng.uniform(-0.07, 0.07) * size)
            _paste_max(canvas, piece, x, baseline - word_font.getmetrics()[0] + top + offset)
            inked = np.where(piece.max(axis=0) > 40)[0]
            ink_right = int(inked[-1]) + 1 if len(inked) else 0
            x += max(int(word_font.getlength(word)), ink_right)
    return canvas


def _smooth_field(np_rng: np.random.Generator, shape: tuple[int, int], cells: int, amplitude: float) -> np.ndarray:
    small = np_rng.normal(0.0, 1.0, size=(max(2, shape[0] // cells + 2), max(2, shape[1] // cells + 2))).astype(np.float32)
    field = Image.fromarray(small, mode="F").resize((shape[1], shape[0]), Image.Resampling.BICUBIC)
    return np.asarray(field, dtype=np.float32) * amplitude


def _remap(array: np.ndarray, dx: np.ndarray, dy: np.ndarray) -> np.ndarray:
    height, width = array.shape
    ys, xs = np.mgrid[0:height, 0:width].astype(np.float32)
    sx = np.clip(xs + dx, 0, width - 1.001)
    sy = np.clip(ys + dy, 0, height - 1.001)
    x0, y0 = sx.astype(np.int32), sy.astype(np.int32)
    fx, fy = sx - x0, sy - y0
    source = array.astype(np.float32)
    top = source[y0, x0] * (1 - fx) + source[y0, x0 + 1] * fx
    bottom = source[y0 + 1, x0] * (1 - fx) + source[y0 + 1, x0 + 1] * fx
    return top * (1 - fy) + bottom * fy


def _geometry(mask: np.ndarray, size: int, rng: random.Random, np_rng: np.random.Generator) -> np.ndarray:
    height, width = mask.shape
    ys, xs = np.mgrid[0:height, 0:width].astype(np.float32)
    slant = rng.uniform(-0.38, 0.32)
    dx = slant * (ys - height / 2)
    wave = rng.uniform(0.0, 0.09) * size
    period = rng.uniform(1.5, 5.0) * size
    dy = wave * np.sin(xs / period * 2 * np.pi + rng.uniform(0, 6.3))
    curve = rng.uniform(-0.05, 0.05) * size
    dy += curve * ((xs - width / 2) / max(1.0, width / 2)) ** 2
    if rng.random() < 0.7:
        amplitude = rng.uniform(0.4, 1.6) * size / 32
        dx += _smooth_field(np_rng, mask.shape, max(6, size // 3), amplitude)
        dy += _smooth_field(np_rng, mask.shape, max(6, size // 3), amplitude)
    return np.clip(_remap(mask, dx, dy), 0, 255)


def _paper(
    shape: tuple[int, int], size: int, rng: random.Random, np_rng: np.random.Generator
) -> tuple[np.ndarray, float]:
    height, width = shape
    base = rng.uniform(175, 252)
    paper = np.full(shape, base, np.float32)
    paper += _smooth_field(np_rng, shape, max(8, size), rng.uniform(1, 9))
    paper += np_rng.normal(0, rng.uniform(0.5, 4), size=shape).astype(np.float32)
    style = rng.random()
    line_tone = base - rng.uniform(25, 80)
    if style < 0.35:
        spacing = rng.uniform(1.0, 1.6) * size
        start = rng.uniform(0, spacing)
        for y in np.arange(start, height, spacing):
            row = int(y)
            paper[row : row + rng.choice([1, 1, 2])] = line_tone
    elif style < 0.45:
        spacing = rng.uniform(0.5, 1.0) * size
        for y in np.arange(rng.uniform(0, spacing), height, spacing):
            paper[int(y)] = line_tone
        for x in np.arange(rng.uniform(0, spacing), width, spacing):
            paper[:, int(x)] = line_tone
    if rng.random() < 0.08:
        column = int(rng.uniform(0.02, 0.15) * width)
        paper[:, column : column + 2] = base - rng.uniform(40, 90)
    if rng.random() < 0.15:
        cy, cx = rng.uniform(0, height), rng.uniform(0, width)
        radius = rng.uniform(0.5, 2.5) * size
        ys, xs = np.mgrid[0:height, 0:width]
        blob = np.exp(-(((ys - cy) ** 2 + (xs - cx) ** 2) / (2 * radius**2)))
        paper -= blob.astype(np.float32) * rng.uniform(15, 60)
    return paper, base


def render_handwriting(
    text: str,
    rng: random.Random,
    np_rng: np.random.Generator,
    fonts: tuple[tuple[str, frozenset], ...],
    size_min: int = 26,
    size_max: int = 56,
    augment: bool = True,
) -> Image.Image:
    font_path = _pick_font(text, rng, fonts)
    if font_path is None:
        raise DataError(f"no handwriting font can draw {text!r}")
    size = rng.randint(size_min, size_max)
    mask = _ink_mask(text, font_path, size, rng)
    if augment:
        mask = _geometry(mask, size, rng, np_rng)
        mask_image = Image.fromarray(mask.astype(np.uint8), mode="L")
        roll = rng.random()
        if roll < 0.25:
            mask_image = mask_image.filter(ImageFilter.MaxFilter(3))
        elif roll < 0.38 and size > 34:
            thinned = mask_image.filter(ImageFilter.MinFilter(3))
            before = int((np.asarray(mask_image) > 128).sum())
            if (np.asarray(thinned) > 128).sum() >= 0.6 * before:
                mask_image = thinned
        mask = np.asarray(mask_image, dtype=np.float32)
    rows = np.where(mask.max(axis=1) > 40)[0]
    cols = np.where(mask.max(axis=0) > 40)[0]
    if len(rows) == 0 or len(cols) == 0:
        raise DataError(f"rendered handwriting for {text!r} has no ink")
    pad_y = int(rng.uniform(0.08, 0.45) * size)
    pad_x = int(rng.uniform(0.1, 0.6) * size)
    top, bottom = max(0, rows[0] - pad_y), min(mask.shape[0], rows[-1] + pad_y + 1)
    left, right = max(0, cols[0] - pad_x), min(mask.shape[1], cols[-1] + pad_x + 1)
    alpha = mask[top:bottom, left:right] / 255.0
    if augment:
        pressure = 1.0 - np.clip(_smooth_field(np_rng, alpha.shape, max(6, size // 2), rng.uniform(0.0, 0.25)), -0.15, 0.35)
        alpha = np.clip(alpha * pressure * rng.uniform(0.8, 1.0), 0.0, 1.0)
        paper, base = _paper(alpha.shape, size, rng, np_rng)
        ink = rng.uniform(0, min(95.0, base - 90.0))
    else:
        paper, ink = np.full(alpha.shape, 245.0, np.float32), 20.0
    page = paper * (1 - alpha) + ink * alpha
    image = Image.fromarray(np.clip(page, 0, 255).astype(np.uint8), mode="L")
    if augment:
        for kind in rng.sample(["light", "blur", "noise", "jpeg", "lowres"], k=rng.randint(0, 3)):
            if kind == "light":
                image = _lighting(image, np_rng, strength=rng.uniform(0.1, 0.35))
            elif kind == "blur":
                image = image.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.3, 1.2)))
            elif kind == "noise":
                image = _noise(image, np_rng, rng.uniform(2, 12))
            elif kind == "jpeg":
                image = _jpeg(image, rng.randint(30, 85))
            elif image.height > 40:
                image = _down_up(image, rng)
    return image
