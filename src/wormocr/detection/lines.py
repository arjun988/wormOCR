"""Projection-profile line detection and reading order.

This is a classical layout pass for simple pages, receipts, and screenshots.
It is not a learned detector and it is not a table parser. The line recognizer
does not depend on it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw

from wormocr.preprocessing import border_color, to_grayscale


@dataclass(frozen=True)
class Box:
    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def width(self) -> int:
        return max(0, self.x1 - self.x0)

    @property
    def height(self) -> int:
        return max(0, self.y1 - self.y0)

    def as_xywh(self) -> tuple[int, int, int, int]:
        return (self.x0, self.y0, self.width, self.height)

    def pad(self, amount: int, limit_width: int, limit_height: int) -> "Box":
        return Box(
            max(0, self.x0 - amount),
            max(0, self.y0 - amount),
            min(limit_width, self.x1 + amount),
            min(limit_height, self.y1 + amount),
        )


def otsu_threshold(gray: np.ndarray) -> int:
    hist = np.bincount(gray.ravel().astype(np.uint8), minlength=256).astype(np.float64)
    total = float(gray.size)
    sum_total = float(np.dot(np.arange(256), hist))
    sum_background = 0.0
    weight_background = 0.0
    best = 0
    best_variance = -1.0
    for threshold in range(256):
        weight_background += hist[threshold]
        if weight_background == 0:
            continue
        weight_foreground = total - weight_background
        if weight_foreground == 0:
            break
        sum_background += threshold * hist[threshold]
        mean_background = sum_background / weight_background
        mean_foreground = (sum_total - sum_background) / weight_foreground
        variance = weight_background * weight_foreground * (mean_background - mean_foreground) ** 2
        if variance > best_variance:
            best_variance = variance
            best = threshold
    return best


def binarize(gray: np.ndarray) -> np.ndarray:
    values = np.asarray(gray, dtype=np.uint8)
    threshold = otsu_threshold(values)
    if float(values.mean()) < 110.0:
        return values >= threshold
    return values <= threshold


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for index, active in enumerate(mask.tolist()):
        if active and start is None:
            start = index
        elif not active and start is not None:
            runs.append((start, index))
            start = None
    if start is not None:
        runs.append((start, len(mask)))
    return runs


def _valley_split(start: int, end: int, row_fraction: np.ndarray, typical: int, depth: int = 0) -> list[tuple[int, int]]:
    height = end - start
    if depth > 4 or height < max(20, int(typical * 2.2)):
        return [(start, end)]
    middle_lo = start + height // 4
    middle_hi = end - height // 4
    if middle_hi <= middle_lo:
        return [(start, end)]
    segment = row_fraction[middle_lo:middle_hi]
    valley = int(np.argmin(segment)) + middle_lo
    peak = float(row_fraction[start:end].max())
    if float(row_fraction[valley]) > max(peak * 0.35, 0.01):
        return [(start, end)]
    return _valley_split(start, valley, row_fraction, typical, depth + 1) + _valley_split(
        valley + 1, end, row_fraction, typical, depth + 1
    )


def line_boxes(gray: np.ndarray) -> list[Box]:
    if gray.ndim != 2:
        raise ValueError("line detection expects a grayscale image")
    ink = binarize(gray)
    height, width = ink.shape
    if width == 0 or height == 0 or int(ink.sum()) < 8:
        return []
    row_fraction = ink.mean(axis=1)
    bands = _runs(row_fraction > 0.004)
    if not bands:
        return []
    heights = [end - start for start, end in bands]
    typical = sorted(heights)[len(heights) // 2]
    if len(bands) == 1:
        typical = max(8, heights[0] // 3)
    split_bands: list[tuple[int, int]] = []
    for start, end in bands:
        split_bands.extend(_valley_split(start, end, row_fraction, typical))
    column_gap = max(48, int(0.08 * width))
    boxes: list[Box] = []
    for y0, y1 in split_bands:
        if y1 - y0 < 6:
            continue
        columns = _runs(ink[y0:y1].mean(axis=0) > 0.01)
        if not columns:
            continue
        merged: list[list[int]] = [[columns[0][0], columns[0][1]]]
        for start, end in columns[1:]:
            if start - merged[-1][1] < column_gap:
                merged[-1][1] = end
            else:
                merged.append([start, end])
        for start, end in merged:
            if end - start < 6:
                continue
            boxes.append(Box(int(start), int(y0), int(end), int(y1)))
    return boxes


def vertical_split(boxes: list[Box], min_gap: int) -> tuple[list[Box], list[Box]] | None:
    if len(boxes) < 2:
        return None
    events: list[tuple[int, int]] = []
    for box in boxes:
        events.append((box.x0, 1))
        events.append((box.x1, -1))
    events.sort(key=lambda item: (item[0], item[1]))
    coverage = 0
    in_gap = False
    gap_start = 0
    gaps: list[tuple[int, int, int]] = []
    for position, delta in events:
        if delta < 0:
            coverage += delta
            if coverage == 0:
                in_gap = True
                gap_start = position
        else:
            if in_gap and coverage == 0 and position - gap_start >= min_gap:
                gaps.append((gap_start, position, position - gap_start))
            in_gap = False
            coverage += delta
    best: tuple[tuple[int, int], list[Box], list[Box]] | None = None
    for start, end, gap in gaps:
        left = [box for box in boxes if box.x1 <= start]
        right = [box for box in boxes if box.x0 >= end]
        if not left or not right or len(left) + len(right) != len(boxes):
            continue
        score = (min(len(left), len(right)), gap)
        if best is None or score > best[0]:
            best = (score, left, right)
    if best is None:
        return None
    return best[1], best[2]


def _horizontal_bands(boxes: list[Box], min_gap: int) -> list[list[Box]]:
    ordered = sorted(boxes, key=lambda box: (box.y0, box.x0))
    bands: list[list[Box]] = [[ordered[0]]]
    bottom = ordered[0].y1
    for box in ordered[1:]:
        if box.y0 - bottom >= min_gap:
            bands.append([box])
            bottom = box.y1
        else:
            bands[-1].append(box)
            bottom = max(bottom, box.y1)
    return bands


def reading_order(boxes: list[Box], min_gap_x: int = 16, min_gap_y: int = 4) -> list[Box]:
    """Read down each column when a gutter exists. Full-width lines stay in place."""
    return _reading_order(list(boxes), min_gap_x, min_gap_y)


def _reading_order(boxes: list[Box], min_gap_x: int, min_gap_y: int) -> list[Box]:
    if len(boxes) <= 1:
        return list(boxes)
    x0 = min(box.x0 for box in boxes)
    x1 = max(box.x1 for box in boxes)
    group_width = max(1, x1 - x0)
    spanning = [box for box in boxes if box.width >= 0.7 * group_width]
    body = [box for box in boxes if box not in spanning]
    split = vertical_split(body, min_gap_x) if len(body) >= 2 else None
    if split is not None:
        left, right = split
        if spanning:
            return _interleave_spanning(spanning, left, right, min_gap_x, min_gap_y)
        return _reading_order(left, min_gap_x, min_gap_y) + _reading_order(right, min_gap_x, min_gap_y)
    bands = _horizontal_bands(boxes, min_gap_y)
    if len(bands) > 1:
        ordered: list[Box] = []
        for band in bands:
            ordered.extend(_reading_order(band, min_gap_x, min_gap_y))
        return ordered
    return sorted(boxes, key=lambda box: (box.y0, box.x0))


def _interleave_spanning(
    spanning: list[Box],
    left: list[Box],
    right: list[Box],
    min_gap_x: int,
    min_gap_y: int,
) -> list[Box]:
    left_ordered = _reading_order(left, min_gap_x, min_gap_y)
    right_ordered = _reading_order(right, min_gap_x, min_gap_y)
    spans = sorted(spanning, key=lambda box: box.y0)
    output: list[Box] = []
    for index, span in enumerate(spans):
        y_lo = spans[index - 1].y1 if index else -10**9
        output.extend(box for box in left_ordered if y_lo <= box.y0 < span.y0)
        output.extend(box for box in right_ordered if y_lo <= box.y0 < span.y0)
        output.append(span)
    y_lo = spans[-1].y1
    output.extend(box for box in left_ordered if box.y0 >= y_lo)
    output.extend(box for box in right_ordered if box.y0 >= y_lo)
    return output


def projection_score(gray: np.ndarray) -> float:
    ink = binarize(gray)
    if ink.size == 0:
        return 0.0
    return float(ink.mean(axis=1).var())


def estimate_skew(gray: np.ndarray, max_angle: float = 5.0, step: float = 0.5) -> float:
    """Return the counter-clockwise degrees that straighten horizontal text."""
    sample = np.asarray(gray, dtype=np.uint8)
    if sample.shape[1] > 700:
        scale = 700 / sample.shape[1]
        resized = Image.fromarray(sample).resize(
            (700, max(1, int(round(sample.shape[0] * scale)))),
            Image.Resampling.BILINEAR,
        )
        sample = np.asarray(resized, dtype=np.uint8)

    ink = Image.fromarray(binarize(sample).astype(np.uint8) * 255)

    def score(angle: float) -> float:
        rotated = ink.rotate(angle, resample=Image.Resampling.NEAREST, expand=False, fillcolor=0)
        return float(np.asarray(rotated, dtype=np.float32).mean(axis=1).var())

    best_angle = 0.0
    best_score = score(0.0)
    angle = -float(max_angle)
    while angle <= max_angle + 1e-6:
        if abs(angle) > 1e-6:
            value = score(angle)
            if value > best_score * 1.03:
                best_score = value
                best_angle = float(angle)
        angle += step
    return best_angle


def pil_rotate_matrix(width: int, height: int, angle_deg: float) -> tuple[list[float], tuple[int, int]]:
    """Match Pillow's expand=True affine map from rotated pixels back to the original."""
    angle = -math.radians(angle_deg % 360.0)
    center = (width / 2.0, height / 2.0)
    matrix = [
        round(math.cos(angle), 15),
        round(math.sin(angle), 15),
        0.0,
        round(-math.sin(angle), 15),
        round(math.cos(angle), 15),
        0.0,
    ]

    def transform(x: float, y: float) -> tuple[float, float]:
        a, b, c, d, e, f = matrix
        return a * x + b * y + c, d * x + e * y + f

    matrix[2], matrix[5] = transform(-center[0], -center[1])
    matrix[2] += center[0]
    matrix[5] += center[1]
    xs: list[float] = []
    ys: list[float] = []
    for x, y in ((0, 0), (width, 0), (width, height), (0, height)):
        mapped_x, mapped_y = transform(x, y)
        xs.append(mapped_x)
        ys.append(mapped_y)
    new_width = math.ceil(max(xs)) - math.floor(min(xs))
    new_height = math.ceil(max(ys)) - math.floor(min(ys))
    matrix[2], matrix[5] = transform(-(new_width - width) / 2.0, -(new_height - height) / 2.0)
    return matrix, (int(new_width), int(new_height))


def map_box_to_original(box: Box, matrix: list[float], original_size: tuple[int, int]) -> Box:
    a, b, c, d, e, f = matrix
    corners = (
        (box.x0, box.y0),
        (box.x1, box.y0),
        (box.x0, box.y1),
        (box.x1, box.y1),
    )
    mapped = [(a * x + b * y + c, d * x + e * y + f) for x, y in corners]
    width, height = original_size
    x0 = max(0, int(math.floor(min(point[0] for point in mapped))))
    y0 = max(0, int(math.floor(min(point[1] for point in mapped))))
    x1 = min(width, int(math.ceil(max(point[0] for point in mapped))))
    y1 = min(height, int(math.ceil(max(point[1] for point in mapped))))
    if x1 <= x0:
        x1 = min(width, x0 + 1)
    if y1 <= y0:
        y1 = min(height, y0 + 1)
    return Box(x0, y0, x1, y1)


@dataclass
class PageLayout:
    boxes: list[Box]
    crops: list[Image.Image]
    angle: float
    layout: str
    original_size: tuple[int, int]


def segment_page(image: Image.Image, correct_orientation: bool = True) -> PageLayout:
    gray = to_grayscale(image)
    angle = estimate_skew(np.asarray(gray)) if correct_orientation else 0.0
    if abs(angle) < 0.4:
        angle = 0.0
    if angle == 0.0:
        working = gray
        matrix = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    else:
        working = gray.rotate(
            angle,
            resample=Image.Resampling.BILINEAR,
            expand=True,
            fillcolor=border_color(gray),
        )
        matrix, _expected = pil_rotate_matrix(gray.width, gray.height, angle)
    boxes = [box.pad(2, working.width, working.height) for box in line_boxes(np.asarray(working))]
    ordered = reading_order(boxes)
    crops = [working.crop((box.x0, box.y0, box.x1, box.y1)) for box in ordered]
    if angle != 0.0:
        reported = [map_box_to_original(box, matrix, gray.size) for box in ordered]
    else:
        reported = ordered
    layout = "multi-column" if vertical_split(ordered, 16) is not None else "single-column"
    return PageLayout(reported, crops, angle, layout, gray.size)


def draw_boxes(image: Image.Image, boxes: list[Box]) -> Image.Image:
    canvas = image.convert("RGB")
    draw = ImageDraw.Draw(canvas)
    for box in boxes:
        draw.rectangle((box.x0, box.y0, box.x1 - 1, box.y1 - 1), outline=(220, 30, 30))
    return canvas
