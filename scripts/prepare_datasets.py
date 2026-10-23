"""Turn downloaded datasets into line/word crops plus `relative<TAB>text` label files.

Output: artifacts/datasets/prepared/<name>/
    images/...            grayscale crops, height 64 px (aspect kept)
    train.tsv val.tsv test.tsv
    STATS.json            kept / dropped counts and the reasons

Splits are fixed (by source split or by a hash of the page/file id), so the
same command always gives the same split.

    python scripts/prepare_datasets.py rxhand iam_test gnhk hw_names cord bentham
"""

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import io
import json
import math
import re
import unicodedata
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from wormocr.vocabulary import ctc_time_needed, default_charset

ROOT = Path(__file__).resolve().parents[1] / "artifacts" / "datasets"
RAW = ROOT / "raw"
OUT = ROOT / "prepared"
STORE_HEIGHT = 64
FRAME_STRIDE = 4
REFERENCE_HEIGHT = 32
CHARSET = set(default_charset())

REPLACEMENTS = {
    "\u2018": "'", "\u2019": "'", "\u201a": "'", "\u201b": "'", "\u2032": "'", "\u00b4": "'", "`": "'",
    "\u201c": '"', "\u201d": '"', "\u201e": '"', "\u2033": '"', "\u00ab": '"', "\u00bb": '"',
    "\u2013": "-", "\u2014": "-", "\u2012": "-", "\u2212": "-", "\u2010": "-", "\u2011": "-",
    "\u2026": "...", "\u00d7": "x", "\u00a0": " ", "\u2009": " ", "\u200b": "", "\u00b7": ".",
}


# ---------------------------------------------------------------- shared helpers


def normalize_text(text: str) -> tuple[str | None, str]:
    """Map lookalike characters into the charset. Returns (text, reason-if-dropped)."""
    text = "".join(REPLACEMENTS.get(char, char) for char in text)
    text = unicodedata.normalize("NFKC", text)
    text = "".join(REPLACEMENTS.get(char, char) for char in text)
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return None, "empty"
    outside = sorted({char for char in text if char not in CHARSET})
    if outside:
        return None, "charset:" + "".join(outside[:3])
    return text, ""


def split_of(key: str, val: float = 0.05, test: float = 0.0) -> str:
    bucket = int(hashlib.sha1(key.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
    if bucket < test:
        return "test"
    if bucket < test + val:
        return "val"
    return "train"


def finish_crop(image: Image.Image, text: str) -> Image.Image | None:
    """Grayscale, store at STORE_HEIGHT, and widen if CTC could not fit the label."""
    gray = ImageOps.grayscale(image)
    if gray.width < 4 or gray.height < 6:
        return None
    width = max(4, round(gray.width * STORE_HEIGHT / gray.height))
    needed = ctc_time_needed(text) + 2
    frames = width * REFERENCE_HEIGHT / STORE_HEIGHT / FRAME_STRIDE
    if frames < needed:
        if frames < needed / 1.6:
            return None
        width = math.ceil(needed * FRAME_STRIDE * STORE_HEIGHT / REFERENCE_HEIGHT)
    return gray.resize((width, STORE_HEIGHT), Image.Resampling.LANCZOS)


def quad_crop(image: Image.Image, points: np.ndarray, pad_x: float = 0.12, pad_y: float = 0.18) -> Image.Image:
    """Straighten the region around `points` (N x 2) along its main direction."""
    points = np.asarray(points, dtype=np.float64)
    center = points.mean(axis=0)
    centered = points - center
    if len(points) >= 4:
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        direction = vt[0] if vt[0][0] >= 0 else -vt[0]
    else:
        direction = np.array([1.0, 0.0])
    angle = math.atan2(direction[1], direction[0])
    if abs(angle) > math.radians(40):
        angle = 0.0
    cos, sin = math.cos(angle), math.sin(angle)
    u = centered @ np.array([cos, sin])
    v = centered @ np.array([-sin, cos])
    height = v.max() - v.min()
    u0, u1 = u.min() - pad_x * height, u.max() + pad_x * height
    v0, v1 = v.min() - pad_y * height, v.max() + pad_y * height

    def back(uu: float, vv: float) -> tuple[float, float]:
        return (center[0] + uu * cos - vv * sin, center[1] + uu * sin + vv * cos)

    quad = [*back(u0, v0), *back(u0, v1), *back(u1, v1), *back(u1, v0)]
    size = (max(4, int(round(u1 - u0))), max(4, int(round(v1 - v0))))
    return image.transform(size, Image.Transform.QUAD, quad, resample=Image.Resampling.BICUBIC, fillcolor=(255, 255, 255))


class Writer:
    def __init__(self, name: str) -> None:
        self.folder = OUT / name
        (self.folder / "images").mkdir(parents=True, exist_ok=True)
        self.rows: dict[str, list[str]] = collections.defaultdict(list)
        self.dropped: collections.Counter = collections.Counter()

    def add(self, split: str, key: str, image: Image.Image | None, raw_text: str) -> None:
        text, reason = normalize_text(raw_text)
        if text is None:
            self.dropped[reason.split(":")[0]] += 1
            return
        if not re.search(r"[A-Za-z0-9]", text):
            self.dropped["no_letters_or_digits"] += 1
            return
        crop = finish_crop(image, text) if image is not None else None
        if crop is None:
            self.dropped["image_too_small_or_narrow"] += 1
            return
        relative = f"images/{key}.png"
        crop.save(self.folder / relative, optimize=False, compress_level=3)
        self.rows[split].append(f"{relative}\t{text}")

    def close(self, extra: dict | None = None) -> dict:
        stats = {"kept": {split: len(rows) for split, rows in sorted(self.rows.items())}, "dropped": dict(self.dropped)}
        if extra:
            stats.update(extra)
        for split, rows in self.rows.items():
            (self.folder / f"{split}.tsv").write_text("\n".join(rows) + "\n", encoding="utf-8")
        (self.folder / "STATS.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
        print(json.dumps(stats), flush=True)
        return stats


def _parquet_rows(paths: list[Path], columns: list[str]):
    import pyarrow.parquet as pq

    for path in paths:
        table = pq.ParquetFile(path)
        for batch in table.iter_batches(batch_size=256, columns=columns):
            yield from batch.to_pylist()


def _image_from_cell(cell) -> Image.Image:
    data = cell["bytes"] if isinstance(cell, dict) else cell
    return Image.open(io.BytesIO(data)).convert("RGB")


# ---------------------------------------------------------------- datasets


def prepare_rxhand() -> None:
    """Prescription words, distributed squashed to 128x128: crop the ink and restore a plausible width."""
    writer = Writer("rxhand")
    archive = zipfile.ZipFile(RAW / "rxhandbd" / "RxHand Original.zip")
    rows = list(csv.reader(io.StringIO(archive.read("Prescription_Labels.csv").decode("utf-8-sig"))))[1:]
    root = "RxHand-Handwritten Prescription Word Image Dataset/"
    for row in rows:
        name, label = row[0], row[1].strip()
        if "?" in label:
            writer.dropped["non_latin_label"] += 1
            continue
        image = Image.open(io.BytesIO(archive.read(root + name))).convert("L")
        array = np.asarray(image, dtype=np.float32)
        ink = array < (np.median(array) - 40)
        rows_ink, cols_ink = np.where(ink.any(axis=1))[0], np.where(ink.any(axis=0))[0]
        if len(rows_ink) < 4 or len(cols_ink) < 4:
            writer.dropped["no_ink"] += 1
            continue
        top, bottom = max(0, rows_ink[0] - 4), min(image.height, rows_ink[-1] + 5)
        left, right = max(0, cols_ink[0] - 4), min(image.width, cols_ink[-1] + 5)
        crop = image.crop((left, top, right, bottom))
        target_width = int(crop.height * max(1.2, 0.62 * len(label)))
        crop = crop.resize((max(crop.width, target_width), crop.height), Image.Resampling.BICUBIC)
        writer.add(split_of(name, val=0.1, test=0.2), Path(name).stem, crop, label)
    writer.close({"note": "images were squashed to 128x128 by the publisher; widths are restored heuristically. "
                  "Some labels name the medicine rather than transcribe the writing (e.g. 'Tab' labelled 'Tablet')."})


def detokenize_iam(text: str) -> str:
    """IAM separates punctuation with spaces ("it 's a start ."); write it the way it was handwritten."""
    tokens = text.split(" ")
    out: list[str] = []
    attach_next = False
    quote_open = False
    for position, token in enumerate(tokens):
        if token == '"':
            if (quote_open or position == len(tokens) - 1) and out:
                out[-1] += '"'
            else:
                out.append('"')
                attach_next = True
            quote_open = not quote_open
            continue
        if attach_next and out:
            out[-1] += token
            attach_next = False
            continue
        if out and (re.fullmatch(r"[.,;:!?)]+|\.\.\.", token) or re.fullmatch(r"'(s|re|ll|ve|d|m|t)|n't", token)):
            out[-1] += token
            continue
        out.append(token)
        if token == "(":
            attach_next = True
    return " ".join(out)


def prepare_iam_test() -> None:
    writer = Writer("iam_test")
    for index, row in enumerate(_parquet_rows([RAW / "iam_test" / "data" / "test.parquet"], ["image", "text"])):
        writer.add("test", f"iam_{index:05d}", _image_from_cell(row["image"]), detokenize_iam(row["text"]))
    writer.close({"license": "IAM: non-commercial research only. EVALUATION ONLY.",
                  "note": "IAM tokenizes punctuation with spaces; spaces before punctuation were removed."})


def _gnhk_items(split_dir: Path, split_name: str):
    for meta in sorted(split_dir.rglob("*.json")):
        image_path = meta.with_suffix(".jpg")
        if image_path.exists():
            yield meta, image_path, split_name


def _gnhk_page(args):
    meta_path, image_path, source_split = args
    words = json.loads(Path(meta_path).read_text(encoding="utf-8"))
    image = ImageOps.exif_transpose(Image.open(image_path)).convert("RGB")
    page = Path(image_path).stem
    split = "test" if source_split == "test" else split_of(page, val=0.06)
    results = []
    lines: dict[int, list[dict]] = collections.defaultdict(list)
    for word in words:
        lines[int(word.get("line_idx", -1))].append(word)
    for line_index, members in lines.items():
        members.sort(key=lambda w: min(w["polygon"][f"x{i}"] for i in range(4)))
        run: list[dict] = []
        runs: list[list[dict]] = []
        for word in members:
            text = str(word.get("text", ""))
            if text.startswith("%") and text.endswith("%"):
                if run:
                    runs.append(run)
                run = []
                continue
            run.append(word)
        if run:
            runs.append(run)
        for run_index, run in enumerate(runs):
            points = np.array([[w["polygon"][f"x{i}"], w["polygon"][f"y{i}"]] for w in run for i in range(4)], dtype=np.float64)
            text = " ".join(str(w["text"]) for w in run)
            results.append((split, f"{page}_l{line_index}_{run_index}", quad_crop(image, points, 0.10, 0.08), text))
    return results


def prepare_gnhk() -> None:
    writer = Writer("gnhk")
    root = RAW / "gnhk" / "extracted"
    if not root.exists():
        with zipfile.ZipFile(RAW / "gnhk" / "GNHK-dataset.zip") as archive:
            archive.extractall(root)
        for inner in list(root.rglob("*.zip")):
            with zipfile.ZipFile(inner) as archive:
                archive.extractall(inner.parent)
    jobs = []
    for split_dir in sorted(p for p in root.rglob("*") if p.is_dir() and p.name.lower() in {"train", "test"}):
        jobs.extend(_gnhk_items(split_dir, split_dir.name.lower()))
    if not jobs:
        raise SystemExit(f"no GNHK pages found under {root}")
    with ProcessPoolExecutor(max_workers=12) as pool:
        for results in pool.map(_gnhk_page, jobs, chunksize=4):
            for split, key, crop, text in results:
                writer.add(split, key, crop, text)
    writer.close({"pages": len(jobs), "note": "lines built from word polygons; %...% placeholder tokens split lines"})


def main_band(image: Image.Image) -> Image.Image:
    """Keep the dominant horizontal band of ink.

    The names crops often include printed form text above or below the
    handwritten name (DATE DE NAISSANCE, CODE RNE...). The name is usually the
    largest band, so a band score of height x ink picks it.
    """
    gray = np.asarray(image.convert("L"), dtype=np.float32)
    ink = gray < min(float(np.median(gray)) - 50.0, 160.0)
    profile = ink.sum(axis=1)
    on = profile > max(1, 0.01 * gray.shape[1])
    bands: list[list[int]] = []
    for row, active in enumerate(on):
        if not active:
            continue
        if bands and row - bands[-1][1] <= 2:
            bands[-1][1] = row
        else:
            bands.append([row, row])
    if not bands:
        return image
    top, bottom = max(bands, key=lambda band: (band[1] - band[0] + 1) * int(profile[band[0] : band[1] + 1].sum()))
    cols = np.where(ink[top : bottom + 1].any(axis=0))[0]
    if len(cols) == 0:
        return image
    pad = max(2, int(0.15 * (bottom - top + 1)))
    return image.crop(
        (max(0, int(cols[0]) - pad), max(0, top - pad), min(gray.shape[1], int(cols[-1]) + pad + 1), min(gray.shape[0], bottom + pad + 1))
    )


def prepare_hw_names() -> None:
    writer = Writer("hw_names")
    folder = RAW / "hw_names" / "data"
    sources = {
        "train": sorted(folder.glob("train-*.parquet")),
        "val": sorted(folder.glob("validation-*.parquet")),
        "test": sorted(folder.glob("test-*.parquet")),
    }
    for split, paths in sources.items():
        for index, row in enumerate(_parquet_rows(paths, ["image", "text"])):
            text = str(row["text"] or "").strip()
            if text.upper() in {"EMPTY", "UNREADABLE", "NAN", ""}:
                writer.dropped["unreadable_or_empty"] += 1
                continue
            writer.add(split, f"{split}_{index:06d}", main_band(_image_from_cell(row["image"])), text)
    writer.close({"note": "dominant ink band kept; run scripts/filter_hw_names.py to drop crops with printed form labels"})


def prepare_cord() -> None:
    writer = Writer("cord")
    folder = RAW / "cord" / "data"
    for source in ("train", "validation", "test"):
        split = {"validation": "val"}.get(source, source)
        for index, row in enumerate(_parquet_rows(sorted(folder.glob(f"{source}-*.parquet")), ["image", "ground_truth"])):
            image = ImageOps.exif_transpose(_image_from_cell(row["image"]))
            truth = json.loads(row["ground_truth"])
            meta = truth.get("meta", {}).get("image_size", {})
            scale_x = image.width / float(meta.get("width") or image.width)
            scale_y = image.height / float(meta.get("height") or image.height)
            groups: dict[tuple, list[dict]] = collections.defaultdict(list)
            for line in truth.get("valid_line", []):
                for word in line.get("words", []):
                    groups[(word.get("row_id"), line.get("group_id"))].append(word)
            for key, words in groups.items():
                words.sort(key=lambda w: w["quad"]["x1"])
                points = np.array(
                    [[w["quad"][f"x{i}"] * scale_x, w["quad"][f"y{i}"] * scale_y] for w in words for i in range(1, 5)]
                )
                text = " ".join(w["text"] for w in words)
                writer.add(split, f"{source}_{index:04d}_{key[0]}_{key[1]}", quad_crop(image, points, 0.08, 0.15), text)
    writer.close({"note": "receipt photos; lines = words sharing a row_id within a group"})


def _page_lines(page_xml: str):
    """Lines from PAGE XML (TextLine / TextEquiv / Coords), which is what the Bentham mirror stores."""
    import xml.etree.ElementTree as ET

    root = ET.fromstring(page_xml)
    page_w = page_h = None
    for element in root.iter():
        if element.tag.endswith("Page") and element.get("imageWidth"):
            page_w, page_h = float(element.get("imageWidth")), float(element.get("imageHeight"))
            break
    for element in root.iter():
        if not element.tag.endswith("TextLine"):
            continue
        texts = [
            child.text.strip()
            for child in element.iter()
            if child.tag.endswith("Unicode") and child.text and child.text.strip()
        ]
        polygon = None
        for child in element:
            if not child.tag.endswith("Coords"):
                continue
            if child.get("points"):
                values = [float(v) for v in re.split(r"[ ,]+", child.get("points").strip()) if v]
                polygon = np.array(values, dtype=np.float64).reshape(-1, 2)
            else:
                points = [
                    (float(point.get("x")), float(point.get("y")))
                    for point in child
                    if point.tag.endswith("Point")
                ]
                if points:
                    polygon = np.array(points, dtype=np.float64)
            break
        if polygon is None or len(polygon) < 2:
            continue
        yield texts[-1] if texts else "", polygon, page_w, page_h


def prepare_bentham() -> None:
    writer = Writer("bentham")
    folder = RAW / "bentham" / "data"
    # The mirror's official test pages, and three training shards, have region boxes only.
    # The held-out slice is a hash split of the pages that do carry line transcripts.
    pages = transcribed = 0
    for row in _parquet_rows(sorted(folder.glob("train-*.parquet")), ["image", "alto", "filename"]):
        pages += 1
        image = None
        page = Path(str(row["filename"])).stem
        split = split_of(page, val=0.06, test=0.10)
        for line_index, (text, polygon, page_w, page_h) in enumerate(_page_lines(row["alto"])):
            if not text:
                writer.dropped["no_transcript"] += 1
                continue
            if image is None:
                image = _image_from_cell(row["image"])
                transcribed += 1
            if page_w and page_h and (image.width != page_w or image.height != page_h):
                polygon = polygon * [image.width / page_w, image.height / page_h]
            writer.add(split, f"{page}_{line_index:03d}", quad_crop(image, polygon, 0.04, 0.12), text)
    writer.close({
        "pages": pages,
        "pages_with_transcripts": transcribed,
        "note": "PAGE XML line transcripts. Official test labels are absent from this mirror, so test is a hash split of the transcribed training pages.",
    })


PREPARERS = {
    "rxhand": prepare_rxhand,
    "iam_test": prepare_iam_test,
    "gnhk": prepare_gnhk,
    "hw_names": prepare_hw_names,
    "cord": prepare_cord,
    "bentham": prepare_bentham,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("names", nargs="+", help=f"any of: {', '.join(PREPARERS)}")
    args = parser.parse_args()
    for name in args.names:
        if name not in PREPARERS:
            parser.error(f"unknown dataset {name}")
        print(f"[{name}]", flush=True)
        PREPARERS[name]()


if __name__ == "__main__":
    main()
