"""Crop FUNSD word boxes into a wormOCR label file.

FUNSD is licensed for non-commercial research and education only. Use the
output to evaluate a model locally; do not train released weights on it or
commit the crops.

    python scripts/prepare_funsd.py artifacts/datasets/funsd/dataset/testing_data artifacts/datasets/funsd_test
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

from wormocr.vocabulary import Vocabulary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("split", help="FUNSD split folder with images/ and annotations/")
    parser.add_argument("output", help="folder for crops and labels.tsv")
    parser.add_argument("--pad", type=int, default=2, help="pixels added around each box")
    parser.add_argument("--min-height", type=int, default=8)
    args = parser.parse_args()

    vocabulary = Vocabulary()
    split = Path(args.split)
    output = Path(args.output)
    crops = output / "crops"
    crops.mkdir(parents=True, exist_ok=True)
    kept = skipped_chars = skipped_small = 0
    lines: list[str] = []
    for annotation in sorted((split / "annotations").glob("*.json")):
        image_path = split / "images" / f"{annotation.stem}.png"
        page = Image.open(image_path).convert("L")
        form = json.loads(annotation.read_text(encoding="utf-8"))["form"]
        for block in form:
            for word in block["words"]:
                text = word["text"].strip().replace("\u2013", "-").replace("\u2019", "'")
                if not text:
                    continue
                try:
                    vocabulary.encode(text)
                except ValueError:
                    skipped_chars += 1
                    continue
                left, top, right, bottom = word["box"]
                if bottom - top < args.min_height or right <= left:
                    skipped_small += 1
                    continue
                box = (
                    max(0, left - args.pad),
                    max(0, top - args.pad),
                    min(page.width, right + args.pad),
                    min(page.height, bottom + args.pad),
                )
                name = f"{annotation.stem}_{kept:05d}.png"
                page.crop(box).save(crops / name)
                lines.append(f"crops/{name}\t{text}")
                kept += 1
    (output / "labels.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "words": kept,
        "skipped_outside_charset": skipped_chars,
        "skipped_too_small": skipped_small,
        "labels": str(output / "labels.tsv"),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
