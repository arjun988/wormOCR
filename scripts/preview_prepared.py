"""Save a contact sheet of prepared crops with their labels, for a visual check.

    python scripts/preview_prepared.py gnhk train 24
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1] / "artifacts" / "datasets" / "prepared"


def main() -> None:
    name, split = sys.argv[1], sys.argv[2]
    count = int(sys.argv[3]) if len(sys.argv) > 3 else 24
    folder = ROOT / name
    rows = [line.split("\t", 1) for line in (folder / f"{split}.tsv").read_text(encoding="utf-8").splitlines() if line]
    picks = random.Random(0).sample(rows, min(count, len(rows)))
    width, row_height = 900, 64 + 22
    sheet = Image.new("RGB", (width, row_height * len(picks)), "white")
    draw = ImageDraw.Draw(sheet)
    for index, (relative, text) in enumerate(picks):
        crop = Image.open(folder / relative).convert("RGB")
        if crop.width > width - 8:
            crop = crop.resize((width - 8, int(crop.height * (width - 8) / crop.width)))
        top = index * row_height
        sheet.paste(crop, (4, top + 2))
        draw.text((4, top + 66), text[:140], fill=(200, 0, 0))
    target = Path("artifacts") / f"preview_{name}_{split}.png"
    sheet.save(target)
    print(target)


if __name__ == "__main__":
    main()
