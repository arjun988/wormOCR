"""Recognize one image with a local checkpoint.

Train first. This script does not download weights.

    wormocr train --config configs/tiny.yaml --mode debug
    python examples/quickstart.py artifacts/checkpoints/tiny/best.pt path/to/line.png
"""

from __future__ import annotations

import sys

from wormocr import OCR


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: python examples/quickstart.py CHECKPOINT IMAGE")
        return 2
    ocr = OCR(checkpoint=sys.argv[1], device="cpu")
    result = ocr.predict(sys.argv[2])
    print(result.text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
