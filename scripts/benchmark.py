"""Write a benchmark report. Example: python scripts/benchmark.py --model artifacts/checkpoints/tiny/best.pt"""

from __future__ import annotations

import sys

from wormocr.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["benchmark", *sys.argv[1:]]))
