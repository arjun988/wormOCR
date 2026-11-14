# wormOCR

**Local OCR for printed and handwritten English text.**

[![PyPI](https://img.shields.io/pypi/v/wormocr.svg)](https://pypi.org/project/wormocr/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-ee4c2c.svg)](https://pytorch.org)

Install the package, download the weights once, and read text on your machine. No cloud API. No silent downloads on import.

## Quick start

```bash
pip install wormocr
wormocr download
```

```python
from wormocr import OCR

ocr = OCR()
print(ocr.predict("invoice_line.png").text)
# Invoice #1042 Total: $18.50
```

That uses **wormocr-2M** (2.17M parameters, 8.7 MB). For the larger model:

```bash
wormocr download wormocr-9M
```

```python
ocr = OCR(model="wormocr-9M")
```

> **v1.0.0.** Clean printed lines and receipts are strong. Messy notes and historical handwriting are usable and still short of a careful human reading. See [Limitations](#limitations).

---

## Contents

- [Highlights](#highlights)
- [Models](#models)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Training](#training)
- [Evaluation, export, and benchmarks](#evaluation-export-and-benchmarks)
- [How it works](#how-it-works)
- [Project layout](#project-layout)
- [Limitations](#limitations)
- [Contributing](#contributing)
- [Citation](#citation)
- [License](#license)

## Highlights

- **Small enough to ship.** wormocr-2M is 8.7 MB. wormocr-9M is 33.5 MB. Both are ordinary CRNN + CTC models.
- **Fast on CPU.** About 14 ms per line end to end with wormocr-2M on a laptop CPU. No GPU required for inference.
- **Fully local.** Importing the package never touches the network. Weights come from a pinned GitHub release only when you run `wormocr download`, and every file is checked against a SHA-256 hash.
- **Print and handwriting.** One recognizer family for invoices, receipts, handwritten names, notes, and historical lines. No third-party OCR weights.
- **Full toolkit.** Train, resume, evaluate (CER / WER / exact match), export ONNX, benchmark, page layout, PDFs, and a local browser demo.

## Models

| | wormocr-2M | wormocr-9M |
| --- | --- | --- |
| Parameters | 2,174,880 | 8,776,416 |
| Weights (`.safetensors`, FP32) | 8.7 MB | 33.5 MB |
| ONNX (FP32) | 8.7 MB | 33.5 MB |
| Architecture | Separable CNN + 2-layer BiGRU (256) + CTC | Same family, BiGRU 512 |
| Input height | 32 px | 48 px |
| Output stride | 4 | 4 |
| Character set | 95 printable ASCII + CTC blank | Same |
| Training data | Synthetic print & handwriting, GNHK, names, Bentham, CORD. FUNSD / IAM / RxHand evaluation only | Same mix |
| Best checkpoint | step 58,000 of 60,000 | step 19,000 (early-stopped at 30,000) |
| Default for `OCR()` | Yes | Pass `model="wormocr-9M"` |

## Results

Character error rate (CER), greedy decode, NVIDIA RTX 5060 Laptop GPU. wormocr-2M measured 23 October 2026. wormocr-9M measured 6 November 2026 on the same frozen test lines. Lower is better. Older printed-only scores live in [docs/BENCHMARKS.md](docs/BENCHMARKS.md).

| Test set | Lines | wormocr-2M | wormocr-9M |
| --- | ---: | ---: | ---: |
| Synthetic printed lines | 1,000 | 3.1% | 3.7% |
| [FUNSD](https://guillaumejaume.github.io/FUNSD/) form words (eval only) | 8,526 | 11.5% | **10.2%** |
| CORD receipts | 795 | 3.7% | **3.1%** |
| Synthetic handwriting | 1,000 | 9.2% | **7.1%** |
| Handwritten form names | 26,170 | **5.3%** | 5.3% |
| GNHK handwritten notes | 2,410 | 23.3% | **19.2%** |
| IAM lines (eval only) | 2,915 | 22.4% | **18.3%** |
| Bentham historical lines | 1,135 | 15.1% | **8.8%** |
| RxHand words, case-insensitive (eval only) | 1,107 | 37.9% | **35.2%** |

FUNSD, IAM, and RxHand were never used for training.

### Speed (CPU, wormocr-2M)

Intel Core i9-14900HX, PyTorch 2.14.1, FP32. Medians after warmup (22 October 2026).

| Path | Batch | Median | Throughput |
| --- | ---: | ---: | ---: |
| Model only, 256 px wide | 1 | 26.0 ms | 38 lines/s |
| Model only, 256 px wide | 8 | 51.0 ms | 157 lines/s |
| Preprocess + model + decode | 1 | 14.4 ms | 69 lines/s |
| Preprocess + model + decode | 8 | 42.2 ms | 189 lines/s |

wormocr-9M is larger and slower. Use it when the handwriting gains matter more than size.

### Training cost

| Run | Hardware | Steps | Effective batch | Wall time |
| --- | --- | ---: | ---: | ---: |
| wormocr-2M | RTX 5060 Laptop, mixed precision | 60,000 | 64 | 125 min |
| wormocr-9M | RTX 5060 Laptop, mixed precision | 30,000 (best at 19,000) | 64 (16 × 4 accum) | about 150 min |

## Installation

Requires Python 3.10+.

### From PyPI (recommended)

```bash
pip install wormocr
wormocr download              # wormocr-2M
# wormocr download wormocr-9M
```

Optional extras:

```bash
pip install "wormocr[pdf]"      # PDF input
pip install "wormocr[export]"   # ONNX export
pip install "wormocr[dev]"      # pytest
```

A CPU build of PyTorch is enough for inference. For GPU training, install a CUDA build from [pytorch.org](https://pytorch.org/get-started/locally/) first, then install wormOCR.

### From source

```bash
git clone https://github.com/arjun988/wormOCR.git
cd wormOCR
python -m venv .venv
.venv\Scripts\activate            # Windows
# source .venv/bin/activate       # macOS / Linux
pip install -e ".[dev,export,pdf]"
wormocr download
```

```bash
wormocr info
```

### Weights

Weights are **not** on PyPI and **not** in git. They are GitHub release assets, verified by SHA-256 hashes baked into the package:

| Release | Size |
| --- | ---: |
| [`wormocr-2M`](https://github.com/arjun988/wormOCR/releases/tag/wormocr-2M) | 8.7 MB |
| [`wormocr-9M`](https://github.com/arjun988/wormOCR/releases/tag/wormocr-9M) | 33.5 MB |

```bash
wormocr download
wormocr download wormocr-9M
wormocr download wormocr-2M --force
```

Files land in `~/.cache/wormocr/<name>/` (override with `WORMOCR_HOME` or `--dir`). If weights are missing, `OCR()` tells you to run the download command. You can also pass a local file:

```python
OCR(checkpoint="path/to/wormocr-2M.safetensors")
OCR(model="wormocr-9M")
OCR(checkpoint="artifacts/checkpoints/tiny/best.pt")
```

Training checkpoints (`.pt`) are PyTorch pickles. Load only files you trust ([SECURITY.md](SECURITY.md)).

## Usage

### Python

```python
from wormocr import OCR

ocr = OCR()
print(ocr.predict("line.png", mode="line").text)

page = ocr.predict("page.png")   # mode="auto" by default
for item in page.lines:
    print(item.order, item.text, item.confidence, item.bbox)
```

`predict` accepts a path, a PIL image, or a PDF (with the `pdf` extra). Pass `device="cuda"` or `beam_width=8` if you need them.

| Mode | Behavior |
| --- | --- |
| `line` | Whole image is one text line |
| `page` | Line detection, deskew, reading order |
| `auto` | `line` for short wide images, else `page` |

### Command line

```bash
wormocr download
wormocr predict --image line.png
wormocr predict --image page.png --json
wormocr predict --image document.pdf --model wormocr-9M
```

Commands: `info`, `download`, `predict`, `train`, `evaluate`, `analyze`, `export`, `benchmark`, `generate`, `serve`.

### Web UI

```bash
wormocr serve            # http://127.0.0.1:8000
```

Drop in an image or PDF, paste with <kbd>Ctrl</kbd>+<kbd>V</kbd>, or try the sample invoice. Recognition streams while the model works. The server listens on localhost by default; files stay on your machine.

```bash
wormocr serve --model wormocr-9M --device cuda --port 8000
```

## Training

Clone the repository for configs and scripts. Synthetic lines are rendered from fonts on your machine. The released weights come from `configs/unified.yaml` (wormocr-2M) and `configs/wormocr-9M.yaml` (wormocr-9M).

```bash
wormocr train --config configs/unified.yaml --mode baseline
wormocr train --config configs/wormocr-9M.yaml --mode baseline
```

Override any field with `--set section.key=value`. Training writes `best.pt`, `last.pt`, `best_weights.pt`, and `train_log.jsonl`.

| Mode | Purpose |
| --- | --- |
| `smoke` | Short check that data, loss, validation, and checkpoints run |
| `debug` | Tiny network experiment |
| `baseline` | Full run as written in the config |
| `resume` | Continue from `last.pt` |

Own labels: `relative/path.png<TAB>exact text`. See [docs/DATASETS.md](docs/DATASETS.md).

```bash
wormocr train --config configs/tiny.yaml --mode baseline \
  --set data.labels=data/train.tsv --set data.image_root=data/images \
  --set data.val_labels=data/val.tsv --set data.val_image_root=data/images
```

## Evaluation, export, and benchmarks

```bash
wormocr evaluate --model artifacts/checkpoints/tiny/best.pt --synthetic-samples 800
wormocr evaluate --model artifacts/checkpoints/tiny/best.pt --labels data/val.tsv --image-root data/images --errors errors.md

python scripts/prepare_funsd.py artifacts/datasets/funsd/dataset/testing_data artifacts/datasets/funsd_test
wormocr evaluate --model artifacts/checkpoints/unified/best.pt --labels artifacts/datasets/funsd_test/labels.tsv

wormocr export --model artifacts/checkpoints/tiny/best.pt --format onnx --output artifacts/export/tiny.onnx
wormocr benchmark --model artifacts/checkpoints/tiny/best.pt --output artifacts/benchmarks/report.md
```

CER counts every character, including spaces. INT8 ONNX exists (`--format onnx-int8`) but is not recommended: it barely shrinks the file and drifts from FP32.

## How it works

1. **Preprocessing.** Grayscale, resize to model height (32 or 48), normalize to `[-1, 1]`, pad with the line's background.
2. **Encoder.** Four-stage depthwise-separable CNN. Height pools to 2; width by 4.
3. **Sequence model.** Linear projection into a 2-layer bidirectional GRU (256 or 512).
4. **Decoder.** CTC over 96 classes, greedy by default; beam search optional.
5. **Page layout.** Deskew, Otsu, projection profiles, gutters, reading order.

Training uses CTC + AdamW, warmup then cosine decay, EMA 0.999, AMP on CUDA, and scan-style augmentation. Details: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md), [docs/DATASETS.md](docs/DATASETS.md).

## Project layout

```text
wormOCR/
├── configs/            unified.yaml → wormocr-2M; wormocr-9M.yaml → wormocr-9M
├── docs/               Architecture, datasets, benchmarks, roadmap
├── scripts/            Dataset prep and helpers
├── src/wormocr/
│   ├── api.py          OCR class
│   ├── cli.py          wormocr command
│   ├── hub.py          Releases, download, SHA-256 checks
│   ├── models/         CRNN
│   ├── training/       Loop and EMA
│   ├── inference/      Engine and CTC decode
│   ├── detection/      Line finding and reading order
│   ├── evaluation/     CER / WER
│   ├── export/         ONNX
│   └── web/            Local serve UI
└── tests/
```

## Limitations

- **English printable ASCII only.** Accented letters and other scripts are rejected, not silently dropped.
- **Messy handwriting is still hard.** On wormocr-9M: GNHK 19.2%, IAM 18.3%, Bentham 8.8%, RxHand 35.2% case-insensitive (eval only).
- **Real scans beat clean synth.** FUNSD form words are about 10–11.5% CER. Tiny one-character crops remain weak.
- **Simple page layout.** Projection profiles, not a learned detector. Tables and heavy clutter can fail.
- **No language-model correction.** Output is exactly what the decoder emits.
- **No third-party OCR baseline yet.** Tesseract / PaddleOCR were not measured on this machine.

See [docs/ROADMAP.md](docs/ROADMAP.md).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

```bash
python -m pytest
```

When you report accuracy, include the dataset, line count, exact command, and a set the model was not trained on.

## Citation

```bibtex
@software{shukla2026wormocr,
  author  = {Shukla, Arjun},
  title   = {wormOCR: a small local OCR engine for printed and handwritten English text},
  year    = {2026},
  version = {1.0.0},
  url     = {https://github.com/arjun988/wormOCR}
}
```

Also [CITATION.cff](CITATION.cff).

## License

[MIT License](LICENSE). Copyright (c) 2026 arjun shukla.

wormOCR ships no third-party OCR weights and redistributes no fonts. FUNSD is evaluation-only and is not included in this repository.
