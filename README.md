# wormOCR

**A small, local OCR engine for printed and handwritten English text.**

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-ee4c2c.svg)](https://pytorch.org)
[![ONNX](https://img.shields.io/badge/export-ONNX-005ced.svg)](https://onnx.ai)

wormOCR reads printed and handwritten text from images, scans, and PDFs on an ordinary CPU. The released model, **wormocr-2M**, has **2.17 million parameters** (8.7 MB of weights). It runs entirely on your machine: there is no cloud API, and importing the package never downloads anything. You fetch the weights once, explicitly, with `wormocr download`.

```python
from wormocr import OCR

ocr = OCR()                                   # wormocr-2M, after `wormocr download`
print(ocr.predict("invoice_line.png").text)
# Invoice #1042 Total: $18.50
```

> **Status: early release (v0.1.0).** Clean printed lines and receipt photos are strong. Messy notes and historical handwriting are usable and still well short of a careful human reading. See [Limitations](#limitations).

---

## Contents

- [Highlights](#highlights)
- [Model](#model)
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

- **Small.** 2,174,880 parameters, an 8.7 MB weights file, and an 8.7 MB ONNX graph.
- **Fast on CPU.** About 14 ms per text line end to end (preprocessing, model, and decoding) on a laptop CPU, with no GPU needed for inference.
- **Fully local.** No network calls, no telemetry, and no hidden weight downloads. Weights come from a pinned GitHub release only when you run `wormocr download`, and every file is checked against a SHA-256 hash.
- **One model for print and handwriting.** The released weights read printed lines, handwritten notes, names, and historical pages. No third-party OCR weights are used.
- **Complete toolkit.** One package handles training, resuming, evaluation (CER, WER, exact match), error analysis, ONNX export, benchmarking, page layout, and PDF input.
- **Browser demo.** `wormocr serve` opens a local page where you can drop in an image or PDF and see the extracted text with line boxes.
- **Honest metadata.** Each checkpoint records how it was trained (`smoke-tested`, `debug-trained`, or `trained`), and wormOCR warns you when you load an immature one.

## Model

| | wormocr-2M |
| --- | --- |
| Parameters | 2,174,880 (2.17M) |
| Weights file (`wormocr-2M.safetensors`, FP32) | 8.7 MB |
| ONNX (FP32) | 8.7 MB |
| Architecture | CRNN: depthwise-separable CNN, then a 2-layer BiGRU (256 hidden units), then a CTC head |
| Input | Grayscale line, height 32 px, variable width |
| Output stride | 4 input columns per output frame |
| Character set | 95 printable ASCII characters (letters, digits, punctuation, space) plus the CTC blank |
| Training data | Synthetic print, synthetic handwriting, and real lines (GNHK, handwritten names, Bentham, CORD). FUNSD, IAM, and RxHand were evaluation only |
| Maturity label | `trained` (best checkpoint, step 58,000 of a 60,000-step run) |

## Results

All numbers below are from one greedy-decode run on 23 October 2026, on an NVIDIA RTX 5060 Laptop GPU. This release replaces the printed-only weights previously published under the same name. The earlier scores (0.70% CER on clean synthetic lines, 16.1% on FUNSD) describe those replaced weights and are kept in [docs/BENCHMARKS.md](docs/BENCHMARKS.md).

### Accuracy

| Test set | Lines | CER | WER | Exact match |
| --- | ---: | ---: | ---: | ---: |
| Synthetic printed lines | 1,000 | 3.1% | 9.1% | 90.0% |
| [FUNSD](https://guillaumejaume.github.io/FUNSD/) test, scanned form words (evaluation only) | 8,526 | **11.5%** | 35.8% | 65.6% |
| CORD receipt lines | 795 | **3.7%** | 14.9% | 71.4% |
| Synthetic handwriting | 1,000 | 9.2% | 27.7% | 55.2% |
| Handwritten form names | 26,170 | **5.3%** | 18.7% | 82.3% |
| GNHK handwritten notes | 2,410 | 23.3% | 61.4% | 16.1% |
| IAM lines (evaluation only) | 2,915 | 22.4% | 60.2% | 1.4% |
| Bentham historical lines | 1,135 | 15.1% | 43.0% | 7.1% |
| RxHand prescription words, case-insensitive (evaluation only) | 1,107 | 37.9% | 92.8% | 6.8% |

CER is the character error rate and WER is the word error rate; lower is better. FUNSD, IAM, and RxHand were not used for training. Bentham lines are long, so a few wrong characters make the whole line miss exact match.

### Speed (CPU, single process)

Intel Core i9-14900HX, PyTorch 2.14.1, FP32. Times are medians after warmup.

| Path | Batch | Median latency | Throughput |
| --- | ---: | ---: | ---: |
| Model only, 256 px wide line | 1 | 26.0 ms | 38 lines/s |
| Model only, 256 px wide line | 8 | 51.0 ms | 157 lines/s |
| Preprocess + model + decode | 1 | 14.4 ms | 69 lines/s |
| Preprocess + model + decode | 8 | 42.2 ms | 189 lines/s |

The network is the same as the previous release, so these CPU timings still describe it. They were measured on 22 October 2026. The end-to-end rows use generated lines, which are often narrower than 256 px, so they can be faster than the model-only rows. Loading the checkpoint takes about 60 ms.

### Training cost

| Run | Hardware | Steps | Batch size | Wall time |
| --- | --- | ---: | ---: | ---: |
| wormocr-2M (this release) | NVIDIA RTX 5060 Laptop GPU, mixed precision | 60,000 | 64 | 125 min |

## Installation

Requires Python 3.10 or newer.

```bash
git clone https://github.com/arjun988/wormOCR.git
cd wormOCR
python -m venv .venv
.venv\Scripts\activate            # Windows
# source .venv/bin/activate       # macOS / Linux
pip install -e ".[dev,export,pdf]"
```

The extras are optional:

| Extra | Installs | Needed for |
| --- | --- | --- |
| `dev` | `pytest` | Running the test suite |
| `export` | `onnx`, `onnxruntime` | ONNX and INT8 export |
| `pdf` | `pypdfium2` | Reading PDF files |

The core dependencies are `torch`, `numpy`, `pillow`, `pyyaml`, and `safetensors`. A CPU build of PyTorch is enough for inference. For GPU training, install a CUDA build of PyTorch from [pytorch.org](https://pytorch.org/get-started/locally/).

Check your setup:

```bash
wormocr info
```

This prints the Python and PyTorch versions, whether CUDA is available, and how many usable fonts were found.

### Weights

Model weights are **not stored in git**. They are published as assets of the [`wormocr-2M` GitHub release](https://github.com/arjun988/wormOCR/releases/tag/wormocr-2M):

| File | Contents | Size |
| --- | --- | ---: |
| `wormocr-2M.safetensors` | Weights (tensors only, no pickled code) | 8.7 MB |
| `wormocr-2M.json` | Architecture config, charset, maturity label, training step, measured scores | 4 KB |
| `wormocr-2M.onnx` | FP32 ONNX graph for ONNX Runtime | 8.7 MB |
| `SHA256SUMS.txt` | SHA-256 of each file | |

Download and verify them once:

```bash
wormocr download              # same as: wormocr download wormocr-2M
wormocr download --force     # replace a copy downloaded before 23 October 2026
```

Files go to `~/.cache/wormocr/wormocr-2M/` (set `WORMOCR_HOME` to change this, or pass `--dir`). The expected SHA-256 hashes are built into the package, so a corrupted or altered file is rejected. Nothing else in wormOCR uses the network. If the weights are missing, `OCR()` raises an error that tells you to run the download command.

You can also load any local weights file directly. This works for released `.safetensors` files (with the `.json` file next to them) and for `.pt` checkpoints from your own training:

```python
OCR(checkpoint="path/to/wormocr-2M.safetensors")
OCR(checkpoint="artifacts/checkpoints/tiny/best.pt")
```

Training checkpoints (`.pt`) are PyTorch pickle files. Load only files you trust (see [SECURITY.md](SECURITY.md)).

## Usage

### Python

```python
from wormocr import OCR

ocr = OCR()                                   # wormocr-2M; or OCR(checkpoint="path/to/weights")

line = ocr.predict("line.png", mode="line")   # one text line
print(line.text)

page = ocr.predict("page.png")                # mode="auto": line or page, chosen by aspect ratio
for item in page.lines:
    print(item.order, item.text, item.confidence, item.bbox)
```

`predict` accepts a file path, a PIL image, or a PDF (with the `pdf` extra). `predict_batch` takes a list of these. Each entry in `result.lines` has `text`, `confidence`, `bbox` (an axis-aligned `(x, y, width, height)` in original image coordinates), and `order` (its reading-order position). Pass `device="cuda"` or `beam_width=8` to `OCR(...)` if you need them.

| Mode | Behavior |
| --- | --- |
| `line` | Treats the whole image as one text line |
| `page` | Runs line detection, deskew, and reading order first |
| `auto` | Uses `line` for short, wide images and `page` otherwise |

### Command line

```bash
wormocr download                                  # once
wormocr predict --image line.png
wormocr predict --image page.png --json
wormocr predict --image document.pdf --checkpoint artifacts/checkpoints/tiny/best.pt
```

Run `wormocr <command> --help` for every option. The available commands are `info`, `train`, `predict`, `evaluate`, `analyze`, `export`, `benchmark`, `generate`, and `serve`.

### Web UI

```bash
wormocr serve            # then open http://127.0.0.1:8000
```

`wormocr serve` opens a local web app for using wormOCR on your own files. Drop in a file, paste an image with <kbd>Ctrl</kbd>+<kbd>V</kbd>, or click **Try a sample invoice**. Recognition starts immediately, and the results stream in while the model works.

The app fills the browser window and does not scroll as a page. The document, the results, and the activity log each scroll within their own panel.

- **Document panel.** Every PDF page is shown in order with its detected line boxes. Boxes fill in as lines are recognized. Click a box to find its text, or click a text line to find its box. You can zoom in and out, fit the page to the panel, or hide the boxes.
- **Results panel.** A Text tab grouped by page (low-confidence lines are underlined), a Lines table with page, confidence, and CTC time steps, and a JSON tab. Copy or download any of them.
- **Activity log.** Every step the server performs, with real timings: image size, deskew angle, how many lines were found, each recognition batch, and each line as soon as it is decoded. Turn off **Line events** to show only the pipeline steps.
- **Status bar.** Upload, Decode, Layout, Recognize, and Done are marked as each step finishes, along with per-stage timings.
- **Modes.** Auto, Page, or Line. Changing the mode re-runs recognition.
- **Theme.** Light and dark themes. The app follows your system setting, and you can switch with the toggle.

Open `http://127.0.0.1:8000/#sample` to run the sample invoice as soon as the page loads.

The server uses only the Python standard library. It listens on `127.0.0.1` by default, so files never leave your computer. It accepts uploads up to 25 MB and PDFs up to 20 pages. Use `--port`, `--device cuda`, or `--checkpoint path/to/best.pt` to change the defaults. On CPU the server uses half the logical cores, up to 16 threads, because running PyTorch on every core makes the recurrent layer slower; set the count yourself with `--threads`. On Windows the server also opts out of power throttling, which would otherwise move it to efficiency cores and make recognition 2–3 times slower.

| Endpoint | Returns |
| --- | --- |
| `POST /api/ocr/stream?mode=auto&filename=x.png` | NDJSON events as the pipeline runs: `start`, `log`, `document`, `page`, `layout`, `line`, `page_done`, `done` (or `error`) |
| `POST /api/ocr?mode=auto&filename=x.png` | One JSON result with pages, lines, boxes, confidences, and timings |
| `GET /api/health` | The loaded model's name, parameter count, architecture, and device |

Send the raw file bytes as the request body.

## Training

Synthetic lines are rendered on the fly from fonts on your machine. On Windows these come from `C:\Windows\Fonts`. You can point `data.font_dir` at any folder of `.ttf` files. Each synthetic sample is generated from its index and the seed, so a run can be reproduced exactly.

The released wormocr-2M weights are the best validation checkpoint from `configs/unified.yaml`: step 58,000 of a 60,000-step GPU run (about 2 hours on an RTX 5060 Laptop, batch 64, mixed precision). Training started from the previous printed-only weights and mixed synthetic print, synthetic handwriting, and real line crops.

```bash
wormocr train --config configs/unified.yaml --mode baseline
```

Any config field can be overridden with `--set section.key=value`. Training writes `best.pt` (lowest validation CER), `last.pt` (newest step, used for resuming), `best_weights.pt` (weights only, for distribution), and `train_log.jsonl`.

| Mode | Purpose |
| --- | --- |
| `smoke` | A few steps to check that data, loss, validation, and checkpointing all run |
| `debug` | A short experiment on a tiny network |
| `baseline` | A full run with the config as written |
| `resume` | Continue from `last.pt` with the weights, optimizer, EMA, and step counter |

To train on your own labelled images, use a tab-separated file containing `relative/path.png<TAB>exact text`. See [docs/DATASETS.md](docs/DATASETS.md).

```bash
wormocr train --config configs/tiny.yaml --mode baseline \
  --set data.labels=data/train.tsv --set data.image_root=data/images \
  --set data.val_labels=data/val.tsv --set data.val_image_root=data/images
```

## Evaluation, export, and benchmarks

```bash
# Accuracy on held-out synthetic lines, or on your own label file
wormocr evaluate --model artifacts/checkpoints/tiny/best.pt --synthetic-samples 800
wormocr evaluate --model artifacts/checkpoints/tiny/best.pt --labels data/val.tsv --image-root data/images --errors errors.md

# Score a label file. FUNSD is evaluation only; download it yourself.
python scripts/prepare_funsd.py artifacts/datasets/funsd/dataset/testing_data artifacts/datasets/funsd_test
wormocr evaluate --model artifacts/checkpoints/unified/best.pt --labels artifacts/datasets/funsd_test/labels.tsv

# ONNX export (checked against PyTorch; max abs difference 5e-5 on the released model)
wormocr export --model artifacts/checkpoints/tiny/best.pt --format onnx --output artifacts/export/tiny.onnx

# Size, accuracy, latency, and memory report
wormocr benchmark --model artifacts/checkpoints/tiny/best.pt --output artifacts/benchmarks/report.md
```

The evaluator does not lowercase or strip text. CER counts every character, including spaces.

An INT8 export is available (`--format onnx-int8`), but it is **not recommended** for this model. Dynamic quantization only shrinks the file from 8.7 MB to 8.1 MB, and its outputs drift noticeably from FP32. Use the FP32 ONNX graph.

## How it works

1. **Preprocessing.** The image is converted to grayscale and resized to 32 px tall, keeping its aspect ratio, then normalized to `[-1, 1]`. Batches are padded on the right with the line's own background value.
2. **Encoder.** A four-stage convolutional network with depthwise-separable convolutions. It pools height from 32 to 2 and width by 4.
3. **Sequence model.** A linear projection feeds a 2-layer bidirectional GRU. That reads the line left to right and right to left.
4. **Decoder.** A CTC head over 96 classes, decoded greedily by default. Beam search is optional.
5. **Page layout** (page mode only). Optional deskew within ±5°, Otsu binarization, horizontal projection profiles to find lines, gutter detection for columns, and top-to-bottom reading order within each column. Boxes are mapped back to the original image.

Training details:

- **Loss and optimizer.** CTC loss with AdamW, linear warmup, then cosine decay.
- **Weight averaging.** An exponential moving average of the weights with decay 0.999. The averaged weights are the ones evaluated and shipped.
- **Mixed precision.** Used on CUDA.
- **Augmentation.** Blur, noise, JPEG artifacts, lighting gradients, rotation, shear, stroke thickening and thinning, binarization, and a low-resolution "scan" pass.
- **Form-style text.** Amounts, phone numbers, IDs, and `Field:` labels.

More detail is in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) and [docs/DATASETS.md](docs/DATASETS.md).

## Project layout

```text
wormOCR/
├── configs/            Model and training configs (unified.yaml produced the released wormocr-2M)
├── docs/               Architecture, datasets, benchmarks, roadmap
├── scripts/            Dataset preparation and benchmark helpers
├── src/wormocr/
│   ├── api.py          OCR class (public Python API)
│   ├── cli.py          `wormocr` command
│   ├── checkpoints.py  Checkpoint loading and lookup
│   ├── hub.py          Released models, `wormocr download`, SHA-256 checks
│   ├── preprocessing.py
│   ├── vocabulary.py   Character set and CTC helpers
│   ├── data/           Synthetic renderer, label-file datasets, fonts
│   ├── detection/      Line finding, deskew, reading order
│   ├── inference/      Line engine and CTC decoding
│   ├── models/         CRNN definition
│   ├── training/       Training loop and EMA
│   ├── evaluation/     CER/WER, error reports, robustness slices
│   ├── export/         ONNX and INT8 export
│   ├── benchmark/      Benchmark report
│   └── web/            `wormocr serve`: local server and HTML/CSS/JS UI
└── tests/              pytest suite
```

## Limitations

- **English printable ASCII only.** Accented letters, symbols such as `€` or `°`, and other scripts are rejected with an error rather than silently dropped.
- **Messy handwriting is still hard.** Historical pages and notes in the wild are around 15–23% character error. Prescription words are harder still (about 38% case-insensitive on RxHand, which was not used for training).
- **Real scans are harder than clean text.** FUNSD form words are about 11.5% character error. Very small crops of one or two characters remain a weak spot.
- **Simple page layout.** Page layout uses projection profiles, not a learned text detector. Tables, rotated text, and cluttered layouts can merge or drop lines.
- **No language-model correction.** The output is exactly what the decoder emits.
- **No external comparison yet.** Tesseract and PaddleOCR were not installed on the benchmark machine, so no accuracy comparison against them has been run.

See [docs/ROADMAP.md](docs/ROADMAP.md) for planned work.

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md). Before opening a pull request:

```bash
python -m pytest
```

When you report accuracy, include the dataset, the number of lines, and the exact command, and run it on a set the model was not trained on.

## Citation

If you use wormOCR in your work, please cite it (see [CITATION.cff](CITATION.cff)):

```bibtex
@software{shukla2026wormocr,
  author  = {Shukla, Arjun},
  title   = {wormOCR: a small local OCR engine for printed English text},
  year    = {2026},
  version = {0.1.0},
  url     = {https://github.com/arjun988/wormOCR}
}
```

## License

Released under the [MIT License](LICENSE). Copyright (c) 2026 arjun shukla.

wormOCR ships no third-party OCR weights and redistributes no fonts. The training images are rendered from fonts on your own machine. FUNSD is used only for evaluation and is not included in this repository.
