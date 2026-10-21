# wormOCR

Small, local OCR for printed English text. The recognizer is a convolutional encoder, a bidirectional GRU, and a CTC decoder. It runs on CPU. It does not call a cloud API, and importing it does not download weights.

This repository is an early, working system. The software can train, evaluate, export, and recognize text. A checkpoint produced by a short synthetic run is not a competitive document OCR model. Maturity of any checkpoint is stored inside the file and printed when you load a smoke-tested or untrained model.

## What works

- English printed text lines, including letters, digits, punctuation, and case.
- Variable-width lines, batched CPU inference, and greedy CTC decoding. Beam search is optional.
- Synthetic data generation, a PyTorch training loop, checkpoints, and resume.
- CER, WER, exact match, and an error report.
- A classical page pass: line finding, optional deskew, and reading order for simple one- and two-column pages.
- ONNX export and dynamic INT8 quantization when `onnx` and `onnxruntime` are installed.
- PDF rendering when the optional package `pypdfium2` is installed.

## What does not work yet

- Competitive accuracy on real scans, photos, or forms. The training run in this environment was short and synthetic.
- Handwriting, tables, and languages other than English.
- A learned text detector. Page layout uses projection profiles.
- Shipped weights. You train locally, or you pass a checkpoint you already have.

## Install

Python 3.10 or newer. PyTorch, NumPy, Pillow, and PyYAML are required. CUDA is optional; the code uses the CPU when PyTorch cannot see a GPU.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev,export]"
```

On macOS or Linux, activate with `source .venv/bin/activate`.

Without an editable install, from the repository root:

```bash
set PYTHONPATH=src
python -m wormocr info
```

`wormocr info` prints the interpreter, whether CUDA is available, how many Latin fonts were found, and the parameter counts of the three configs. It does not download anything.

## Recognize a line

Train or place a checkpoint first. This example expects one at `artifacts/checkpoints/tiny/best.pt`.

```python
from wormocr import OCR

ocr = OCR(model="tiny")  # or OCR(checkpoint="artifacts/checkpoints/tiny/best.pt")
result = ocr.predict("document.png")
print(result.text)
```

`OCR(model="tiny")` looks for `artifacts/checkpoints/tiny/best.pt` in the current directory or the repository. If that file is missing, it raises an error. It will not download a model.

`result.lines` contains text, a confidence, a bounding box in the original image (`x, y, width, height`), and reading order. `mode="line"` skips page layout. `mode="page"` always runs line detection. `mode="auto"` treats a short, wide image as a single line.

```bash
python -m wormocr predict --image document.png --model tiny
python -m wormocr predict --image document.png --checkpoint artifacts/checkpoints/tiny/best.pt --json
```

## Train

Configs live in `configs/`. `tiny` is the main recognizer. `micro` is the smaller one. `small` is the wider fallback. `debug` is a tiny network for a short experiment.

```bash
python -m wormocr train --config configs/tiny.yaml --mode smoke
python -m wormocr train --config configs/tiny.yaml --mode debug
python -m wormocr train --config configs/tiny.yaml --mode baseline
python -m wormocr train --config configs/tiny.yaml --mode resume --checkpoint artifacts/checkpoints/tiny/last.pt
```

`smoke` checks that data, loss, validation, and checkpointing run. It does not produce a useful model. `debug` is a short experiment. `baseline` uses the config as written. Add `--set train.max_steps=100` or `--set train.batch_size=8` to override any dotted field. Overrides win over the mode preset.

The loop uses AdamW, a warmup followed by cosine decay, gradient clipping, CTC loss, and early stopping on validation CER. Checkpoints store the weights, optimizer, charset, config, and maturity. `best.pt` is the lowest validation CER. `last.pt` is the newest step. Load only checkpoints you trust: they are PyTorch pickle files.

A decreasing loss only means the optimizer is moving. Check CER on held-out lines before trusting a checkpoint.

## Evaluate, export, benchmark

```bash
python -m wormocr evaluate --model artifacts/checkpoints/tiny/best.pt --synthetic-samples 100
python -m wormocr evaluate --model artifacts/checkpoints/tiny/best.pt --labels data/val.tsv --image-root data/images --errors artifacts/errors.md
python -m wormocr export --model artifacts/checkpoints/tiny/best.pt --format onnx --output artifacts/export/tiny.onnx
python -m wormocr export --model artifacts/checkpoints/tiny/best.pt --format onnx-int8 --output artifacts/export/tiny.int8.onnx
python -m wormocr benchmark --model artifacts/checkpoints/tiny/best.pt --output artifacts/benchmarks/report.md
```

A label file is `relative-image-path<TAB>text`, one pair per line. The evaluator does not lowercase or strip text. CER counts every character, including spaces. WER splits on single spaces, so a doubled space counts.

`--synthetic-samples` builds a seeded held-out set from the same renderer used for training, with augmentation off. That number is not a real-document score. Pass `--labels` for a real set. Dataset notes are in [docs/DATASETS.md](docs/DATASETS.md).

ONNX export checks that FP32 outputs stay within `1e-3` of PyTorch. INT8 is dynamic quantization. The benchmark records its actual max absolute difference instead of assuming it is accurate or faster.

## Model size

Measured with `count_parameters` on the configs in this repository, vocabulary of 95 printable ASCII characters plus the CTC blank (96 classes):

| Variant | Parameters | FP32 weight storage | Role |
| --- | ---: | ---: | --- |
| micro | 1,217,744 | 4.65 MB | Tight CPU and memory budget |
| tiny | 3,319,264 | 12.66 MB | Default recognizer |
| small | 11,292,832 | 43.08 MB | Wider fallback |

Weight storage is `parameter_count * 4` bytes. A checkpoint file is larger because it also stores the optimizer. The benchmark writes the checkpoint size, ONNX size, and INT8 size when export succeeds. Architecture notes are in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Fonts and synthetic data

Training images are rendered from fonts already installed on the machine. wormOCR does not ship font files. On Windows it looks in `C:\Windows\Fonts` and prefers common Latin families such as Arial, Calibri, and Georgia. If none of those can draw Latin letters, set `data.font_dir` to a directory of `.ttf` files.

```bash
python -m wormocr generate --config configs/tiny.yaml --samples 32 --out artifacts/data/sample --seed 0
```

That writes PNGs and `manifest.jsonl`. Online training does not need those files; the dataset is seeded from the sample index.

## Page images and PDFs

Page mode deskews slightly, finds text lines with a horizontal projection, splits wide gutters into columns, and reads each column from top to bottom. Full-width lines, such as a title, stay in vertical order. Boxes are mapped back to the original image.

This is a simple layout pass. It is not a table parser, and it will merge or drop lines on cluttered pages. The line recognizer can be used without it.

PDF input needs `pip install pypdfium2`. Without that package, predicting a PDF raises an error that names the missing dependency.

## Development

```bash
python -m pytest
```

The suite covers the vocabulary, variable-width preprocessing, the width formula, CTC collapse of repeated characters, CER/WER, synthetic reproducibility, line detection and reading order, checkpoint resume, a four-line memorization run, the CLI, and ONNX parity when the export packages are installed.

## Limitations

- English printable ASCII only. A character outside that set is rejected rather than silently dropped.
- No language-model correction. The text you get is the decoder output.
- Quantization and a larger model are not automatically better. Use the benchmark.
- External baselines are reported only when the tool is installed. This environment's comparison is written up in [docs/BASELINE_COMPARISON.md](docs/BASELINE_COMPARISON.md) and [docs/BENCHMARKS.md](docs/BENCHMARKS.md).

## License

MIT. See [LICENSE](LICENSE). Copyright (c) 2026 arjun shukla.

No third-party recognition weights are included. Fonts used to render training images stay on your machine and are not redistributed.
