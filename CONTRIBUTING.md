# Contributing

Issues and patches are welcome. Keep changes small enough to review, and add a test when you change decoding, the width formula, metrics, or the training loop.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev,export]"
python -m pytest
```

`python -m wormocr train --config configs/debug.yaml --mode smoke` should finish in about a minute on a CPU and write a smoke-tested checkpoint. Do not treat that checkpoint as a model to publish.

## Conventions

- Do not download datasets or pretrained weights from tests or from import.
- Do not lowercase or strip references inside CER/WER. Spacing errors are errors.
- Reject characters outside the charset instead of dropping them.
- If you change `model.channels` or the pooling pattern, update `output_width` and the test that compares it with a real forward pass.
- New checkpoints should keep the maturity field honest.
- Benchmark tables in the docs should be copied from a `wormocr benchmark` report, not typed from memory.

## Licenses

This project is MIT. Do not add code or weights under a license that conflicts with that without calling it out in the pull request. Fonts used for synthetic data are read from the machine and must not be committed.
