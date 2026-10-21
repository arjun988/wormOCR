# Roadmap

Status of the first build. "Done" means the code path exists and the test suite exercises it, not that the model matches a production OCR system.

## Done

- [x] Repository inspected. It started as a license and a one-line README.
- [x] CRNN recognizer with adjustable width, separable or ordinary convolutions, and a width formula checked against the forward pass.
- [x] Printable ASCII vocabulary, variable-width preprocessing, CTC loss, greedy and beam decoders.
- [x] Synthetic renderer, seeded dataset, label-file dataset, manifest export.
- [x] Training loop: smoke, debug, baseline, resume, cosine schedule, checkpoints, early stopping, CPU and CUDA device selection.
- [x] CER, WER, exact match, error report, robustness slices.
- [x] Python API and CLI for predict, train, evaluate, export, benchmark, generate, analyze, info.
- [x] ONNX FP32 parity test and INT8 quantization path.
- [x] Classical page segmentation, deskew, and reading order. Line recognition still works on its own.
- [x] Parameter counts measured for micro, tiny, and small.
- [x] Four clean lines can be memorized by the training loss and greedy decoder.

## Measured, not finished

- [ ] A multi-epoch tiny checkpoint evaluated on a real held-out line set.
- [ ] The same images run through Tesseract or an English PP-OCR mobile model.
- [ ] A choice between micro and tiny based on that comparison, not on parameter count alone.
- [ ] INT8 latency and accuracy on the same real set. Keep FP32 if INT8 loses too much.

## Later

- [ ] Real document-line corpus in the TSV format, with a frozen split.
- [ ] Optional learned detector, sized and licensed separately from the recognizer.
- [ ] More than English, one script at a time, with its own charset and eval set.
- [ ] Tables and handwriting as separate features with their own tests.
- [ ] Mobile runtime (ONNX Runtime Mobile or ExecuTorch) after the CPU numbers are acceptable.

## Not the goal

Academic papers, novelty claims, and publishing a model card with numbers that were not measured.
