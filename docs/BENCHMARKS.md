# Benchmarks

The numbers below come from one local run on 22 October 2026. The raw reports are in `artifacts/benchmarks/`, which is gitignored. Synthetic scores measure how well the model learned the renderer. FUNSD measures real scanned forms.

## Checkpoint

wormOCR tiny, `artifacts/checkpoints/tiny/best.pt`. This is the EMA weights at the lowest validation CER, reached after 40,000 steps. The maturity label is `trained`.

| Phase | Hardware | Steps | Batch size | Wall time |
| --- | --- | ---: | ---: | ---: |
| Baseline | Intel i9-14900HX, CPU | 0 → 4,000 | 32 | about 66 min |
| Resume | NVIDIA RTX 5060 Laptop GPU, AMP | 4,000 → 40,000 | 64 | about 65 min |

## Machine

- Windows 11, Python 3.14.3, PyTorch 2.14.1+cu130
- Intel64 Family 6 Model 183 (i9-14900HX), 24 torch threads
- Latency was measured on the CPU, not the GPU.
- Loading the checkpoint took 60.6 ms.

## Size

| Item | Value |
| --- | ---: |
| Parameters | 2,174,880 |
| FP32 weights (`parameters × 4`) | 8.30 MiB |
| `best_weights.pt` (weights only) | 8,726,815 bytes |
| `best.pt` (includes the optimizer and raw weights) | 34,863,061 bytes |
| FP32 ONNX | 8,706,790 bytes |
| INT8 ONNX (dynamic) | 8,118,000 bytes |

FP32 ONNX matched PyTorch with a maximum absolute difference of 4.9e-5 in log-probabilities. INT8 dynamic quantization differed by up to 2.28 and saved only 7% of the size. Its CER was not measured, so it is not recommended.

## Accuracy

| Set | Lines | CER | WER | Exact match |
| --- | ---: | ---: | ---: | ---: |
| Synthetic validation (seeded, augmentation off), step 40,000 best | 800 | 0.0070 | 0.0366 | 0.960 |
| `wormocr benchmark` synthetic set | 40 | 0.0304 | 0.1224 | 0.850 |
| Same architecture, random weights, same 40 lines | 40 | 0.9962 | 1.0000 | 0.000 |
| FUNSD test split, word crops | 8,526 | 0.1615 | 0.4475 | 0.567 |

The 40-line benchmark set is small, so its CER moves a lot from a few mistakes. The 800-line validation set is the steadier synthetic number.

The FUNSD crops come from `scripts/prepare_funsd.py` (see [DATASETS.md](DATASETS.md)). FUNSD was never used for training. The step-2,000 checkpoint scored 0.2266 CER on the same crops.

FUNSD error breakdown at step 40,000:

- 3,879 substitutions, 1,984 insertions, and 1,164 deletions
- 2,166 errors involve punctuation
- The most frequent confusions are `T`→`I`, `O`→`Q`, `e`→`c`, `F`→`E`, `1`→`l`, and `,`→`.`
- Crops of one or two characters, such as `/` or `4`, sometimes decode to several characters

## CPU latency

Measured after warmup. The fixed-width rows use a random tensor 256 px wide. The end-to-end rows use a generated line, which is often narrower, so they are not directly comparable to the fixed-width rows.

| Path | Batch | Median | p95 | Throughput |
| --- | ---: | ---: | ---: | ---: |
| Model only, width 256 | 1 | 26.03 ms | 33.04 ms | 38.4 lines/s |
| Model only, width 256 | 8 | 50.98 ms | 70.58 ms | 156.9 lines/s |
| Preprocess + model + greedy decode | 1 | 14.42 ms | 25.46 ms | 69.4 lines/s |
| Preprocess + model + greedy decode | 8 | 42.23 ms | 54.58 ms | 189.4 lines/s |

The process working set peaked at 600 MB during timing. That figure includes Python and the CUDA build of PyTorch, not just the model.

## Baselines that were not run

Tesseract is not on PATH, and PaddleOCR is not installed. Published file sizes for those systems are in [BASELINE_COMPARISON.md](BASELINE_COMPARISON.md). This run does not compare accuracy with them.
