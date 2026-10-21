# Baseline comparison

Decision: ship a trainable PyTorch CRNN (separable CNN, BiGRU, CTC) instead of wrapping PaddleOCR or Tesseract as the product model.

The product has to train on a CPU, stay under a few million parameters for the main recognizer, and export to ONNX without a Paddle runtime. PaddleOCR's English mobile recognizer is a strong existing system and a fair later comparison. It was not used as the implementation because its weights were not downloaded and its runtime was not installed. Tesseract was not used as the implementation because the `tesseract` binary is not on PATH here, and the goal is a small neural recognizer we can train. The benchmark calls Tesseract when the binary exists and records it as unavailable otherwise.

## What was checked

| System | License | Recognition task | Published size | CPU inference | Trainable here | Used in this repo |
| --- | --- | --- | --- | --- | --- | --- |
| wormOCR tiny | MIT (this project) | English line recognition, printable ASCII | 3,319,264 parameters, 12.66 MB FP32 weights, measured locally | Yes, PyTorch CPU | Yes | This is the implementation |
| wormOCR micro | MIT | Same task, narrower network | 1,217,744 parameters, 4.65 MB FP32 weights, measured locally | Yes | Yes | Config only until trained |
| wormOCR small | MIT | Same task, wider network | 11,292,832 parameters, 43.08 MB FP32 weights, measured locally | Yes | Yes | Config only until trained |
| Tesseract 5 LSTM | Apache-2.0 for the engine. Language data is separate. | Printed text, 100+ languages, full pages or a single line (`--psm 7`) | `eng.traineddata` in tessdata_best is 25,823,179 bytes ([file](https://github.com/tesseract-ocr/tessdata_best/blob/master/eng.traineddata)) | Yes | Not as a PyTorch model | Not installed. Benchmark marks it unavailable. |
| PP-OCRv4 English mobile recognizer (`en_PP-OCRv4_mobile_rec`) | Apache-2.0 for PaddleOCR. Confirm the weight file's terms at download time. | English and digits, line recognition | Inference package listed as 9.7 MB in the [PaddleOCR model list](https://github.com/PaddlePaddle/PaddleOCR/blob/main/docs/version2.x/ppocr/model_list.md) | Yes, through Paddle or ONNX | Fine-tuning needs PaddleOCR | Not installed. Weights not downloaded. |
| PP-OCRv4 mobile recognizer (Chinese-focused multilingual line model) | Apache-2.0 on the [official model card](https://huggingface.co/PaddlePaddle/PP-OCRv4_mobile_rec) | Chinese and English lines. The card's accuracy is line-level exact match on their set, not character error rate. | Model card lists 11 MB storage and a CPU time on their hardware | Yes | Same as above | Not downloaded. Do not compare their 83.28 line accuracy figure with wormOCR CER. |

Published sizes above are from those project pages. They were not re-measured on this machine. Local wormOCR sizes were.

## Why not a Transformer recognizer

A small attention or Transformer decoder can help on long or unusual lines, and it also adds latency, memory, and export friction. It is not in the default model. Add one only after a measured CER drop pays for the extra size and CPU time on the same held-out lines.

## How to compare fairly later

- Use the same line images for wormOCR and for Tesseract `--psm 7`. That is line recognition, not full-page OCR.
- Use the same page images, including detection, only when both systems see the page. Do not compare wormOCR's model-only latency with Tesseract's end-to-end time unless the report says which is which. `wormocr benchmark` keeps those rows separate.
- Report CER, WER, file size, CPU median and p95, and process working set. Include failures.
- PP-OCR weights are appropriate to download later for a comparison run. Do not vendor them into this repository until the license file that ships with the weights has been read.

## Local external tools on the development machine

Checked while building this tree:

- `tesseract` was not on PATH.
- The `paddleocr` package was not installed.
- PyTorch was `2.11.0+cpu`. `torch.cuda.is_available()` was false, even though the machine has an NVIDIA GPU. Training and the benchmark used the CPU build. A CUDA wheel was not installed.

The benchmark JSON from `wormocr benchmark` is the place for numbers measured on a checkpoint. See [BENCHMARKS.md](BENCHMARKS.md) after a run.
