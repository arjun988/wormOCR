# Architecture

wormOCR's recognizer is a CRNN: a narrow convolutional encoder, a bidirectional GRU, and a CTC head. That stack is the usual lightweight baseline for line recognition. It trains on CPU, exports to ONNX, and keeps the horizontal resolution high enough for repeated and narrow characters.

## Line recognizer

Input is one grayscale line, resized to height 32 and normalized to `[-1, 1]`. Width stays proportional to the line. Batches pad on the right with the line's own border value. The CTC length uses the unpadded width, so padding is not trained as extra text.

The encoder has four stages. The first convolution sees the single input channel. Later convolutions are depthwise-separable when `model.separable` is true, and ordinary convolutions otherwise. Width is pooled once, by 2. Height is pooled four times, from 32 down to 2, then flattened. A linear layer feeds a bidirectional GRU. The classifier emits log-probabilities over the charset plus a blank at index 0.

Greedy decoding is the default. A repeated character is kept when a blank, or a different class, separates the two peaks. Beam search is available and off by default; it is only worth enabling if a measurement shows fewer errors.

The three configs differ by channel counts, GRU width, and whether the convolutions are separable:

| Config | Channels | GRU | Separable | Parameters |
| --- | --- | --- | --- | ---: |
| micro | 32, 48, 96, 128 | 192 x 2 | yes | 1,217,744 |
| tiny | 48, 96, 160, 192 | 320 x 2 | yes | 3,319,264 |
| small | 64, 128, 256, 320 | 576 x 2 | no | 11,292,832 |

Parameter counts were measured by instantiating each config with the 96-class vocabulary. They are not estimates from a formula.

## Page pass

The page pass is not a neural detector.

1. Optional deskew, searched at half-degree steps inside ±5 degrees. The angle is kept only when it clearly sharpens the horizontal projection.
2. Otsu ink mask, then horizontal bands of ink.
3. A wide horizontal gap splits a band into columns. Ordinary word spaces do not.
4. Reading order walks down a column before moving to the next. A line that spans most of the width, such as a title, is emitted at its vertical position instead of being forced into one column.
5. Each crop is recognized by the line model. Boxes are mapped back through the same affine Pillow uses to rotate with `expand=True`.

Coordinates in `OCRResult.lines` are axis-aligned boxes on the original image. After deskew, a box is the upright bounds of the rotated crop, so it can be a little loose.

## Training

CTC loss uses `zero_infinity` so an impossible alignment does not become NaN, and the trainer still rejects a label that is longer than the line's time axis. AdamW is used with no weight decay on biases and normalization parameters. The learning rate warms up linearly, then follows a cosine down to 10% of the base rate. Gradients are clipped. Mixed precision runs only when CUDA is available and `train.amp` is true.

Checkpoints record maturity:

- `smoke-tested` after `--mode smoke`
- `debug-trained` after `--mode debug`, or after a baseline that has not reached 500 steps
- `trained` after a baseline of at least 500 steps

Those labels describe how the run was executed. They are not a claim of accuracy.

## Export

The ONNX graph is the network only: a float tensor `(batch, 1, 32, width)` in, log-probabilities `(batch, time, classes)` out. Preprocessing and CTC decoding stay in Python so the exported file is easy to compare with PyTorch. Dynamic INT8 quantization is applied with ONNX Runtime when those packages are installed. The original checkpoint is kept either way.
