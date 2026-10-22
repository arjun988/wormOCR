# Architecture

wormOCR's recognizer is a CRNN: a narrow convolutional encoder, a bidirectional GRU, and a CTC head. That stack is the usual lightweight baseline for line recognition. It trains on CPU, exports to ONNX, and keeps the horizontal resolution high enough for repeated and narrow characters.

## Line recognizer

Input is one grayscale line, resized to height 32 and normalized to `[-1, 1]`. Width stays proportional to the line. Batches pad on the right with the line's own border value. The CTC length uses the unpadded width, so padding is not trained as extra text.

The encoder has four stages. The first convolution sees the single input channel. Later convolutions are depthwise-separable when `model.separable` is true, and ordinary convolutions otherwise. Width is pooled by `model.width_stride`, 4 by default, so one output frame covers four input columns. Height is pooled four times, from 32 down to 2, then flattened. Stride 2 is still accepted, and old checkpoints without the key load as stride 2. At stride 2 the tiny model sat on the CTC plateau (loss near ln 96, blank everywhere) for thousands of steps. At stride 4 it left that plateau after about 500 steps. A linear layer feeds a bidirectional GRU. The classifier emits log-probabilities over the charset plus a blank at index 0.

Greedy decoding is the default. A repeated character is kept when a blank, or a different class, separates the two peaks. Beam search is available and off by default; it is only worth enabling if a measurement shows fewer errors.

The three configs differ by channel counts, GRU width, and whether the convolutions are separable:

| Config | Channels | GRU | Separable | Parameters |
| --- | --- | --- | --- | ---: |
| micro | 32, 64, 96, 128 | 128 x 2 | no | 737,408 |
| tiny | 48, 96, 160, 192 | 256 x 2 | yes | 2,174,880 |
| small | 64, 128, 256, 320 | 512 x 2 | no | 9,411,232 |

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

CTC loss uses `zero_infinity` so an impossible alignment does not become NaN. The trainer rejects a label that needs more frames than the line has, counting an extra frame wherever the same character repeats. The blank class starts with a negative bias so a new model does not emit blank at every frame. AdamW is used with no weight decay on biases and normalization parameters. The learning rate warms up linearly, then follows a cosine down to 10% of the base rate, measured over the steps of this run. A resume does not inherit a schedule that has already decayed. Gradients are clipped. With `train.ema_decay` above zero, an exponential moving average of the weights is kept, and that average is what gets validated and saved as `model_state`. The raw weights are saved as `train_state` so a resume continues from them. `best_weights.pt` holds the averaged weights only, without optimizer state. `train.eval_every` validates every N steps instead of once per epoch. Evaluation scores a label that cannot fit the line's frames as an error and leaves it out of the loss, so a narrow real crop does not abort the run. Mixed precision runs only when CUDA is available and `train.amp` is true. Dropout, when configured, sits between GRU layers and is not applied to every frame of the encoder output.

Checkpoints record maturity:

- `smoke-tested` after `--mode smoke`
- `debug-trained` after `--mode debug`, or after a baseline or resume that has not reached 500 steps
- `trained` after a baseline or resume of at least 500 steps

Those labels describe how the run was executed. They are not a claim of accuracy.

## Export

The ONNX graph is the network only: a float tensor `(batch, 1, 32, width)` in, log-probabilities `(batch, time, classes)` out. Preprocessing and CTC decoding stay in Python so the exported file is easy to compare with PyTorch. Dynamic INT8 quantization is applied with ONNX Runtime when those packages are installed. The original checkpoint is kept either way.
