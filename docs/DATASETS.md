# Data

The first training path is synthetic and local. Real datasets are supported as a label file. Nothing in the training code downloads them.

## Synthetic lines

`wormocr train` renders lines on the fly from installed fonts. The sample index and seed decide the text, font, and augmentation, so a run can be repeated without storing the images. Augmentation is font, size, case, polarity, blur, noise, JPEG, brightness, a light gradient, mild rotation or shear, and a downscale-upscale pass. The label is chosen before augmentation and is not rewritten.

Held-out synthetic evaluation uses a different seed and turns augmentation off. It answers "did this run learn the renderer?" It does not answer "how good is this on scans?"

```bash
python -m wormocr generate --config configs/tiny.yaml --samples 100 --out artifacts/data/train --seed 0
python -m wormocr generate --config configs/tiny.yaml --samples 20 --out artifacts/data/val --seed 1000003
```

`manifest.jsonl` has `image`, `text`, and `width`.

If `wormocr info` reports no fonts, install a Latin family (Arial, Calibri, or DejaVu Sans) or pass a directory:

```bash
python -m wormocr train --config configs/tiny.yaml --mode debug --set data.font_dir=C:/path/to/fonts
```

## Your own images

Create a UTF-8 label file. Each line is a path relative to the image root, a tab, and the exact text. Do not add a header.

```text
line001.png	Invoice 1042
line002.png	Total $18.50
```

```bash
python -m wormocr train --config configs/tiny.yaml --mode baseline --set data.labels=data/train.tsv --set data.image_root=data/images --set data.val_labels=data/val.tsv --set data.val_image_root=data/images
python -m wormocr evaluate --model artifacts/checkpoints/tiny/best.pt --labels data/val.tsv --image-root data/images
```

Characters outside the printable ASCII charset raise an error naming the character. The sample is not silently rewritten. Empty labels are rejected.

Keep training images out of the validation file. The loader does not split a file for you.

## Public datasets that were not downloaded

These are relevant later. They were not fetched in this workspace, and their terms need to be read before use.

| Dataset | Task | Why it was left alone |
| --- | --- | --- |
| [MJSynth / Synth90k](https://www.robots.ox.ac.uk/~vgg/data/text/) | About 9 million synthetic word images | Multi-gigabyte download. Useful as a later pretraining source, not as a silent dependency. |
| [IIIT 5K-Word](https://cvit.iiit.ac.in/research/projects/cvit-projects/the-iiit-5k-word-dataset) | 5,000 real word crops from scene images | Scene text, not clean document lines. Access and license should be checked at the source. The official split must be preserved. |
| Document collections such as FUNSD or DocBank | Full pages | Different task. They belong with the page pipeline, after the line recognizer is measured on lines. |

A practical next measurement is a few hundred real document lines in the TSV format above, with the same files given to wormOCR and to any external OCR you compare against.

## Robustness slices

```bash
python -m wormocr evaluate --model artifacts/checkpoints/tiny/best.pt --robustness 20 --synthetic-samples 20
```

The same texts are rendered clean, blurred, noisy, JPEG-compressed, downscaled, rotated, shadowed, and at a smaller source size. The report is still synthetic. It shows which corruption the current checkpoint handles, not a field failure rate.
