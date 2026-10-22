# Data

The first training path is synthetic and local. Real datasets are supported as a label file. Nothing in the training code downloads them.

## Synthetic lines

`wormocr train` renders lines on the fly from installed fonts. The sample index and seed decide the text, font, and augmentation, so a run can be repeated without storing the images. Most lines are one to three words, with form-style strings mixed in: amounts such as `$1,234.56`, phone numbers, `#123`, and `Field:` labels. About 40% of lines get tight padding, as word boxes on real forms do. About 30% of augmented samples are left clean. The rest receive one or two operations: blur, noise, JPEG, tone or a light gradient, a downscale pass, a small rotation or shear, bolder or thinner strokes, binarization, or a "scan" pass that shrinks the line to 10–18 px tall, adds noise and JPEG, and scales it back. The label is chosen before augmentation and is not rewritten.

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

## FUNSD (evaluation only)

[FUNSD](https://guillaumejaume.github.io/FUNSD/) has 199 scanned forms with word boxes. Its license allows non-commercial research use only, so it is used here only to measure the model. Released weights are not trained on it. The archive is not committed.

```bash
python scripts/prepare_funsd.py artifacts/datasets/funsd/dataset/testing_data artifacts/datasets/funsd_test
python -m wormocr evaluate --model artifacts/checkpoints/tiny/best.pt --labels artifacts/datasets/funsd_test/labels.tsv
```

The script crops each word box from the official test split with 2 px padding. It skips words with characters outside the charset and boxes under 8 px tall. The archive used had SHA-256 `C31735649E4F441BCBB4FD0F379574F7520B42286E80B01D80B445649D54761F`. It produced 8,526 words, skipping 57 for the charset and 124 for size.

## Public datasets that were not downloaded

These are relevant later. They were not fetched in this workspace, and their terms need to be read before use.

| Dataset | Task | Why it was left alone |
| --- | --- | --- |
| [MJSynth / Synth90k](https://www.robots.ox.ac.uk/~vgg/data/text/) | About 9 million synthetic word images | Multi-gigabyte download. Useful as a later pretraining source, not as a silent dependency. |
| [IIIT 5K-Word](https://cvit.iiit.ac.in/research/projects/cvit-projects/the-iiit-5k-word-dataset) | 5,000 real word crops from scene images | Scene text, not clean document lines. Access and license should be checked at the source. The official split must be preserved. |
| DocBank and similar page collections | Full pages | Different task. They belong with the page pipeline. |

A practical next measurement is a few hundred real document lines in the TSV format above, with the same files given to wormOCR and to any external OCR you compare against.

## Robustness slices

```bash
python -m wormocr evaluate --model artifacts/checkpoints/tiny/best.pt --robustness 20 --synthetic-samples 20
```

The same texts are rendered clean, blurred, noisy, JPEG-compressed, downscaled, rotated, shadowed, and at a smaller source size. The report is still synthetic. It shows which corruption the current checkpoint handles, not a field failure rate.
