# Larger model, same data

wormocr-2M stays as it is. This plan is the next model: more parameters, the same training files, the same character set, and the same test slices. The release name is `wormocr-<counted millions>`, set after `count_parameters` on the real config. It is not written over the `wormocr-2M` tag.

The point of the extra capacity is the slices that are still weak. Names (5.3% character error) and CORD receipts (3.7%) have little room. GNHK (23.3%), IAM (22.4%), Bentham (15.1%), and RxHand (37.9% case-insensitive) are where a larger network can move the number. A larger network does not create new writers or new page styles, so those four are the success test, not a promise that every set drops near zero.

## What does not change

- GPU only. The config sets `device: cuda` and training stops if CUDA is missing. The machine is an RTX 5060 Laptop, 8 GB.
- Same charset: 95 printable ASCII characters plus the CTC blank.
- Same trainable files, and the same ban on training data: IAM, RxHand, and FUNSD stay evaluation-only.
- Same frozen test slices and the same greedy decoder, so the comparison with wormocr-2M is the one already recorded.
- No PyPI and no Hugging Face unless that is asked for separately.

## Data the 2M run actually used

Real lines that entered training (the train split only):

| Set | Train lines | Role |
| --- | ---: | --- |
| Handwritten names, after the printed-label filter | 209,956 | training |
| Bentham transcribed pages | 9,418 | training |
| GNHK notes | 7,305 | training |
| CORD receipt lines | 6,687 | training |
| **Train total** | **233,366** | |

The image folders for those four sets, including their validation and test crops, are 5,405,048,708 bytes (5.41 GB). Validation and test were not trained on. Synthetic print and synthetic handwriting were rendered during training and are not stored.

The 60,000-step run drew 3,840,000 samples (batch 64). The mix was 30% synthetic print, 30% synthetic handwriting, 15% GNHK, 10% names, 10% Bentham, 5% CORD. GNHK, Bentham, and CORD were repeated many times. Most name images were seen once or not at all, because that set is large and its mix weight is 10%.

The first larger run keeps this mix, so a gain is from capacity rather than from a new sampling trick. A later run may raise the name weight only after the same-mix comparison exists.

## Model

wormocr-2M is 2,174,880 parameters: separable channels `[48, 96, 160, 192]`, projection 256, 2-layer BiGRU with hidden 256, image height 32, width stride 4.

The candidate, counted with the same CRNN code and 96 classes (blank plus 95 characters):

| | wormocr-2M | Candidate |
| --- | --- | --- |
| Parameters | 2,174,880 | 8,776,416 |
| Image height | 32 | 48 |
| Channels | 48, 96, 160, 192 | 96, 192, 320, 384 |
| Projection | 256 | 512 |
| BiGRU | 2 × 256 | 2 × 512 |
| Width stride | 4 | 4 |
| Separable convolutions | yes | yes |

8,776,416 parameters rounds the same way 2,174,880 became wormocr-2M, so the release name is **wormocr-9M** unless the built module counts differently. Height 48 is the other half of the change: handwriting strokes survive the encoder better than they do at height 32.

Almost no tensor shape matches wormocr-2M, so this run starts from a fresh initialization. Partial weight copy would leave the GRU and the projection random anyway.

A second, larger option (13,060,064 parameters, hidden 640, channels `[128, 256, 384, 512]`, height 48) waits until the 9M run has a measured step time and a memory reading. It is not the first training job.

## Training on 8 GB

- Mixed precision on, as in the 2M run.
- Start at batch 16. If that fits with room, try 24. If it does not, use 8. Gradient accumulation makes the optimizer step match an effective batch of 64, which is what wormocr-2M used.
- 14 data workers, 12 torch threads, EMA 0.999, gradient clip 5, AdamW, the same cosine schedule.
- 100,000 steps, or stop after 8 validations with no character-error improvement on the mixed validation set. `max_minutes` is a safety cap at 12 hours, not a target.
- The first 200 steps are timed. That measurement, not a guess, decides whether 100,000 steps fits in the cap.
- Checkpoints go to `artifacts/checkpoints/wormocr-9M/`. wormocr-2M files are not opened for writing.

Config file when this plan is executed: `configs/wormocr-9M.yaml`. Same `data.sources` as `configs/unified.yaml`.

## When it is better

Evaluate `best.pt` with `scripts/eval_slices.py` on the same slices. Keep the model only if all of these are true:

- GNHK, Bentham, and IAM character error are each at least 3 points lower than wormocr-2M (23.3%, 15.1%, 22.4%).
- Printed synthetic, FUNSD, and CORD do not get worse by more than 1 point (3.1%, 11.5%, 3.7%).
- Names do not get worse (5.3%).

RxHand is reported, not gated: it was never trained on, and a drop there is a bonus rather than the pass mark.

If the 9M model misses that bar, it is not released and the 13M option is not started. The next change would be training procedure on the 9M network (longer run, or a higher name weight), not another architecture.
