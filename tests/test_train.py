import pytest
import torch
from PIL import Image, ImageDraw, ImageFont

from wormocr.checkpoints import load_checkpoint
from wormocr.config import load_config
from wormocr.data.fonts import discover_fonts
from wormocr.inference.decode import greedy_decode
from wormocr.models.crnn import CRNN, build_model
from wormocr.preprocessing import prepare_line
from wormocr.training.loop import compute_loss, maturity_for, schedule_span, train
from wormocr.vocabulary import Vocabulary


def test_resume_schedule_is_counted_from_zero():
    local = schedule_span(128, epochs=6, batches_per_epoch=64, max_steps=None)
    assert local == 384
    from wormocr.training.loop import learning_rate

    assert learning_rate(0, local, warmup=20, base_lr=0.001, min_ratio=0.1) == pytest.approx(0.00005)


def test_maturity_stays_debug_until_five_hundred_steps():
    assert maturity_for("smoke", 2) == "smoke-tested"
    assert maturity_for("debug", 40) == "debug-trained"
    assert maturity_for("baseline", 64) == "debug-trained"
    assert maturity_for("resume", 256) == "debug-trained"
    assert maturity_for("baseline", 500) == "trained"
    assert maturity_for("resume", 500) == "trained"


def test_ema_checkpoint_keeps_raw_weights_for_resume(tmp_path):
    config = load_config(
        "configs/debug.yaml",
        [
            f"train.checkpoint_dir={tmp_path.as_posix()}",
            "train.max_steps=4",
            "train.ema_decay=0.9",
            "train.eval_every=2",
            "data.train_samples=16",
            "data.val_samples=4",
            "data.augment=false",
            "train.epochs=2",
        ],
        mode="baseline",
    )
    result = train(config, mode="baseline")
    assert result.steps == 4
    payload = load_checkpoint(result.last_checkpoint)
    assert "train_state" in payload
    assert len(payload["history"]) == 2
    weights = load_checkpoint(tmp_path / "best_weights.pt")
    assert "train_state" not in weights and "optimizer_state" not in weights


def test_evaluation_scores_a_crop_too_narrow_for_its_label():
    from wormocr.data.dataset import collate_lines
    from wormocr.training.loop import evaluate_model

    vocabulary = Vocabulary()
    model = CRNN(num_classes=vocabulary.size, channels=(8, 16, 16, 16), rnn_hidden=16, width_stride=4)
    samples = [
        {"image": torch.zeros(1, 32, 8), "text": "82092117", "pad_value": 0.0},
        {"image": torch.zeros(1, 32, 64), "text": "ok", "pad_value": 0.0},
    ]
    totals, _ = evaluate_model(model, [collate_lines(samples)], vocabulary, torch.device("cpu"))
    assert totals.samples == 2
    assert totals.character_edits >= 1


def test_smoke_saves_and_resumes(tmp_path):
    config = load_config(
        "configs/debug.yaml",
        [f"train.checkpoint_dir={tmp_path.as_posix()}", "train.max_steps=2", "data.train_samples=8", "data.val_samples=4"],
        mode="smoke",
    )
    result = train(config, mode="smoke")
    assert result.steps == 2
    assert result.maturity == "smoke-tested"
    payload = load_checkpoint(result.best_checkpoint)
    assert "optimizer_state" in payload
    resumed_dir = tmp_path / "resume"
    resumed = load_config(
        "configs/debug.yaml",
        [
            f"train.checkpoint_dir={resumed_dir.as_posix()}",
            "train.max_steps=4",
            "data.train_samples=8",
            "data.val_samples=4",
            "train.epochs=2",
        ],
        mode="resume",
    )
    continued = train(resumed, mode="resume", resume_from=result.last_checkpoint)
    assert continued.steps == 4
    again = load_checkpoint(continued.last_checkpoint)
    model = build_model(again["config"], Vocabulary(again["charset"]).size)
    model.load_state_dict(again["model_state"])
    assert count_finite(model)


def test_clean_lines_can_be_memorized():
    vocabulary = Vocabulary()
    font = ImageFont.truetype(discover_fonts("")[0], 28)
    items = []
    for text in ("a", "bb", "cat", "HI"):
        probe = ImageDraw.Draw(Image.new("L", (1, 1)))
        box = probe.textbbox((0, 0), text, font=font)
        canvas = Image.new("L", (box[2] - box[0] + 12, box[3] - box[1] + 10), 255)
        ImageDraw.Draw(canvas).text((6 - box[0], 4 - box[1]), text, font=font, fill=0)
        tensor, pad = prepare_line(canvas, height=32, align=2, max_width=256)
        items.append({"image": tensor, "text": text, "pad_value": pad, "width": int(tensor.shape[-1])})
    model = CRNN(
        num_classes=vocabulary.size,
        channels=[32, 32, 64, 64],
        separable=False,
        rnn_hidden=128,
        rnn_layers=1,
        rnn_dropout=0.0,
        proj_size=128,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    widths = torch.tensor([item["width"] for item in items])
    max_width = int(widths.max())
    batch = torch.zeros(len(items), 1, 32, max_width)
    for index, item in enumerate(items):
        width = item["width"]
        batch[index, :, :, :width] = item["image"]
        if width < max_width:
            batch[index, :, :, width:] = item["pad_value"]
    texts = [item["text"] for item in items]
    matched = False
    for _step in range(160):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss, _log_probs, lengths = compute_loss(model, batch, texts, widths, vocabulary)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        if _step % 20 == 19:
            model.eval()
            with torch.no_grad():
                decoded = [
                    vocabulary.decode(sequence)
                    for sequence in greedy_decode(model(batch), lengths, vocabulary.blank)
                ]
            matched = decoded == texts
            if matched:
                break
    assert matched, "the recognizer did not memorize four clean lines"


def count_finite(model) -> bool:
    return all(torch.isfinite(parameter).all() for parameter in model.parameters())
