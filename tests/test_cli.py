import json
from pathlib import Path

import pytest

from wormocr.cli import main


def test_help_and_info():
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    for command in ("train", "predict", "evaluate", "export", "benchmark"):
        with pytest.raises(SystemExit) as exc:
            main([command, "--help"])
        assert exc.value.code == 0
    assert main(["info"]) == 0


def test_predict_without_weights_fails(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    image = tmp_path / "blank.png"
    from PIL import Image

    Image.new("L", (32, 16), 255).save(image)
    assert main(["predict", "--image", str(image), "--model", "tiny"]) == 1


def test_cli_smoke_evaluate_and_generate(tmp_path):
    checkpoint_dir = tmp_path / "ckpt"
    code = main(
        [
            "train",
            "--config",
            "configs/debug.yaml",
            "--mode",
            "smoke",
            "--set",
            f"train.checkpoint_dir={checkpoint_dir.as_posix()}",
        ]
    )
    assert code == 0
    best = checkpoint_dir / "best.pt"
    assert best.exists()
    out = tmp_path / "lines"
    assert main(["generate", "--config", "configs/debug.yaml", "--samples", "2", "--out", str(out), "--seed", "1"]) == 0
    assert (out / "manifest.jsonl").exists()
    assert main(["evaluate", "--model", str(best), "--synthetic-samples", "4", "--seed", "2"]) == 0
    errors = tmp_path / "errors.md"
    assert main(["analyze", "--model", str(best), "--synthetic-samples", "4", "--output", str(errors)]) == 0
    assert "Error" in errors.read_text(encoding="utf-8") or "error" in errors.read_text(encoding="utf-8")
    image = next(out.glob("*.png"))
    assert main(["predict", "--image", str(image), "--checkpoint", str(best), "--json"]) == 0
    metrics = json.loads((checkpoint_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["maturity"] == "smoke-tested"
