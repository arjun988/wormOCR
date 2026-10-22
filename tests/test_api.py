from pathlib import Path

import pytest

from wormocr.api import OCR, resolve_checkpoint
from wormocr.checkpoints import save_checkpoint
from wormocr.config import load_config
from wormocr.data.synthetic import generate_dataset
from wormocr.exceptions import CheckpointError, RecognitionError
from wormocr.models.crnn import build_model
from wormocr.vocabulary import Vocabulary


def _write_checkpoint(directory: Path) -> Path:
    config = load_config("configs/debug.yaml")
    vocabulary = Vocabulary()
    model = build_model(config, vocabulary.size)
    path = directory / "best.pt"
    save_checkpoint(
        path,
        {
            "model_state": model.state_dict(),
            "config": config,
            "charset": vocabulary.charset,
            "maturity": "smoke-tested",
            "step": 0,
        },
    )
    return path


def test_missing_checkpoint_explains_that_nothing_is_downloaded(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(CheckpointError) as exc:
        resolve_checkpoint("not-a-trained-variant", None)
    assert "does not download" in str(exc.value)


def test_api_predicts_a_line_and_a_page(tmp_path):
    checkpoint = _write_checkpoint(tmp_path)
    config = load_config("configs/debug.yaml")
    manifest_dir = tmp_path / "lines"
    generate_dataset(config, 1, manifest_dir, seed=1, augment=False)
    ocr = OCR(checkpoint=checkpoint, device="cpu")
    line = ocr.predict(manifest_dir / "000000.png", mode="line")
    assert isinstance(line.text, str)
    assert line.layout == "line"
    assert line.lines[0].bbox[2] > 0
    from PIL import Image, ImageDraw, ImageFont
    from wormocr.data.fonts import discover_fonts

    page = Image.new("L", (480, 180), 255)
    font = ImageFont.truetype(discover_fonts("")[0], 28)
    ImageDraw.Draw(page).text((20, 30), "Page line", font=font, fill=0)
    ImageDraw.Draw(page).text((20, 100), "Another", font=font, fill=0)
    result = ocr.predict(page, mode="page")
    assert len(result.lines) >= 1
    assert result.lines[0].bbox[1] >= 0
    assert result.image_width == 480


def test_pdf_dependency_is_explicit(tmp_path):
    checkpoint = _write_checkpoint(tmp_path)
    ocr = OCR(checkpoint=checkpoint, device="cpu")
    try:
        import pypdfium2 as pdfium
    except ImportError:
        pdf = tmp_path / "doc.pdf"
        pdf.write_bytes(b"%PDF-1.4\n")
        with pytest.raises(RecognitionError) as exc:
            ocr.predict(pdf)
        assert "pypdfium2" in str(exc.value)
        return
    pdf = tmp_path / "doc.pdf"
    document = pdfium.PdfDocument.new()
    document.new_page(200, 80)
    document.save(pdf)
    document.close()
    result = ocr.predict(pdf)
    assert result.layout == "pdf"
    assert result.pages is not None and len(result.pages) == 1
