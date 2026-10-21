from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from wormocr.exceptions import RecognitionError
from wormocr.preprocessing import coerce_image, load_image, prepare_line


def test_variable_width_keeps_height_and_aspect():
    narrow = Image.new("L", (20, 40), 255)
    wide = Image.new("L", (90, 40), 255)
    narrow_tensor, _pad = prepare_line(narrow, height=32, align=2)
    wide_tensor, _pad = prepare_line(wide, height=32, align=2)
    assert narrow_tensor.shape[0] == 1
    assert narrow_tensor.shape[1] == 32
    assert wide_tensor.shape[1] == 32
    assert wide_tensor.shape[2] > narrow_tensor.shape[2]
    assert narrow_tensor.shape[2] % 2 == 0
    assert float(narrow_tensor.min()) >= -1.01
    assert float(narrow_tensor.max()) <= 1.01


def test_empty_image_is_rejected():
    with pytest.raises(ValueError):
        prepare_line(Image.new("L", (0, 10), 255), height=32)


def test_missing_and_malformed_files(tmp_path: Path):
    missing = tmp_path / "nope.png"
    with pytest.raises(RecognitionError):
        load_image(missing)
    bogus = tmp_path / "notes.png"
    bogus.write_text("this is not an image", encoding="utf-8")
    with pytest.raises(RecognitionError):
        load_image(bogus)


def test_numpy_and_rgba_inputs():
    array = np.full((16, 40, 3), 255, dtype=np.uint8)
    array[:, 5:8, :] = 0
    tensor, _pad = prepare_line(coerce_image(array), height=32)
    assert tensor.shape[1] == 32
    rgba = Image.new("RGBA", (30, 20), (0, 0, 0, 0))
    tensor, _pad = prepare_line(rgba, height=32)
    assert torch.is_floating_point(tensor)
