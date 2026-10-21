from pathlib import Path

import pytest
import torch

from wormocr.checkpoints import save_checkpoint
from wormocr.config import load_config
from wormocr.export.onnx_export import export_onnx, onnx_max_abs_diff, quantize_onnx
from wormocr.inference.decode import greedy_decode
from wormocr.models.crnn import build_model
from wormocr.vocabulary import Vocabulary

pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")


def test_onnx_matches_pytorch_and_int8_runs(tmp_path: Path):
    config = load_config("configs/debug.yaml")
    vocabulary = Vocabulary()
    model = build_model(config, vocabulary.size)
    model.eval()
    fp32 = tmp_path / "model.onnx"
    export_onnx(model, fp32, opset=17)
    diff = onnx_max_abs_diff(model, fp32, width=96, batch=2)
    assert diff < 1e-3
    int8 = tmp_path / "model.int8.onnx"
    quantize_onnx(fp32, int8)
    assert int8.exists()
    assert int8.stat().st_size > 0
    sample = torch.randn(1, 1, 32, 96)
    with torch.no_grad():
        reference = greedy_decode(model(sample), blank=0)[0]
    import onnxruntime as ort

    session = ort.InferenceSession(str(int8), providers=["CPUExecutionProvider"])
    output = session.run(None, {"image": sample.numpy()})[0]
    decoded = greedy_decode(torch.from_numpy(output), blank=0)[0]
    assert isinstance(reference, list)
    assert isinstance(decoded, list)
    checkpoint = tmp_path / "best.pt"
    save_checkpoint(
        checkpoint,
        {
            "model_state": model.state_dict(),
            "config": config,
            "charset": vocabulary.charset,
            "maturity": "smoke-tested",
        },
    )
