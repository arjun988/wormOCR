"""ONNX export and dynamic INT8 quantization."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from wormocr.exceptions import ExportError


def export_onnx(model: torch.nn.Module, path: str | Path, opset: int = 17) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    height = int(model.image_height)
    dummy = torch.zeros(1, 1, height, 128, dtype=torch.float32)
    try:
        torch.onnx.export(
            model,
            dummy,
            str(destination),
            input_names=["image"],
            output_names=["log_probs"],
            dynamic_axes={"image": {0: "batch", 3: "width"}, "log_probs": {0: "batch", 1: "time"}},
            opset_version=int(opset),
            dynamo=False,
        )
    except TypeError:
        torch.onnx.export(
            model,
            dummy,
            str(destination),
            input_names=["image"],
            output_names=["log_probs"],
            dynamic_axes={"image": {0: "batch", 3: "width"}, "log_probs": {0: "batch", 1: "time"}},
            opset_version=int(opset),
        )
    if not destination.exists() or destination.stat().st_size == 0:
        raise ExportError(f"ONNX export did not write {destination}")
    return destination


def quantize_onnx(source: str | Path, destination: str | Path) -> Path:
    try:
        from onnxruntime.quantization import QuantType, quantize_dynamic
    except ImportError as exc:
        raise ExportError(
            "INT8 export needs the optional packages onnx and onnxruntime. "
            "Install them with: pip install \"wormocr[export]\""
        ) from exc
    output = Path(destination)
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        quantize_dynamic(str(source), str(output), weight_type=QuantType.QInt8)
    except Exception as exc:
        raise ExportError(f"INT8 quantization failed: {exc}") from exc
    return output


def onnx_max_abs_diff(model: torch.nn.Module, path: str | Path, width: int = 160, batch: int = 2) -> float:
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise ExportError("onnxruntime is not installed") from exc
    model.eval()
    sample = torch.randn(batch, 1, int(model.image_height), width)
    with torch.no_grad():
        reference = model(sample).detach().cpu().numpy()
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    output = session.run(None, {"image": sample.numpy().astype(np.float32)})[0]
    if output.shape != reference.shape:
        raise ExportError(f"ONNX shape {output.shape} != PyTorch shape {reference.shape}")
    return float(np.max(np.abs(reference - output)))
