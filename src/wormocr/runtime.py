"""Device selection and a small environment report."""

from __future__ import annotations

import os
import platform
import shutil

import torch

from wormocr.exceptions import WormOCRError


def resolve_device(name: str | None) -> torch.device:
    requested = (name or "auto").strip().lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise WormOCRError(
            "CUDA was requested, but this PyTorch installation cannot see a GPU. "
            f"torch {torch.__version__} is the build in use. Run with --device cpu."
        )
    try:
        return torch.device(requested)
    except (TypeError, RuntimeError) as exc:
        raise WormOCRError(f"unknown device {name!r}") from exc


def environment_report() -> dict:
    report: dict = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu": platform.processor() or os.environ.get("PROCESSOR_IDENTIFIER", "unknown"),
        "cpu_count": os.cpu_count(),
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_build": torch.version.cuda,
        "torch_threads": int(torch.get_num_threads()),
        "mkldnn": bool(torch.backends.mkldnn.is_available()),
    }
    if torch.cuda.is_available():
        report["gpu"] = torch.cuda.get_device_name(0)
    try:
        import onnxruntime

        report["onnxruntime"] = onnxruntime.__version__
    except ImportError:
        report["onnxruntime"] = None
    try:
        import onnx

        report["onnx"] = onnx.__version__
    except ImportError:
        report["onnx"] = None
    report["tesseract"] = shutil.which("tesseract")
    return report
