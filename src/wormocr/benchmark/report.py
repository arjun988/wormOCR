"""Measure size, accuracy, latency, and memory for a checkpoint."""

from __future__ import annotations

import importlib.util
import json
import math
import os
import platform
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import torch

from wormocr.data.dataset import OnlineSynthDataset
from wormocr.evaluation.metrics import MetricTotals
from wormocr.export.onnx_export import export_onnx, onnx_max_abs_diff, quantize_onnx
from wormocr.inference.engine import LineEngine
from wormocr.models.crnn import build_model, count_parameters, state_dict_num_bytes
from wormocr.runtime import environment_report
from wormocr.training.loop import evaluate_model, make_loader


def working_set_bytes() -> int | None:
    if os.name == "nt":
        import ctypes

        class Counters(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_ulong),
                ("PageFaultCount", ctypes.c_ulong),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.POINTER(Counters), ctypes.c_ulong]
        psapi.GetProcessMemoryInfo.restype = ctypes.c_int
        counters = Counters()
        counters.cb = ctypes.sizeof(Counters)
        handle = kernel32.GetCurrentProcess()
        if not psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
            return None
        return int(counters.WorkingSetSize)
    try:
        import resource

        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if platform.system() == "Darwin":
            return int(usage)
        return int(usage) * 1024
    except Exception:
        return None


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * fraction
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    return ordered[low] * (high - position) + ordered[high] * (position - low)


def time_calls(function, warmup: int, repeats: int) -> dict:
    for _ in range(warmup):
        function()
    before = working_set_bytes()
    peak = before or 0
    samples: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        function()
        samples.append(time.perf_counter() - start)
        current = working_set_bytes()
        if current is not None:
            peak = max(peak, current)
    median = _percentile(samples, 0.5)
    return {
        "median_ms": median * 1000.0,
        "p95_ms": _percentile(samples, 0.95) * 1000.0,
        "repeats": repeats,
        "warmup": warmup,
        "working_set_before_bytes": before,
        "working_set_peak_bytes": peak if before is not None else None,
        "working_set_delta_bytes": None if before is None else max(0, peak - before),
        "throughput_per_sec": (1.0 / median) if median > 0 else None,
    }


def run_benchmark(
    checkpoint: str | Path,
    output: str | Path,
    samples: int = 40,
    seed: int = 12345,
    repeats: int = 20,
    warmup: int = 5,
    device: str | None = "cpu",
    compare_untrained: bool = True,
) -> dict:
    started = time.perf_counter()
    path = Path(checkpoint)
    engine = LineEngine.load(path, device=device, beam_width=0)
    model = engine.model
    parameters = count_parameters(model)
    report: dict = {
        "checkpoint": str(path),
        "maturity": engine.maturity,
        "parameters": parameters,
        "fp32_weight_bytes": state_dict_num_bytes(model),
        "checkpoint_bytes": path.stat().st_size,
        "environment": environment_report(),
        "device": str(engine.device),
        "input_height": engine.image_height,
        "notes": [
            "Latency is measured after warmup.",
            "Working set is the process working set, including Python and PyTorch, not model weights alone.",
            "Synthetic evaluation is not a real-document benchmark.",
        ],
    }
    weights_path = path.with_name("best_weights.pt")
    if weights_path.exists():
        report["weights_only_bytes"] = weights_path.stat().st_size

    startup = time.perf_counter()
    LineEngine.load(path, device=device)
    report["startup_ms"] = (time.perf_counter() - startup) * 1000.0

    height = engine.image_height
    tensor = torch.randn(1, 1, height, 256)
    batch_tensor = torch.randn(8, 1, height, 256)

    def model_one() -> None:
        with torch.no_grad():
            model(tensor.to(engine.device))

    def model_batch() -> None:
        with torch.no_grad():
            model(batch_tensor.to(engine.device))

    report["latency"] = {
        "model_only_batch1_width256": time_calls(model_one, warmup, repeats),
        "model_only_batch8_width256": time_calls(model_batch, warmup, repeats),
    }
    report["latency"]["model_only_batch8_width256"]["throughput_per_sec"] = (
        8.0 / (report["latency"]["model_only_batch8_width256"]["median_ms"] / 1000.0)
        if report["latency"]["model_only_batch8_width256"]["median_ms"]
        else None
    )

    dataset = OnlineSynthDataset(engine.config, max(1, samples), seed, augment=False)
    sample_image = _first_image(dataset)
    batch_images = [_sample_image(dataset, index) for index in range(min(8, len(dataset)))]

    def e2e_one() -> None:
        engine.recognize([sample_image])

    def e2e_batch() -> None:
        engine.recognize(batch_images)

    report["latency"]["end_to_end_line"] = time_calls(e2e_one, min(3, warmup), repeats)
    report["latency"]["end_to_end_batch8"] = time_calls(e2e_batch, min(2, warmup), max(5, repeats // 2))
    if report["latency"]["end_to_end_batch8"]["median_ms"]:
        report["latency"]["end_to_end_batch8"]["throughput_per_sec"] = len(batch_images) / (
            report["latency"]["end_to_end_batch8"]["median_ms"] / 1000.0
        )

    loader = make_loader(dataset, batch_size=8, shuffle=False, seed=seed, num_workers=0)
    totals, _previews = evaluate_model(model, loader, engine.vocabulary, engine.device, collect=False)
    report["synthetic_held_out"] = totals.to_dict()
    report["synthetic_held_out"]["seed"] = seed
    report["synthetic_held_out"]["augment"] = False
    report["synthetic_held_out"]["includes_text_detection"] = False

    if compare_untrained:
        fresh = build_model(engine.config, engine.vocabulary.size).to(engine.device)
        fresh.eval()
        untrained, _unused = evaluate_model(fresh, loader, engine.vocabulary, engine.device)
        report["untrained_same_architecture"] = untrained.to_dict()

    report["external"] = {
        "tesseract": _tesseract_status(dataset, min(samples, 20)),
        "paddleocr": _paddle_status(),
    }
    report["export"] = _try_export(model, Path(output).parent)
    report["elapsed_sec"] = time.perf_counter() - started
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    destination.write_text(render_markdown(report), encoding="utf-8")
    return report


def _first_image(dataset: OnlineSynthDataset):
    return _sample_image(dataset, 0)


def _sample_image(dataset: OnlineSynthDataset, index: int):
    from wormocr.data.synthetic import _tensor_to_pil

    return _tensor_to_pil(dataset[index]["image"])


def _tesseract_status(dataset: OnlineSynthDataset, limit: int) -> dict:
    binary = shutil.which("tesseract")
    if not binary:
        return {
            "available": False,
            "reason": "tesseract is not on PATH. It was not installed for this run.",
        }
    totals = MetricTotals()
    durations: list[float] = []
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)
        for index in range(min(limit, len(dataset))):
            item = dataset[index]
            image = _sample_image(dataset, index)
            image_path = folder / f"{index}.png"
            image.save(image_path)
            start = time.perf_counter()
            completed = subprocess.run(
                [binary, str(image_path), "stdout", "--psm", "7", "-l", "eng"],
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )
            durations.append(time.perf_counter() - start)
            if completed.returncode != 0:
                return {
                    "available": False,
                    "reason": completed.stderr.strip() or "tesseract failed",
                }
            hypothesis = completed.stdout.replace("\n", "").strip()
            totals.update(item["text"], hypothesis)
    return {
        "available": True,
        "mode": "single text line, psm 7, English. Detection is not included.",
        "metrics": totals.to_dict(),
        "median_ms": _percentile(durations, 0.5) * 1000.0,
        "p95_ms": _percentile(durations, 0.95) * 1000.0,
    }


def _paddle_status() -> dict:
    if importlib.util.find_spec("paddleocr") is None:
        return {
            "available": False,
            "reason": (
                "PaddleOCR is not installed and its weights were not downloaded. "
                "Published PP-OCRv4 mobile recognizer sizes are listed in docs/BASELINE_COMPARISON.md."
            ),
        }
    return {
        "available": False,
        "reason": "PaddleOCR is installed, but wormOCR does not auto-download or run its weights in this benchmark.",
    }


def _try_export(model, directory: Path) -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    fp32 = directory / "benchmark.fp32.onnx"
    try:
        export_onnx(model, fp32, opset=int(model_opset(model)))
        diff = onnx_max_abs_diff(model, fp32)
        result = {
            "onnx_fp32_bytes": fp32.stat().st_size,
            "onnx_fp32_max_abs_diff": diff,
        }
    except Exception as exc:
        return {"available": False, "reason": str(exc)}
    int8 = directory / "benchmark.int8.onnx"
    try:
        quantize_onnx(fp32, int8)
        result["onnx_int8_bytes"] = int8.stat().st_size
        result["onnx_int8_max_abs_diff"] = onnx_max_abs_diff(model, int8)
    except Exception as exc:
        result["onnx_int8_reason"] = str(exc)
    result["available"] = True
    return result


def model_opset(_model) -> int:
    return 17


def render_markdown(report: dict) -> str:
    env = report["environment"]
    latency = report["latency"]
    synthetic = report["synthetic_held_out"]
    lines = [
        "# wormOCR benchmark",
        "",
        "Generated from one local run. Do not treat synthetic scores as document-OCR quality.",
        "",
        "## Environment",
        "",
        f"- platform: {env.get('platform')}",
        f"- cpu: {env.get('cpu')}",
        f"- python: {env.get('python')}",
        f"- torch: {env.get('torch')} (threads {env.get('torch_threads')})",
        f"- cuda available: {env.get('cuda_available')}",
        f"- device used: {report.get('device')}",
        f"- checkpoint maturity: {report.get('maturity')}",
        "",
        "## Size",
        "",
        f"- parameters: {report['parameters']:,}",
        f"- FP32 weights: {_fmt_mb(report['fp32_weight_bytes'])}",
        f"- checkpoint file: {_fmt_mb(report['checkpoint_bytes'])}",
        f"- startup: {report.get('startup_ms', 0):.1f} ms",
        "",
        "## Synthetic held-out lines",
        "",
        "Text detection is not included. Images are generated text lines.",
        "",
        f"- samples: {synthetic['samples']}",
        f"- CER: {synthetic['cer']:.4f}",
        f"- WER: {synthetic['wer']:.4f}",
        f"- exact match: {synthetic['exact_match']:.4f}",
        "",
    ]
    untrained = report.get("untrained_same_architecture")
    if untrained:
        lines.extend(
            [
                "Random initialization of the same architecture, same images:",
                "",
                f"- CER: {untrained['cer']:.4f}",
                f"- WER: {untrained['wer']:.4f}",
                f"- exact match: {untrained['exact_match']:.4f}",
                "",
            ]
        )
    lines.extend(["## Latency", "", "| path | batch | median ms | p95 ms | throughput / s |", "| --- | ---: | ---: | ---: | ---: |"])
    labels = {
        "model_only_batch1_width256": (1, "model only, width 256"),
        "model_only_batch8_width256": (8, "model only, width 256"),
        "end_to_end_line": (1, "preprocess + model + greedy decode"),
        "end_to_end_batch8": (8, "preprocess + model + greedy decode"),
    }
    for key, (batch, label) in labels.items():
        item = latency.get(key)
        if not item:
            continue
        throughput = item.get("throughput_per_sec")
        rate = f"{throughput:.2f}" if throughput else ""
        lines.append(
            f"| {label} | {batch} | {item['median_ms']:.2f} | {item['p95_ms']:.2f} | {rate} |"
        )
    peak = latency["model_only_batch1_width256"].get("working_set_peak_bytes")
    lines.extend(
        [
            "",
            f"Process working set while timing the single-line model call peaked at {_fmt_mb(peak)}.",
            "",
            "## External baselines",
            "",
        ]
    )
    for name, status in report.get("external", {}).items():
        if not status.get("available"):
            lines.append(f"- {name}: unavailable. {status.get('reason')}")
            continue
        metrics = status.get("metrics") or {}
        lines.append(
            f"- {name}: CER {metrics.get('cer', float('nan')):.4f}, "
            f"median {status.get('median_ms', float('nan')):.1f} ms. {status.get('mode', '')}"
        )
    export = report.get("export") or {}
    lines.extend(["", "## Export", ""])
    if not export.get("available"):
        lines.append(f"ONNX export was not measured. {export.get('reason', '')}")
    else:
        lines.append(f"- FP32 ONNX: {_fmt_mb(export.get('onnx_fp32_bytes'))}, max abs diff {export.get('onnx_fp32_max_abs_diff')}")
        if export.get("onnx_int8_bytes"):
            lines.append(
                f"- INT8 ONNX: {_fmt_mb(export.get('onnx_int8_bytes'))}, max abs diff {export.get('onnx_int8_max_abs_diff')}"
            )
        if export.get("onnx_int8_reason"):
            lines.append(f"- INT8: {export['onnx_int8_reason']}")
    lines.extend(["", f"Benchmark wall time: {report.get('elapsed_sec', 0):.1f} s", ""])
    return "\n".join(lines)


def _fmt_mb(num_bytes) -> str:
    if num_bytes is None:
        return "n/a"
    return f"{num_bytes / (1024 * 1024):.2f} MB"
