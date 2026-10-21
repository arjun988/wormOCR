"""Command-line interface.

Every documented command is implemented here. Checkpoints are local files.
Nothing in this module downloads weights.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from wormocr import __version__
from wormocr.benchmark.report import run_benchmark
from wormocr.config import load_config
from wormocr.data.fonts import discover_fonts
from wormocr.data.synthetic import generate_dataset
from wormocr.evaluation.errors import ErrorReport
from wormocr.evaluation.metrics import cer
from wormocr.evaluation.robustness import measure_robustness
from wormocr.exceptions import WormOCRError
from wormocr.export.onnx_export import export_onnx, onnx_max_abs_diff, quantize_onnx
from wormocr.inference.engine import LineEngine
from wormocr.models.crnn import build_model, count_parameters
from wormocr.runtime import environment_report
from wormocr.training.loop import build_datasets, evaluate_model, make_loader, train
from wormocr.vocabulary import Vocabulary


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args) or 0)
    except WormOCRError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except MemoryError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="wormocr", description="Small local OCR for printed English text.")
    parser.add_argument("--version", action="version", version=f"wormocr {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    info = commands.add_parser("info", help="show the runtime, fonts, and model sizes")
    info.set_defaults(func=command_info)

    train_cmd = commands.add_parser("train", help="train or smoke-test a recognizer")
    train_cmd.add_argument("--config", required=True)
    train_cmd.add_argument("--mode", required=True, choices=["smoke", "debug", "baseline", "resume"])
    train_cmd.add_argument("--checkpoint", help="checkpoint to resume from")
    train_cmd.add_argument("--device", default=None)
    train_cmd.add_argument("--set", dest="overrides", action="append", default=[], help="dotted override, such as train.max_steps=20")
    train_cmd.set_defaults(func=command_train)

    predict = commands.add_parser("predict", help="recognize an image or PDF")
    predict.add_argument("--image", action="append", required=True)
    predict.add_argument("--model", default="tiny", help="variant name used to find a local checkpoint")
    predict.add_argument("--checkpoint", default=None)
    predict.add_argument("--device", default="cpu")
    predict.add_argument("--mode", default="auto", choices=["auto", "line", "page"])
    predict.add_argument("--beam", type=int, default=0)
    predict.add_argument("--json", action="store_true")
    predict.add_argument("--no-orientation", action="store_true")
    predict.set_defaults(func=command_predict)

    evaluate = commands.add_parser("evaluate", help="measure CER and WER")
    evaluate.add_argument("--model", required=True, help="checkpoint path")
    evaluate.add_argument("--labels", default=None)
    evaluate.add_argument("--image-root", default=None)
    evaluate.add_argument("--synthetic-samples", type=int, default=100)
    evaluate.add_argument("--seed", type=int, default=12345)
    evaluate.add_argument("--device", default="cpu")
    evaluate.add_argument("--beam", type=int, default=0)
    evaluate.add_argument("--batch-size", type=int, default=8)
    evaluate.add_argument("--errors", default=None, help="write an error-analysis markdown file")
    evaluate.add_argument("--robustness", type=int, default=0, help="samples per corruption; 0 skips it")
    evaluate.add_argument("--json", action="store_true")
    evaluate.set_defaults(func=command_evaluate)

    analyze = commands.add_parser("analyze", help="write an error analysis report")
    analyze.add_argument("--model", required=True)
    analyze.add_argument("--synthetic-samples", type=int, default=50)
    analyze.add_argument("--seed", type=int, default=12345)
    analyze.add_argument("--device", default="cpu")
    analyze.add_argument("--output", required=True)
    analyze.set_defaults(func=command_analyze)

    export = commands.add_parser("export", help="export ONNX or INT8 ONNX")
    export.add_argument("--model", required=True, help="checkpoint path")
    export.add_argument("--format", required=True, choices=["onnx", "onnx-int8"])
    export.add_argument("--output", required=True)
    export.add_argument("--opset", type=int, default=17)
    export.set_defaults(func=command_export)

    benchmark = commands.add_parser("benchmark", help="measure size, accuracy, latency, and memory")
    benchmark.add_argument("--model", required=True, help="checkpoint path")
    benchmark.add_argument("--output", default="artifacts/benchmarks/report.md")
    benchmark.add_argument("--samples", type=int, default=40)
    benchmark.add_argument("--repeats", type=int, default=20)
    benchmark.add_argument("--device", default="cpu")
    benchmark.add_argument("--seed", type=int, default=12345)
    benchmark.set_defaults(func=command_benchmark)

    generate = commands.add_parser("generate", help="write a synthetic line dataset and manifest")
    generate.add_argument("--config", default="configs/tiny.yaml")
    generate.add_argument("--samples", type=int, default=32)
    generate.add_argument("--out", required=True)
    generate.add_argument("--seed", type=int, default=0)
    generate.add_argument("--augment", action="store_true")
    generate.set_defaults(func=command_generate)
    return parser


def command_info(_args: argparse.Namespace) -> int:
    report = environment_report()
    try:
        fonts = discover_fonts("")
        report["fonts"] = len(fonts)
    except WormOCRError as exc:
        report["fonts"] = 0
        report["font_error"] = str(exc)
    charset = Vocabulary()
    for name in ("micro", "tiny", "small"):
        config_name = f"configs/{name}.yaml"
        try:
            config = load_config(config_name)
        except WormOCRError:
            continue
        model = build_model(config, charset.size)
        report[f"{name}_parameters"] = count_parameters(model)
    print(json.dumps(report, indent=2))
    return 0


def command_train(args: argparse.Namespace) -> int:
    config = load_config(args.config, args.overrides, mode=args.mode)
    if args.device:
        config["device"] = args.device
    result = train(config, mode=args.mode, resume_from=args.checkpoint)
    print(json.dumps({
        "last_checkpoint": result.last_checkpoint,
        "best_checkpoint": result.best_checkpoint,
        "steps": result.steps,
        "best_cer": result.best_cer,
        "maturity": result.maturity,
        "final_loss": result.final_loss,
    }, indent=2))
    return 0


def command_predict(args: argparse.Namespace) -> int:
    from wormocr.api import OCR

    ocr = OCR(model=args.model, checkpoint=args.checkpoint, device=args.device, beam_width=args.beam)
    results = [
        ocr.predict(image, mode=args.mode, correct_orientation=not args.no_orientation)
        for image in args.image
    ]
    if args.json or len(results) > 1:
        print(json.dumps([result.to_dict() for result in results], indent=2, ensure_ascii=True))
    else:
        print(results[0].text)
    return 0


def command_evaluate(args: argparse.Namespace) -> int:
    engine = LineEngine.load(args.model, device=args.device, beam_width=args.beam)
    config = dict(engine.config)
    if args.labels:
        config["data"]["labels"] = args.labels
        config["data"]["image_root"] = args.image_root or str(Path(args.labels).parent)
        config["data"]["val_labels"] = None
    else:
        config["data"]["labels"] = None
        config["data"]["train_samples"] = args.synthetic_samples
        config["data"]["augment"] = False
        config["seed"] = args.seed
    if args.labels:
        dataset, _val, vocabulary = build_datasets(config)
    else:
        from wormocr.data.dataset import OnlineSynthDataset

        vocabulary = engine.vocabulary
        dataset = OnlineSynthDataset(config, args.synthetic_samples, args.seed, augment=False)
    loader = make_loader(dataset, args.batch_size, False, args.seed, 0)
    totals, pairs = evaluate_model(
        engine.model, loader, vocabulary, engine.device, collect=bool(args.errors)
    )
    payload = totals.to_dict()
    payload["maturity"] = engine.maturity
    payload["checkpoint"] = str(args.model)
    payload["synthetic"] = not bool(args.labels)
    if args.robustness:
        payload["robustness"] = measure_robustness(engine, args.robustness, args.seed)
    if args.errors:
        report = ErrorReport()
        for reference, hypothesis in pairs:
            report.add(reference, hypothesis, cer(reference, hypothesis))
        Path(args.errors).parent.mkdir(parents=True, exist_ok=True)
        Path(args.errors).write_text(report.to_markdown(), encoding="utf-8")
        payload["errors"] = args.errors
    print(json.dumps(payload, indent=2))
    return 0


def command_analyze(args: argparse.Namespace) -> int:
    args.errors = args.output
    args.labels = None
    args.image_root = None
    args.beam = 0
    args.batch_size = 8
    args.robustness = 0
    args.json = True
    return command_evaluate(args)


def command_export(args: argparse.Namespace) -> int:
    engine = LineEngine.load(args.model, device="cpu")
    output = Path(args.output)
    if args.format == "onnx":
        export_onnx(engine.model, output, opset=args.opset)
        diff = onnx_max_abs_diff(engine.model, output)
        if diff > 1e-3:
            raise WormOCRError(f"ONNX outputs differ from PyTorch by {diff:.4g}")
        print(json.dumps({"output": str(output), "bytes": output.stat().st_size, "max_abs_diff": diff}, indent=2))
        return 0
    fp32 = output.with_suffix(".fp32.onnx")
    export_onnx(engine.model, fp32, opset=args.opset)
    fp32_diff = onnx_max_abs_diff(engine.model, fp32)
    quantize_onnx(fp32, output)
    int8_diff = onnx_max_abs_diff(engine.model, output)
    print(json.dumps({
        "output": str(output),
        "fp32": str(fp32),
        "fp32_bytes": fp32.stat().st_size,
        "int8_bytes": output.stat().st_size,
        "fp32_max_abs_diff": fp32_diff,
        "int8_max_abs_diff": int8_diff,
    }, indent=2))
    return 0


def command_benchmark(args: argparse.Namespace) -> int:
    report = run_benchmark(
        args.model,
        args.output,
        samples=args.samples,
        seed=args.seed,
        repeats=args.repeats,
        device=args.device,
    )
    print(f"wrote {args.output}")
    synthetic = report["synthetic_held_out"]
    print(
        f"parameters={report['parameters']} maturity={report['maturity']} "
        f"CER={synthetic['cer']:.4f} median_model_ms="
        f"{report['latency']['model_only_batch1_width256']['median_ms']:.2f}"
    )
    return 0


def command_generate(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    manifest = generate_dataset(config, args.samples, args.out, args.seed, augment=args.augment)
    print(manifest)
    return 0
