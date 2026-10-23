"""Turn a training checkpoint into release files for GitHub.

Writes <name>.safetensors (tensors only, no pickle), <name>.json (config,
charset, provenance), <name>.onnx, and SHA256SUMS.txt.

    python scripts/package_release.py artifacts/checkpoints/tiny/best_weights.pt wormocr-2M artifacts/release
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import save_file

from wormocr import __version__
from wormocr.checkpoints import load_checkpoint, public_config
from wormocr.export.onnx_export import export_onnx, onnx_max_abs_diff
from wormocr.hub import REPOSITORY, sha256
from wormocr.inference.engine import LineEngine
from wormocr.models.crnn import count_parameters


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("name")
    parser.add_argument("out")
    args = parser.parse_args()

    payload = load_checkpoint(args.checkpoint)
    folder = Path(args.out) / args.name
    folder.mkdir(parents=True, exist_ok=True)

    weights = folder / f"{args.name}.safetensors"
    state = {key: value.detach().cpu().contiguous() for key, value in payload["model_state"].items()}
    save_file(state, str(weights), metadata={"name": args.name, "format": "wormocr-safetensors-1"})

    config = public_config(payload["config"])
    config["model"]["name"] = args.name
    config["device"] = "cpu"
    config.setdefault("train", {})["checkpoint_dir"] = f"artifacts/checkpoints/{args.name}"
    meta = {
        "name": args.name,
        "repository": f"https://github.com/{REPOSITORY}",
        "license": "MIT",
        "maturity": payload.get("maturity", "untrained"),
        "training_step": payload.get("step"),
        "validation_cer": payload.get("best_cer"),
        "wormocr_version": __version__,
        "config": config,
        "charset": payload["charset"],
    }
    meta_path = folder / f"{args.name}.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    engine = LineEngine.load(weights, device="cpu")
    reference = LineEngine.load(args.checkpoint, device="cpu")
    for key, tensor in reference.model.state_dict().items():
        if not torch.equal(tensor, engine.model.state_dict()[key]):
            raise SystemExit(f"tensor {key} changed during conversion")
    meta["parameters"] = count_parameters(engine.model)
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    onnx_path = export_onnx(engine.model, folder / f"{args.name}.onnx")
    diff = onnx_max_abs_diff(engine.model, onnx_path)
    if diff > 1e-3:
        raise SystemExit(f"ONNX differs from PyTorch by {diff}")

    files = [weights, meta_path, onnx_path]
    sums = "".join(f"{sha256(path)}  {path.name}\n" for path in files)
    (folder / "SHA256SUMS.txt").write_text(sums, encoding="utf-8")
    print(sums, end="")
    print(json.dumps({"parameters": meta["parameters"], "onnx_max_abs_diff": diff,
                      "bytes": {path.name: path.stat().st_size for path in files}}, indent=2))


if __name__ == "__main__":
    main()
