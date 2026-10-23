"""Measure a model on the frozen phase-3 evaluation slices.

    python scripts/eval_slices.py wormocr-2M
    python scripts/eval_slices.py artifacts/checkpoints/unified/best.pt --slices gnhk iam

Real slices are prepared test splits (scripts/prepare_datasets.py) read with the
same preprocessing as inference: no label-dependent widening. Synthetic slices
are fixed seeds of the generators. Every slice records a SHA-256 of its labels
so later runs can prove they measured the same lines. Results go to
artifacts/eval/<model>.json. Slices whose data is missing are reported as such,
never estimated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wormocr.checkpoints import load_checkpoint  # noqa: E402
from wormocr.config import load_config  # noqa: E402
from wormocr.data.dataset import OnlineSynthDataset, _read_labels, collate_lines  # noqa: E402
from wormocr.data.mixed import SynthHandwritingDataset  # noqa: E402
from wormocr.evaluation.metrics import MetricTotals  # noqa: E402
from wormocr.hub import RELEASES, local_weights  # noqa: E402
from wormocr.inference.decode import greedy_decode  # noqa: E402
from wormocr.models.crnn import build_model  # noqa: E402
from wormocr.preprocessing import prepare_line  # noqa: E402
from wormocr.runtime import resolve_device  # noqa: E402
from wormocr.vocabulary import Vocabulary  # noqa: E402

PREPARED = ROOT / "artifacts" / "datasets" / "prepared"
SYNTH_SAMPLES = 1000

# name -> (kind, source, case_insensitive, note)
SLICES = {
    "printed_synth": ("synthetic", 424242, False, "printed generator, fixed seed"),
    "handwriting_synth": ("handwriting", 535353, False, "handwriting generator, fixed seed"),
    "funsd": ("labels", "../funsd_test/labels.tsv", False, "scanned form words, FUNSD test, evaluation only"),
    "cord": ("labels", "cord/test.tsv", False, "printed receipts, CORD v2 test"),
    "gnhk": ("labels", "gnhk/test.tsv", False, "handwritten notes in the wild, GNHK test"),
    "hw_names": ("labels", "hw_names/test.tsv", False, "handwritten form names, CC0"),
    "bentham": ("labels", "bentham/test.tsv", False, "historical handwriting, ICDAR 2015 HTRtS test"),
    "iam": ("labels", "iam_test/test.tsv", False, "IAM line test set, evaluation only (non-commercial)"),
    "rxhand": ("labels", "rxhand/test.tsv", True, "prescription words, case-insensitive, evaluation only"),
}


class EvalLines(Dataset):
    def __init__(self, labels: Path, height: int) -> None:
        self.root = labels.parent
        self.records = _read_labels(labels)
        self.height = height

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        from PIL import Image

        relative, text = self.records[index]
        with Image.open(self.root / relative) as handle:
            image = handle.convert("L")
        tensor, pad = prepare_line(image, height=self.height, align=2, max_width=4096)
        return {"image": tensor, "text": text, "pad_value": pad, "width": int(tensor.shape[-1])}


def labels_sha(texts: list[str]) -> str:
    return hashlib.sha256("\n".join(texts).encode("utf-8")).hexdigest()


def load_model(source: str, device: torch.device):
    path = local_weights(source) if source in RELEASES else Path(source)
    if path is None:
        raise SystemExit(f"{source} is not downloaded. Run: wormocr download {source}")
    payload = load_checkpoint(path, map_location=device)
    vocabulary = Vocabulary(payload["charset"])
    model = build_model(payload["config"], vocabulary.size).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model, vocabulary, payload


@torch.no_grad()
def run(model, vocabulary, dataset, device, case_insensitive: bool, batch_size: int = 64) -> tuple[dict, list]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_lines)
    totals, folded = MetricTotals(), MetricTotals()
    worst: list[tuple[float, str, str]] = []
    for batch in loader:
        log_probs = model(batch["images"].to(device))
        lengths = model.output_lengths(batch["widths"]).clamp(min=1, max=int(log_probs.shape[1]))
        for text, indices in zip(batch["texts"], greedy_decode(log_probs, lengths, vocabulary.blank)):
            hypothesis = vocabulary.decode(indices)
            totals.update(text, hypothesis)
            folded.update(text.lower(), hypothesis.lower())
            worst.append((0.0 if text == hypothesis else 1.0, text, hypothesis))
    result = totals.to_dict()
    result["cer_case_insensitive"] = folded.corpus_cer
    result["wer_case_insensitive"] = folded.corpus_wer
    if case_insensitive:
        result["headline_cer"] = folded.corpus_cer
    else:
        result["headline_cer"] = totals.corpus_cer
    errors = [(text, hypothesis) for miss, text, hypothesis in worst if miss][:12]
    return result, errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("model", help="released model name or checkpoint path")
    parser.add_argument("--slices", nargs="*", default=list(SLICES))
    parser.add_argument("--limit", type=int, default=0, help="cap lines per real slice (0 = all)")
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    device = resolve_device("cuda")
    model, vocabulary, payload = load_model(args.model, device)
    height = int(payload["config"]["model"]["image_height"])
    base = load_config(str(ROOT / "configs" / "unified.yaml"))
    base["model"] = payload["config"]["model"]
    report = {
        "model": args.model,
        "parameters": sum(p.numel() for p in model.parameters()),
        "device": torch.cuda.get_device_name(0),
        "decoder": "greedy",
        "slices": {},
    }
    for name in args.slices:
        kind, source, case_insensitive, note = SLICES[name]
        started = time.perf_counter()
        if kind == "labels":
            labels = PREPARED / source
            if not labels.exists():
                report["slices"][name] = {"status": "missing", "note": note, "labels": str(labels)}
                print(f"{name:18s} missing ({labels.relative_to(ROOT)})")
                continue
            dataset = EvalLines(labels, height)
            if args.limit and len(dataset) > args.limit:
                dataset.records = dataset.records[: args.limit]
            texts = [text for _relative, text in dataset.records]
        elif kind == "synthetic":
            dataset = OnlineSynthDataset(base, SYNTH_SAMPLES, int(source), augment=True)
            texts = None
        else:
            dataset = SynthHandwritingDataset(base, SYNTH_SAMPLES, int(source), augment=True)
            texts = None
        if texts is None:
            items = [dataset[index] for index in range(len(dataset))]
            texts = [item["text"] for item in items]
            dataset = items
        metrics, errors = run(model, vocabulary, dataset, device, case_insensitive)
        metrics.update(
            {"status": "ok", "note": note, "labels_sha256": labels_sha(texts),
             "case_insensitive_headline": case_insensitive, "seconds": round(time.perf_counter() - started, 1),
             "sample_errors": errors}
        )
        report["slices"][name] = metrics
        print(
            f"{name:18s} n={metrics['samples']:5d}  CER {metrics['headline_cer'] * 100:6.2f}%  "
            f"WER {metrics['wer'] * 100:6.2f}%  exact {metrics['exact_match'] * 100:5.1f}%"
            + ("  (case-insensitive)" if case_insensitive else "")
        )
    out = Path(args.out) if args.out else ROOT / "artifacts" / "eval" / f"{Path(args.model).stem if args.model not in RELEASES else args.model}.json"
    out = out.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
