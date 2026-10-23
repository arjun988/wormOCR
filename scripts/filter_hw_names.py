"""Drop handwritten-names crops that still show printed form labels.

Some crops keep "NOM :", "PRENOM :", or "DATE DE NAISSANCE" beside or instead
of the handwritten name, while the label holds only the name. wormocr-2M reads
printed text well, so a crop where it finds a form keyword that the label does
not contain is removed. The originals are kept as <split>.unfiltered.tsv and the
counts are added to STATS.json. Runs on the GPU.

    python scripts/filter_hw_names.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from eval_slices import EvalLines, load_model  # noqa: E402
from wormocr.data.dataset import collate_lines  # noqa: E402
from wormocr.inference.decode import greedy_decode  # noqa: E402
from wormocr.runtime import resolve_device  # noqa: E402

FOLDER = ROOT / "artifacts" / "datasets" / "prepared" / "hw_names"
KEYWORDS = re.compile(r"PRENOM|RENOM|NOM\b|NOM:|DATE|NAISS|CLASSE|CODE|RNE|UAI|:")


def flagged(label: str, read: str) -> bool:
    found = set(KEYWORDS.findall(read.upper()))
    return any(word not in label.upper() for word in found)


@torch.no_grad()
def main() -> None:
    device = resolve_device("cuda")
    model, vocabulary, payload = load_model("wormocr-2M", device)
    height = int(payload["config"]["model"]["image_height"])
    stats_path = FOLDER / "STATS.json"
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    report = {}
    for split in ("train", "val", "test"):
        source = FOLDER / f"{split}.tsv"
        backup = FOLDER / f"{split}.unfiltered.tsv"
        if not backup.exists():
            source.replace(backup)
        dataset = EvalLines(backup, height)
        loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4, collate_fn=collate_lines)
        kept, dropped, examples = [], 0, []
        position = 0
        for batch in loader:
            log_probs = model(batch["images"].to(device))
            lengths = model.output_lengths(batch["widths"]).clamp(min=1, max=int(log_probs.shape[1]))
            for indices in greedy_decode(log_probs, lengths, vocabulary.blank):
                relative, label = dataset.records[position]
                position += 1
                read = vocabulary.decode(indices)
                if flagged(label, read):
                    dropped += 1
                    if len(examples) < 8:
                        examples.append({"label": label, "read": read})
                else:
                    kept.append(f"{relative}\t{label}")
        source.write_text("\n".join(kept) + "\n", encoding="utf-8")
        report[split] = {"kept": len(kept), "dropped_printed_label": dropped, "examples": examples}
        print(split, len(kept), "kept,", dropped, "dropped", examples[:3])
    stats["printed_label_filter"] = {
        "model": "wormocr-2M",
        "rule": "drop when the model reads a form keyword (NOM, PRENOM, DATE, CLASSE, CODE, RNE, UAI, ':') absent from the label",
        "splits": report,
    }
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
