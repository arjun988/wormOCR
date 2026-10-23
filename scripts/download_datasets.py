"""Download the real-image datasets used for wormOCR training and evaluation.

Every entry records its license and whether it may be used for training.
Files go to artifacts/datasets/raw/<name>/ (gitignored) with a SOURCE.json.

    python scripts/download_datasets.py            # everything
    python scripts/download_datasets.py gnhk cord  # selected datasets
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "artifacts" / "datasets" / "raw"

DATASETS = {
    "gnhk": {
        "what": "GoodNotes Handwriting Kollection: camera photos of English handwriting, word boxes grouped into lines",
        "license": "CC BY 4.0",
        "use": "train (train split) + eval (test split)",
        "source": "https://github.com/GoodNotes/GNHK-dataset (mirror: huggingface.co/datasets/staghado/GNHK-Dataset)",
        "hf": ("staghado/GNHK-Dataset", ["GNHK-dataset.zip"]),
    },
    "hw_names": {
        "what": "400k handwritten first names and surnames from scanned forms (Kaggle 'Handwriting Recognition')",
        "license": "CC0 1.0 (public domain)",
        "use": "train + eval",
        "source": "https://www.kaggle.com/datasets/landlord/handwriting-recognition (mirror: huggingface.co/datasets/verbrannter/handwriting_recognition_from_kaggle)",
        "hf": (
            "verbrannter/handwriting_recognition_from_kaggle",
            [
                "data/train-00000-of-00003.parquet",
                "data/train-00001-of-00003.parquet",
                "data/train-00002-of-00003.parquet",
                "data/validation-00000-of-00001.parquet",
                "data/test-00000-of-00001.parquet",
            ],
        ),
    },
    "bentham": {
        "what": "ICDAR 2015 HTRtS: Bentham collection pages (historical English handwriting) with ALTO line transcripts",
        "license": "CC BY 4.0",
        "use": "train + eval",
        "source": "https://doi.org/10.5281/zenodo.248733 (mirror: huggingface.co/datasets/SSamDav/icdar-htr-competition-2015)",
        "hf": (
            "SSamDav/icdar-htr-competition-2015",
            [f"data/train-0000{i}-of-00008.parquet" for i in range(8)] + ["data/test-00000-of-00001.parquet"],
        ),
    },
    "rxhandbd": {
        "what": "RxHandBD: 5,578 handwritten words cropped from real doctor prescriptions (Bangladesh)",
        "license": "MIT",
        "use": "train + eval",
        "source": "https://doi.org/10.5281/zenodo.18478741",
        "urls": {
            "RxHand Original.zip": (
                "https://zenodo.org/records/18478741/files/RxHand%20Original.zip?download=1",
                "993b64bdf2a6a08a036aeb9a958cc9a5",
            ),
        },
    },
    "cord": {
        "what": "CORD v2: 1,000 phone photos of printed receipts with line-level text and boxes",
        "license": "CC BY 4.0",
        "use": "train + eval (real printed text in photos)",
        "source": "https://github.com/clovaai/cord (huggingface.co/datasets/naver-clova-ix/cord-v2)",
        "hf": (
            "naver-clova-ix/cord-v2",
            [
                "data/train-00000-of-00004-b4aaeceff1d90ecb.parquet",
                "data/train-00001-of-00004-7dbbe248962764c5.parquet",
                "data/train-00002-of-00004-688fe1305a55e5cc.parquet",
                "data/train-00003-of-00004-2d0cd200555ed7fd.parquet",
                "data/validation-00000-of-00001-cc3c5779fe22e8ca.parquet",
                "data/test-00000-of-00001-9c204eb3f4e11791.parquet",
            ],
        ),
    },
    "iam_test": {
        "what": "IAM handwriting lines, test split only (standard benchmark)",
        "license": "IAM terms: non-commercial research only (the Hugging Face mirror's MIT tag does not override this)",
        "use": "EVALUATION ONLY - never train on it",
        "source": "https://fki.tic.heia-fr.ch/databases/iam-handwriting-database (mirror: huggingface.co/datasets/Teklia/IAM-line)",
        "hf": ("Teklia/IAM-line", ["data/test.parquet"]),
    },
}


def _copy_with_progress(response, out, name: str, done: int = 0) -> None:
    total = int(response.headers.get("Content-Length") or 0) + done
    next_report = (done // (100 << 20) + 1) * (100 << 20)
    while True:
        block = response.read(1 << 20)
        if not block:
            break
        out.write(block)
        done += len(block)
        if done >= next_report:
            share = f" of {total / 1e6:.0f} MB" if total else ""
            print(f"  {name}: {done / 1e6:.0f} MB{share}", flush=True)
            next_report += 100 << 20
    # A dropped connection can end the read with no exception. A short file is
    # not a finished file; the retry loop resumes from the bytes already saved.
    if total and done + (1 << 20) < total:
        raise OSError(f"connection closed at {done / 1e6:.0f} MB of {total / 1e6:.0f} MB")


def _download_url(url: str, target: Path) -> None:
    if target.exists() and target.stat().st_size > 0:
        print(f"  {target.name}: already present", flush=True)
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    for attempt in range(1, 31):
        have = partial.stat().st_size if partial.exists() else 0
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "wormocr-dataset-fetch"})
            if have:
                request.add_header("Range", f"bytes={have}-")
            with urllib.request.urlopen(request, timeout=120) as response:
                resumed = have and response.status == 206
                with open(partial, "ab" if resumed else "wb") as out:
                    _copy_with_progress(response, out, target.name, have if resumed else 0)
            partial.replace(target)
            print(f"  {target.name}: {target.stat().st_size / 1e6:.1f} MB", flush=True)
            return
        except OSError as exc:
            print(f"  {target.name}: attempt {attempt} interrupted at {have / 1e6:.0f} MB: {exc}", flush=True)
            time.sleep(min(60, 5 * attempt))
    raise SystemExit(f"could not download {url}")


def fetch(name: str) -> None:
    spec = DATASETS[name]
    folder = ROOT / name
    folder.mkdir(parents=True, exist_ok=True)
    print(f"[{name}] {spec['what']} ({spec['license']}; {spec['use']})", flush=True)
    if "hf" in spec:
        repo, files = spec["hf"]
        for filename in files:
            url = f"https://huggingface.co/datasets/{repo}/resolve/main/{filename}"
            _download_url(url, folder / filename)
    for filename, (url, md5) in spec.get("urls", {}).items():
        target = folder / filename
        _download_url(url, target)
        actual = hashlib.md5(target.read_bytes()).hexdigest()
        if actual != md5:
            target.unlink()
            raise SystemExit(f"{filename}: MD5 {actual} does not match the published {md5}; deleted, run again")
    meta = {key: value for key, value in spec.items() if key not in {"hf", "urls"}}
    meta["downloaded"] = time.strftime("%Y-%m-%d")
    (folder / "SOURCE.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("names", nargs="*", help=f"any of: {', '.join(DATASETS)}")
    args = parser.parse_args()
    unknown = sorted(set(args.names) - set(DATASETS))
    if unknown:
        parser.error(f"unknown datasets: {', '.join(unknown)}")
    for name in args.names or list(DATASETS):
        fetch(name)
    print("done", flush=True)


if __name__ == "__main__":
    sys.exit(main())
