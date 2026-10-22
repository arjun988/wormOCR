import io
import json

import pytest
import torch
from safetensors.torch import save_file

from wormocr import hub
from wormocr.api import resolve_checkpoint
from wormocr.config import load_config
from wormocr.exceptions import CheckpointError
from wormocr.inference.engine import LineEngine
from wormocr.models.crnn import build_model
from wormocr.vocabulary import Vocabulary


def _write_release(folder, name="wormocr-test"):
    config = load_config("configs/debug.yaml")
    vocabulary = Vocabulary()
    model = build_model(config, vocabulary.size)
    folder.mkdir(parents=True, exist_ok=True)
    weights = folder / f"{name}.safetensors"
    save_file({key: value.contiguous() for key, value in model.state_dict().items()}, str(weights))
    meta = {"name": name, "maturity": "trained", "config": config, "charset": vocabulary.charset}
    (folder / f"{name}.json").write_text(json.dumps(meta), encoding="utf-8")
    return weights, model


def test_safetensors_release_loads_identical_weights(tmp_path):
    weights, model = _write_release(tmp_path)
    engine = LineEngine.load(weights, device="cpu")
    assert engine.maturity == "trained"
    for key, tensor in model.state_dict().items():
        assert torch.equal(tensor, engine.model.state_dict()[key])


def test_released_model_is_never_downloaded_implicitly(tmp_path, monkeypatch):
    monkeypatch.setenv("WORMOCR_HOME", str(tmp_path))
    monkeypatch.setattr(hub.urllib.request, "urlopen", lambda *a, **k: pytest.fail("network used"))
    with pytest.raises(CheckpointError) as exc:
        resolve_checkpoint(hub.DEFAULT_MODEL, None)
    assert f"wormocr download {hub.DEFAULT_MODEL}" in str(exc.value)


def test_download_verifies_checksums(tmp_path, monkeypatch):
    weights, _ = _write_release(tmp_path / "source")
    meta = weights.with_suffix(".json")
    release = hub.Release(
        name="wormocr-test",
        tag="wormocr-test",
        files={weights.name: hub.sha256(weights), meta.name: hub.sha256(meta)},
        description="test",
    )
    monkeypatch.setitem(hub.RELEASES, "wormocr-test", release)
    blobs = {release.url(path.name): path.read_bytes() for path in (weights, meta)}
    monkeypatch.setattr(hub.urllib.request, "urlopen", lambda url, timeout=0: io.BytesIO(blobs[url]))

    path = hub.download("wormocr-test", root=tmp_path / "cache", log=lambda _m: None)
    assert path == hub.local_weights("wormocr-test", tmp_path / "cache")
    assert LineEngine.load(path, device="cpu").maturity == "trained"

    blobs[release.url(meta.name)] = b"{}"
    with pytest.raises(CheckpointError, match="SHA-256 mismatch"):
        hub.download("wormocr-test", root=tmp_path / "cache", force=True, log=lambda _m: None)
    assert hub.sha256(tmp_path / "cache" / "wormocr-test" / meta.name) == release.files[meta.name]
    assert not list((tmp_path / "cache" / "wormocr-test").glob("*.part"))
