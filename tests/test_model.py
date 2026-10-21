import torch

from wormocr.config import load_config
from wormocr.models.crnn import build_model, count_parameters, output_width, build_specs
from wormocr.vocabulary import Vocabulary


def _model():
    config = load_config("configs/debug.yaml")
    return build_model(config, Vocabulary().size), config


def test_forward_matches_width_formula_and_backward():
    model, config = _model()
    specs = build_specs(config["model"]["channels"])
    for width in (8, 15, 16, 17, 32, 100, 257):
        images = torch.rand(2, 1, 32, width)
        log_probs = model(images)
        expected = output_width(width, specs)
        assert log_probs.shape[0] == 2
        assert log_probs.shape[1] == expected
        assert log_probs.shape[2] == Vocabulary().size
        assert torch.allclose(log_probs.exp().sum(dim=-1), torch.ones(2, expected), atol=1e-4)
    loss = log_probs.sum()
    loss.backward()
    assert any(parameter.grad is not None and float(parameter.grad.abs().sum()) > 0 for parameter in model.parameters())


def test_output_lengths_follow_unpadded_width():
    model, _config = _model()
    images = torch.rand(2, 1, 32, 80)
    log_probs = model(images)
    lengths = model.output_lengths(torch.tensor([40, 80]))
    assert int(lengths[1]) == log_probs.shape[1]
    assert int(lengths[0]) < int(lengths[1])


def test_variants_are_ordered_and_within_bounds():
    charset = Vocabulary().size
    counts = {}
    for name in ("micro", "tiny", "small"):
        config = load_config(f"configs/{name}.yaml")
        counts[name] = count_parameters(build_model(config, charset))
    assert counts["micro"] < counts["tiny"] < counts["small"]
    assert counts["micro"] < 4_000_000
    assert counts["tiny"] < 12_000_000
    assert counts["small"] < 30_000_000
    assert counts["small"] > 1_000_000
