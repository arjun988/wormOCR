import pytest
import torch
from PIL import Image, ImageDraw
from torch.utils.data import Dataset

from wormocr.checkpoints import save_checkpoint
from wormocr.config import load_config
from wormocr.data.mixed import MixedDataset, RealLinesDataset
from wormocr.models.crnn import CRNN
from wormocr.training.loop import initialize_weights
from wormocr.vocabulary import Vocabulary, ctc_time_needed


class Constant(Dataset):
    def __init__(self, tag: str, size: int) -> None:
        self.tag = tag
        self.size = size

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> dict:
        return {"tag": self.tag, "index": index}


def test_mixture_is_deterministic_and_follows_weights():
    parts = [(Constant("a", 5), 3.0), (Constant("b", 7), 1.0)]
    first = MixedDataset(parts, 4000, seed=11)
    second = MixedDataset(parts, 4000, seed=11)
    tags = [first[index]["tag"] for index in range(4000)]
    assert tags == [second[index]["tag"] for index in range(4000)]
    assert tags.count("a") / len(tags) == pytest.approx(0.75, abs=0.03)
    assert tags != [MixedDataset(parts, 4000, seed=12)[index]["tag"] for index in range(4000)]


def test_real_lines_fit_ctc_and_skip_labels_outside_the_charset(tmp_path):
    (tmp_path / "images").mkdir()
    rows = []
    for name, text in [("ok", "Hello there"), ("long", "a much longer line for a narrow crop"), ("bad", "caf\u00e9")]:
        image = Image.new("L", (60, 32), 250)
        ImageDraw.Draw(image).text((2, 8), text[:8], fill=10)
        image.save(tmp_path / "images" / f"{name}.png")
        rows.append(f"images/{name}.png\t{text}")
    (tmp_path / "train.tsv").write_text("\n".join(rows), encoding="utf-8")
    config = load_config("configs/unified.yaml")
    dataset = RealLinesDataset(tmp_path / "train.tsv", config, augment=True)
    assert [text for _path, text in dataset.records] == ["Hello there", "a much longer line for a narrow crop"]
    model = CRNN(Vocabulary().size, image_height=32, channels=[48, 96, 160, 192], width_stride=4)
    for index in range(len(dataset)):
        item = dataset[index]
        frames = int(model.output_lengths(torch.tensor([item["width"]]))[0])
        assert frames >= ctc_time_needed(item["text"])


def test_init_from_keeps_fresh_tensors_whose_shape_changed(tmp_path):
    vocabulary = Vocabulary()
    source = CRNN(vocabulary.size, image_height=32, channels=[8, 8, 8, 8], rnn_hidden=16, proj_size=16)
    save_checkpoint(
        tmp_path / "source.pt",
        {"model_state": source.state_dict(), "charset": vocabulary.charset, "config": {}},
    )
    target = CRNN(vocabulary.size, image_height=48, channels=[8, 8, 8, 8], rnn_hidden=16, proj_size=16)
    fresh_proj = target.proj[0].weight.detach().clone()
    initialize_weights(target, str(tmp_path / "source.pt"), vocabulary, torch.device("cpu"))
    assert torch.equal(target.proj[0].weight, fresh_proj)
    assert torch.equal(target.classifier.weight, source.classifier.weight)
    assert torch.equal(target.rnn.weight_hh_l0, source.rnn.weight_hh_l0)


def test_handwriting_samples_are_deterministic_and_trainable():
    from wormocr.data.handwriting import handwriting_fonts
    from wormocr.data.mixed import SynthHandwritingDataset
    from wormocr.exceptions import DataError

    try:
        handwriting_fonts()
    except DataError:
        pytest.skip("handwriting fonts are not downloaded")
    config = load_config("configs/unified.yaml")
    dataset = SynthHandwritingDataset(config, 6, seed=5, augment=True)
    again = SynthHandwritingDataset(config, 6, seed=5, augment=True)
    model = CRNN(Vocabulary().size, image_height=32, channels=[48, 96, 160, 192], width_stride=4)
    for index in range(6):
        item = dataset[index]
        assert item["text"] == again[index]["text"]
        assert torch.equal(item["image"], again[index]["image"])
        frames = int(model.output_lengths(torch.tensor([item["width"]]))[0])
        assert frames >= ctc_time_needed(item["text"])
