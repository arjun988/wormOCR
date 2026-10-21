import torch

from wormocr.inference.decode import ctc_beam_search, greedy_decode
from wormocr.vocabulary import Vocabulary


def _peaked(rows: list[int], classes: int) -> torch.Tensor:
    frames = []
    for label in rows:
        frame = torch.full((classes,), -20.0)
        frame[label] = 0.0
        frames.append(frame.log_softmax(dim=-1))
    return torch.stack(frames).unsqueeze(0)


def test_greedy_collapses_repeats_and_keeps_blank_separated_repeats():
    vocabulary = Vocabulary()
    classes = vocabulary.size
    blank = vocabulary.blank
    code = vocabulary.encode("a")[0]
    collapsed = greedy_decode(_peaked([code, code, code], classes), blank=blank)
    assert vocabulary.decode(collapsed[0]) == "a"
    separated = greedy_decode(_peaked([code, blank, code], classes), blank=blank)
    assert vocabulary.decode(separated[0]) == "aa"
    book = vocabulary.encode("book")
    # b o blank o k, with the repeated o separated by blank.
    sequence = [book[0], book[1], blank, book[2], book[3]]
    assert vocabulary.decode(greedy_decode(_peaked(sequence, classes), blank=blank)[0]) == "book"


def test_lengths_stop_decoding():
    vocabulary = Vocabulary()
    code = vocabulary.encode("ab")
    log_probs = _peaked([code[0], code[1], code[0]], vocabulary.size)
    decoded = greedy_decode(log_probs, lengths=torch.tensor([2]), blank=vocabulary.blank)
    assert vocabulary.decode(decoded[0]) == "ab"


def test_beam_search_matches_a_clear_path():
    vocabulary = Vocabulary()
    classes = vocabulary.size
    blank = vocabulary.blank
    text = "ab"
    encoded = vocabulary.encode(text)
    log_probs = _peaked([encoded[0], blank, encoded[1]], classes).squeeze(0)
    assert vocabulary.decode(ctc_beam_search(log_probs, beam_width=4, blank=blank)) == text
    repeated = _peaked([encoded[0], blank, encoded[0]], classes).squeeze(0)
    assert vocabulary.decode(ctc_beam_search(repeated, beam_width=4, blank=blank)) == "aa"
