"""CTC decoders.

Greedy decoding is the default. Beam search is optional and only worth its cost
when a measurement shows fewer errors.
"""

from __future__ import annotations

import math

import torch


def _logadd(left: float, right: float) -> float:
    if left == float("-inf"):
        return right
    if right == float("-inf"):
        return left
    if left > right:
        return left + math.log1p(math.exp(right - left))
    return right + math.log1p(math.exp(left - right))


def greedy_decode(
    log_probs: torch.Tensor,
    lengths: torch.Tensor | None = None,
    blank: int = 0,
) -> list[list[int]]:
    """Collapse CTC greedy paths. Repeated labels are kept when a blank separates them."""
    if log_probs.ndim != 3:
        raise ValueError("log_probs must have shape (batch, time, classes)")
    chosen = log_probs.argmax(dim=-1).detach().cpu()
    batch, time, _classes = log_probs.shape
    decoded: list[list[int]] = []
    for index in range(batch):
        limit = time if lengths is None else max(0, min(time, int(lengths[index])))
        previous = blank
        sequence: list[int] = []
        row = chosen[index]
        for step in range(limit):
            label = int(row[step])
            if label != blank and label != previous:
                sequence.append(label)
            previous = label
        decoded.append(sequence)
    return decoded


def greedy_confidence(
    log_probs: torch.Tensor,
    length: int,
    blank: int = 0,
) -> float:
    """Mean probability of the labels that greedy decoding actually emitted."""
    limit = max(0, min(int(length), int(log_probs.shape[0])))
    if limit == 0:
        return 0.0
    probabilities = log_probs[:limit].detach().float().exp()
    labels = probabilities.argmax(dim=-1)
    chosen: list[float] = []
    previous = blank
    for step, label in enumerate(labels.tolist()):
        label = int(label)
        if label != blank and label != previous:
            chosen.append(float(probabilities[step, label]))
        previous = label
    if not chosen:
        return float(probabilities[:, blank].mean())
    return float(sum(chosen) / len(chosen))


def ctc_beam_search(log_probs: torch.Tensor, beam_width: int = 8, blank: int = 0) -> list[int]:
    """Prefix beam search for one sequence of shape (time, classes)."""
    if beam_width < 1:
        raise ValueError("beam_width must be at least 1")
    if log_probs.ndim != 2:
        raise ValueError("beam search expects log_probs of shape (time, classes)")
    frame_log = log_probs.detach().float().cpu()
    time, classes = frame_log.shape
    neg = float("-inf")
    beam: dict[tuple[int, ...], list[float]] = {(): [0.0, neg]}

    def add(store: dict[tuple[int, ...], list[float]], prefix: tuple[int, ...], blank_p: float, non_p: float) -> None:
        slot = store.get(prefix)
        if slot is None:
            store[prefix] = [blank_p, non_p]
            return
        slot[0] = _logadd(slot[0], blank_p)
        slot[1] = _logadd(slot[1], non_p)

    for step in range(time):
        frame = frame_log[step]
        ranked = sorted(beam.items(), key=lambda item: _logadd(item[1][0], item[1][1]), reverse=True)[
            :beam_width
        ]
        nxt: dict[tuple[int, ...], list[float]] = {}
        for prefix, (prob_blank, prob_label) in ranked:
            total = _logadd(prob_blank, prob_label)
            add(nxt, prefix, total + float(frame[blank]), neg)
            for label in range(classes):
                if label == blank:
                    continue
                emit = float(frame[label])
                if prefix and label == prefix[-1]:
                    add(nxt, prefix, neg, prob_label + emit)
                    add(nxt, prefix + (label,), neg, prob_blank + emit)
                else:
                    add(nxt, prefix + (label,), neg, total + emit)
        beam = dict(
            sorted(nxt.items(), key=lambda item: _logadd(item[1][0], item[1][1]), reverse=True)[:beam_width]
        )
    best = max(beam.items(), key=lambda item: _logadd(item[1][0], item[1][1]))
    return list(best[0])
