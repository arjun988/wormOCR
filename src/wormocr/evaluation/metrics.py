"""Character and word error rates.

CER counts every character, including spaces. WER splits on single spaces, so a
doubled space is its own token and is not collapsed away. Exact match requires
the full strings to be equal.
"""

from __future__ import annotations

from dataclasses import dataclass


def edit_distance(reference: list | str, hypothesis: list | str) -> int:
    if len(reference) < len(hypothesis):
        reference, hypothesis = hypothesis, reference
    previous = list(range(len(hypothesis) + 1))
    for index, left in enumerate(reference, start=1):
        current = [index]
        for inner, right in enumerate(hypothesis, start=1):
            insert = current[inner - 1] + 1
            delete = previous[inner] + 1
            replace = previous[inner - 1] + (left != right)
            current.append(min(insert, delete, replace))
        previous = current
    return previous[-1]


def cer(reference: str, hypothesis: str) -> float:
    if not reference and not hypothesis:
        return 0.0
    return edit_distance(reference, hypothesis) / max(len(reference), 1)


def words(text: str) -> list[str]:
    return text.split(" ")


def wer(reference: str, hypothesis: str) -> float:
    reference_words = words(reference)
    hypothesis_words = words(hypothesis)
    if not reference_words and not hypothesis_words:
        return 0.0
    return edit_distance(reference_words, hypothesis_words) / max(len(reference_words), 1)


@dataclass
class MetricTotals:
    characters: int = 0
    character_edits: int = 0
    words: int = 0
    word_edits: int = 0
    exact: int = 0
    samples: int = 0
    loss_sum: float = 0.0
    loss_batches: int = 0

    def update(self, reference: str, hypothesis: str) -> None:
        self.samples += 1
        self.characters += len(reference)
        self.character_edits += edit_distance(reference, hypothesis)
        reference_words = words(reference)
        self.words += len(reference_words)
        self.word_edits += edit_distance(reference_words, words(hypothesis))
        self.exact += int(reference == hypothesis)

    def update_loss(self, loss: float) -> None:
        self.loss_sum += float(loss)
        self.loss_batches += 1

    @property
    def corpus_cer(self) -> float:
        if self.characters == 0:
            return 0.0
        return self.character_edits / self.characters

    @property
    def corpus_wer(self) -> float:
        if self.words == 0:
            return 0.0
        return self.word_edits / self.words

    @property
    def exact_match(self) -> float:
        if self.samples == 0:
            return 0.0
        return self.exact / self.samples

    @property
    def mean_loss(self) -> float | None:
        if self.loss_batches == 0:
            return None
        return self.loss_sum / self.loss_batches

    def to_dict(self) -> dict[str, float | int | None]:
        return {
            "samples": self.samples,
            "cer": self.corpus_cer,
            "wer": self.corpus_wer,
            "exact_match": self.exact_match,
            "character_edits": self.character_edits,
            "characters": self.characters,
            "word_edits": self.word_edits,
            "words": self.words,
            "loss": self.mean_loss,
        }
