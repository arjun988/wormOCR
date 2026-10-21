"""Character-level error analysis for recognized lines."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

PUNCTUATION = set("!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~")


def align(reference: str, hypothesis: str) -> list[tuple[str, str | None, str | None]]:
    """Return edit operations in reading order: eq, sub, del, ins."""
    ref_len, hyp_len = len(reference), len(hypothesis)
    scores = [[0] * (hyp_len + 1) for _ in range(ref_len + 1)]
    for i in range(1, ref_len + 1):
        scores[i][0] = i
    for j in range(1, hyp_len + 1):
        scores[0][j] = j
    for i in range(1, ref_len + 1):
        for j in range(1, hyp_len + 1):
            cost = 0 if reference[i - 1] == hypothesis[j - 1] else 1
            scores[i][j] = min(
                scores[i - 1][j] + 1,
                scores[i][j - 1] + 1,
                scores[i - 1][j - 1] + cost,
            )
    operations: list[tuple[str, str | None, str | None]] = []
    i, j = ref_len, hyp_len
    while i > 0 or j > 0:
        if (
            i > 0
            and j > 0
            and reference[i - 1] == hypothesis[j - 1]
            and scores[i][j] == scores[i - 1][j - 1]
        ):
            operations.append(("eq", reference[i - 1], hypothesis[j - 1]))
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and scores[i][j] == scores[i - 1][j - 1] + 1:
            operations.append(("sub", reference[i - 1], hypothesis[j - 1]))
            i -= 1
            j -= 1
        elif i > 0 and scores[i][j] == scores[i - 1][j] + 1:
            operations.append(("del", reference[i - 1], None))
            i -= 1
        else:
            operations.append(("ins", None, hypothesis[j - 1]))
            j -= 1
    operations.reverse()
    return operations


@dataclass
class ErrorReport:
    samples: int = 0
    substitutions: int = 0
    insertions: int = 0
    deletions: int = 0
    punctuation_errors: int = 0
    case_only_substitutions: int = 0
    repeated_character_failures: int = 0
    confusion: Counter = field(default_factory=Counter)
    worst: list[dict[str, float | str]] = field(default_factory=list)

    def add(self, reference: str, hypothesis: str, cer_value: float) -> None:
        self.samples += 1
        operations = align(reference, hypothesis)
        for kind, ref_char, hyp_char in operations:
            if kind == "sub":
                self.substitutions += 1
                self.confusion[(ref_char, hyp_char)] += 1
                if ref_char and hyp_char and ref_char.lower() == hyp_char.lower() and ref_char != hyp_char:
                    self.case_only_substitutions += 1
                if ref_char in PUNCTUATION or (hyp_char in PUNCTUATION if hyp_char else False):
                    self.punctuation_errors += 1
            elif kind == "ins":
                self.insertions += 1
                if hyp_char in PUNCTUATION:
                    self.punctuation_errors += 1
            elif kind == "del":
                self.deletions += 1
                if ref_char in PUNCTUATION:
                    self.punctuation_errors += 1
        self.repeated_character_failures += _repeated_failures(reference, operations)
        self.worst.append({"reference": reference, "hypothesis": hypothesis, "cer": cer_value})

    def to_markdown(self) -> str:
        worst = sorted(self.worst, key=lambda item: float(item["cer"]), reverse=True)[:8]
        lines = [
            "# wormOCR error analysis",
            "",
            f"Samples: {self.samples}",
            "",
            "## Edit operations",
            "",
            f"- substitutions: {self.substitutions}",
            f"- insertions: {self.insertions}",
            f"- deletions: {self.deletions}",
            f"- punctuation errors: {self.punctuation_errors}",
            f"- case-only substitutions: {self.case_only_substitutions}",
            f"- repeated-character run failures: {self.repeated_character_failures}",
            "",
            "## Top substitutions",
            "",
            "| reference | hypothesis | count |",
            "| --- | --- | ---: |",
        ]
        for (ref_char, hyp_char), count in self.confusion.most_common(15):
            lines.append(f"| `{_show(ref_char)}` | `{_show(hyp_char)}` | {count} |")
        if not self.confusion:
            lines.append("|  |  | 0 |")
        lines.extend(["", "## Highest-CER samples", ""])
        if not worst:
            lines.append("No samples.")
        for item in worst:
            lines.append(f"- CER {float(item['cer']):.3f}")
            lines.append(f"  - ref: `{item['reference']}`")
            lines.append(f"  - hyp: `{item['hypothesis']}`")
        lines.append("")
        return "\n".join(lines)


def _show(char: str | None) -> str:
    if char is None:
        return ""
    if char == " ":
        return "␣"
    return char


def _repeated_failures(reference: str, operations: list[tuple[str, str | None, str | None]]) -> int:
    ref_ops = [operation for operation in operations if operation[0] != "ins"]
    failures = 0
    index = 0
    while index < len(reference):
        end = index + 1
        while end < len(reference) and reference[end] == reference[index]:
            end += 1
        run = end - index
        if run >= 2:
            matched = sum(1 for op in ref_ops[index:end] if op[0] == "eq")
            if matched < run:
                failures += 1
        index = end
    return failures
