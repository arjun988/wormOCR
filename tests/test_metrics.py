from wormocr.evaluation.errors import ErrorReport, align
from wormocr.evaluation.metrics import cer, edit_distance, wer


def test_cer_and_wer_do_not_hide_spaces():
    assert cer("cat", "cat") == 0
    assert cer("cat", "car") == 1 / 3
    assert wer("hello world", "hello world") == 0
    assert wer("hello world", "hello") == 0.5
    assert edit_distance("book", "bok") == 1
    assert cer("a  b", "a b") > 0
    assert wer("a  b", "a b") > 0


def test_alignment_reports_a_repeated_character_deletion():
    operations = align("book", "bok")
    assert ("del", "o", None) in operations or any(op[0] == "del" for op in operations)
    report = ErrorReport()
    report.add("book", "bok", cer("book", "bok"))
    report.add("Hello!", "Hello", cer("Hello!", "Hello"))
    assert report.deletions >= 1
    assert report.punctuation_errors >= 1
    assert report.repeated_character_failures >= 1
    text = report.to_markdown()
    assert "book" in text
