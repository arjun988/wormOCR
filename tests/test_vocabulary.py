from wormocr.vocabulary import Vocabulary, default_charset


def test_roundtrip_and_blank_index():
    vocabulary = Vocabulary()
    assert vocabulary.blank == 0
    assert vocabulary.size == len(default_charset()) + 1
    text = "Hello, wormOCR 123!"
    assert vocabulary.decode(vocabulary.encode(text)) == text


def test_unknown_character_is_rejected():
    vocabulary = Vocabulary()
    try:
        vocabulary.encode("café")
    except ValueError as exc:
        assert "é" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_repeated_characters_stay_in_the_label():
    vocabulary = Vocabulary()
    encoded = vocabulary.encode("bookkeeper")
    assert vocabulary.decode(encoded) == "bookkeeper"
    assert encoded[1] == encoded[2]
