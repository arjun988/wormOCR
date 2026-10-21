"""wormOCR: a small CTC recognizer for printed English text."""

__version__ = "0.1.0"


def __getattr__(name: str):
    if name == "OCR":
        from wormocr.api import OCR

        return OCR
    if name == "WormOCRError":
        from wormocr.exceptions import WormOCRError

        return WormOCRError
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["OCR", "WormOCRError", "__version__"]
