"""Recognizer architectures."""

from wormocr.models.crnn import CRNN, build_model, count_parameters

__all__ = ["CRNN", "build_model", "count_parameters"]
