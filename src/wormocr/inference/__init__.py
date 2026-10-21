"""Decoding and recognition runtime."""

from wormocr.inference.decode import ctc_beam_search, greedy_decode

__all__ = ["ctc_beam_search", "greedy_decode"]
