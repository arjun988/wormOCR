"""Synthetic and file-backed OCR datasets."""

from wormocr.data.dataset import LabelDataset, OnlineSynthDataset, collate_lines

__all__ = ["LabelDataset", "OnlineSynthDataset", "collate_lines"]
