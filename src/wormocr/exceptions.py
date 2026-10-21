"""Errors raised by wormOCR. They are intentional and safe to show to users."""


class WormOCRError(Exception):
    """Base class for expected wormOCR failures."""


class ConfigError(WormOCRError):
    """A configuration file or override is invalid."""


class CheckpointError(WormOCRError):
    """A checkpoint is missing, incompatible, or unusable."""


class DataError(WormOCRError):
    """A training or evaluation sample cannot be used."""


class RecognitionError(WormOCRError):
    """An image cannot be recognized."""


class ExportError(WormOCRError):
    """A deployment artifact could not be created or verified."""
