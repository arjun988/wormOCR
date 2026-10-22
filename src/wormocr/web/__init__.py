"""Local web UI for trying wormOCR in a browser."""

from wormocr.web.server import describe_engine, make_server, serve

__all__ = ["describe_engine", "make_server", "serve"]
