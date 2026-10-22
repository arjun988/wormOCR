"""A small standard-library HTTP server for the browser UI.

The server binds to localhost by default. Uploaded files are processed in
memory or in a temporary file that is deleted after the request.

`/api/ocr/stream` runs the same pipeline as `OCR.predict`, one step at a time,
and writes one JSON event per line (NDJSON) as each step finishes.
"""

from __future__ import annotations

import base64
import io
import json
import mimetypes
import os
import tempfile
import threading
import time
from collections.abc import Iterator
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import torch
from PIL import Image, ImageOps, UnidentifiedImageError

from wormocr.detection.lines import segment_page
from wormocr.exceptions import RecognitionError, WormOCRError
from wormocr.preprocessing import looks_like_line, prepare_line

STATIC_DIR = Path(__file__).resolve().parent / "static"
MAX_UPLOAD_BYTES = 25 * 1024 * 1024
MAX_PDF_PAGES = 20
MAX_LINES_PER_PAGE = 400
PDF_SCALE = 2.0
PREVIEW_MAX_SIDE = 1600
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp", ".gif"}


def _preview(image: Image.Image) -> str:
    preview = image.convert("RGB")
    preview.thumbnail((PREVIEW_MAX_SIDE, PREVIEW_MAX_SIDE), Image.Resampling.BILINEAR, reducing_gap=2.0)
    buffer = io.BytesIO()
    preview.save(buffer, format="JPEG", quality=82)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def _ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000.0, 1)


def _log(stage: str, message: str, **extra) -> dict:
    return {"type": "log", "stage": stage, "message": message, **extra}


class OCRService:
    """Holds one loaded line engine and serializes calls to it."""

    def __init__(self, engine, model_info: dict) -> None:
        self.engine = engine
        self.model_info = model_info
        self._lock = threading.Lock()

    def stream(self, data: bytes, filename: str, mode: str) -> Iterator[dict]:
        started = time.perf_counter()
        timings = {"decode": 0.0, "layout": 0.0, "recognize": 0.0}
        info = self.model_info
        yield {"type": "start", "filename": filename, "bytes": len(data), "mode": mode, "model": info}
        yield _log("upload", f"Received {filename} ({len(data) / 1024:.0f} KB)")
        yield _log(
            "model",
            f"{info['name']} ready: {info['parameters']:,} parameters, "
            f"{info.get('architecture', 'CRNN + CTC')}, on {info['device']}",
        )

        suffix = Path(filename).suffix.lower()
        is_pdf = suffix == ".pdf" or data[:5] == b"%PDF-"
        if not is_pdf and suffix and suffix not in IMAGE_SUFFIXES:
            raise WormOCRError(f"unsupported file type {suffix!r}. Upload a PDF or an image.")

        pages_text: list[str] = []
        confidences: list[float] = []
        warnings: list[str] = []
        line_total = 0
        with self._lock:
            step = time.perf_counter()
            if is_pdf:
                pages = self._pdf_pages(data)
                yield _log(
                    "decode",
                    f"Rendered {len(pages)} PDF page(s) at {PDF_SCALE:g}x scale with pdfium",
                    ms=_ms(step),
                )
            else:
                pages = None
            page_count = len(pages) if pages is not None else 1
            timings["decode"] += _ms(step)
            yield {"type": "document", "kind": "pdf" if is_pdf else "image", "pages": page_count}

            for index in range(page_count):
                step = time.perf_counter()
                if pages is None:
                    image = self._image(data)
                    yield _log("decode", f"Decoded image: {image.width}x{image.height} px", ms=_ms(step))
                else:
                    image = pages[index]
                    yield _log("decode", f"Page {index + 1}/{page_count}: {image.width}x{image.height} px", page=index)
                timings["decode"] += _ms(step)
                yield {
                    "type": "page",
                    "page": index,
                    "width": image.width,
                    "height": image.height,
                    "preview": _preview(image),
                }

                page_lines: list[dict] = []
                for event in self._recognize_page(image, index, mode, timings, warnings, page_lines):
                    yield event
                line_total += len(page_lines)
                confidences.extend(line["confidence"] for line in page_lines)
                page_text = "\n".join(line["text"] for line in page_lines)
                pages_text.append(page_text)
                yield {"type": "page_done", "page": index, "text": page_text}

        mean = sum(confidences) / len(confidences) if confidences else None
        elapsed = _ms(started)
        summary = f"Done in {elapsed / 1000:.2f} s: {line_total} line(s)"
        if mean is not None:
            summary += f", mean confidence {mean * 100:.1f}%"
        yield _log("done", summary)
        yield {
            "type": "done",
            "text": "\n\n".join(pages_text),
            "line_count": line_total,
            "mean_confidence": mean,
            "elapsed_ms": elapsed,
            "timings": timings,
            "warnings": warnings,
        }

    def _recognize_page(
        self,
        image: Image.Image,
        page: int,
        mode: str,
        timings: dict,
        warnings: list[str],
        page_lines: list[dict],
    ) -> Iterator[dict]:
        engine = self.engine
        step = time.perf_counter()
        single = mode == "line" or (mode == "auto" and looks_like_line(image))
        if single:
            reason = "forced by mode" if mode == "line" else f"image is only {image.height} px tall"
            yield _log("layout", f"Single-line mode ({reason}); skipping line detection", page=page)
            boxes = [(0, 0, image.width, image.height)]
            crops = [image]
            layout_name, angle = "line", 0.0
        else:
            if mode == "auto":
                yield _log("layout", "Auto mode chose page layout: the image is taller than one text line", page=page)
            layout = segment_page(image)
            angle = layout.angle
            layout_name = layout.layout
            yield _log(
                "layout",
                f"Deskew: rotated {angle:+.1f} degrees" if angle else "Deskew: page is straight, no rotation needed",
                page=page,
            )
            boxes = [box.as_xywh() for box in layout.boxes]
            crops = list(layout.crops)
            if not crops:
                message = "No text lines detected; reading the whole image as one line"
                warnings.append(message)
                yield _log("layout", message, page=page, level="warn")
                boxes, crops, layout_name = [(0, 0, image.width, image.height)], [image], "fallback-line"
            elif len(crops) > MAX_LINES_PER_PAGE:
                message = f"More than {MAX_LINES_PER_PAGE} regions found; keeping the first {MAX_LINES_PER_PAGE}"
                warnings.append(message)
                yield _log("layout", message, page=page, level="warn")
                boxes, crops = boxes[:MAX_LINES_PER_PAGE], crops[:MAX_LINES_PER_PAGE]
            yield _log(
                "layout",
                f"Projection profile found {len(crops)} text line(s), {layout_name} reading order",
                page=page,
                ms=_ms(step),
            )
        timings["layout"] += _ms(step)
        yield {
            "type": "layout",
            "page": page,
            "layout": layout_name,
            "angle": angle,
            "boxes": [{"x": b[0], "y": b[1], "width": b[2], "height": b[3]} for b in boxes],
        }

        batch_size = max(1, int(getattr(engine, "batch_size", 8)))
        batches = (len(crops) + batch_size - 1) // batch_size
        for batch_index, start in enumerate(range(0, len(crops), batch_size)):
            chunk = crops[start : start + batch_size]
            frames = [self._frames(crop) for crop in chunk]
            step = time.perf_counter()
            recognized = engine.recognize(chunk)
            spent = _ms(step)
            timings["recognize"] += spent
            yield _log(
                "recognize",
                f"Batch {batch_index + 1}/{batches}: {len(chunk)} line(s), CNN + BiGRU + greedy CTC, "
                f"{min(frames)}-{max(frames)} time steps",
                page=page,
                ms=spent,
            )
            for offset, item in enumerate(recognized):
                order = start + offset
                x, y, width, height = boxes[order]
                line = {
                    "page": page,
                    "order": order,
                    "text": str(item["text"]),
                    "confidence": float(item["confidence"]),
                    "bbox": {"x": x, "y": y, "width": width, "height": height},
                    "frames": frames[offset],
                }
                page_lines.append(line)
                yield {"type": "line", **line}

    def _frames(self, crop: Image.Image) -> int:
        engine = self.engine
        tensor, _ = prepare_line(crop, height=engine.image_height, align=2, max_width=engine.max_width)
        width = torch.tensor([int(tensor.shape[-1])], dtype=torch.long)
        return int(engine.model.output_lengths(width)[0])

    def run(self, data: bytes, filename: str, mode: str) -> dict:
        """Run the stream to completion and return one combined result."""
        pages: dict[int, dict] = {}
        kind = "image"
        final: dict = {}
        for event in self.stream(data, filename, mode):
            kind_of = event["type"]
            if kind_of == "document":
                kind = event["kind"]
            elif kind_of == "page":
                pages[event["page"]] = {
                    "preview": event["preview"],
                    "image_width": event["width"],
                    "image_height": event["height"],
                    "lines": [],
                    "text": "",
                }
            elif kind_of == "layout":
                pages[event["page"]].update(layout=event["layout"], deskew_degrees=event["angle"])
            elif kind_of == "line":
                pages[event["page"]]["lines"].append({k: v for k, v in event.items() if k != "type"})
            elif kind_of == "page_done":
                pages[event["page"]]["text"] = event["text"]
            elif kind_of == "done":
                final = event
        return {
            "filename": filename,
            "kind": kind,
            "mode": mode,
            "text": final.get("text", ""),
            "pages": [pages[index] for index in sorted(pages)],
            "line_count": final.get("line_count", 0),
            "mean_confidence": final.get("mean_confidence"),
            "elapsed_ms": final.get("elapsed_ms"),
            "timings": final.get("timings", {}),
            "warnings": final.get("warnings", []),
            "model": self.model_info,
        }

    @staticmethod
    def _image(data: bytes) -> Image.Image:
        try:
            image = Image.open(io.BytesIO(data))
            image.load()
        except (UnidentifiedImageError, OSError) as exc:
            raise WormOCRError("the upload is not a readable image") from exc
        return ImageOps.exif_transpose(image).convert("RGB")

    @staticmethod
    def _pdf_pages(data: bytes) -> list[Image.Image]:
        try:
            import pypdfium2 as pdfium
        except ImportError as exc:
            raise RecognitionError(
                "Reading PDFs requires the optional package pypdfium2. Install it with: pip install pypdfium2"
            ) from exc
        handle = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
        path = Path(handle.name)
        try:
            with handle:
                handle.write(data)
            try:
                document = pdfium.PdfDocument(str(path))
            except Exception as exc:  # noqa: BLE001
                raise WormOCRError("the upload is not a readable PDF") from exc
            try:
                count = len(document)
                if count == 0:
                    raise WormOCRError("the PDF has no pages")
                if count > MAX_PDF_PAGES:
                    raise WormOCRError(f"the PDF has {count} pages; the web UI accepts up to {MAX_PDF_PAGES}")
                images = [document[index].render(scale=PDF_SCALE).to_pil().convert("RGB") for index in range(count)]
            finally:
                document.close()
        finally:
            path.unlink(missing_ok=True)
        return images


def _handler(service: OCRService) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "wormOCR"

        def log_message(self, format: str, *args) -> None:  # noqa: A002
            pass

        def _send(self, status: int, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, status: int, payload: dict) -> None:
            self._send(status, json.dumps(payload).encode("utf-8"), "application/json")

        def do_GET(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
            if path == "/api/health":
                self._json(HTTPStatus.OK, {"status": "ok", "model": service.model_info})
                return
            name = "index.html" if path in {"/", "/index.html"} else path.removeprefix("/static/")
            target = (STATIC_DIR / name).resolve()
            if STATIC_DIR not in target.parents or not target.is_file():
                self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})
                return
            content_type = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
            if content_type.startswith("text/") or content_type.endswith("javascript"):
                content_type += "; charset=utf-8"
            self._send(HTTPStatus.OK, target.read_bytes(), content_type)

        def _read_upload(self) -> tuple[bytes, str, str] | None:
            query = parse_qs(urlparse(self.path).query)
            mode = query.get("mode", ["auto"])[0]
            filename = query.get("filename", ["upload"])[0]
            if mode not in {"auto", "line", "page"}:
                self._json(HTTPStatus.BAD_REQUEST, {"error": "mode must be auto, line, or page"})
                return None
            length = int(self.headers.get("Content-Length") or 0)
            if length <= 0:
                self._json(HTTPStatus.BAD_REQUEST, {"error": "empty upload"})
                return None
            if length > MAX_UPLOAD_BYTES:
                self._json(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {"error": "files up to 25 MB are accepted"})
                return None
            return self.rfile.read(length), filename, mode

        def do_POST(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
            if path == "/api/ocr":
                upload = self._read_upload()
                if upload is None:
                    return
                try:
                    payload = service.run(*upload)
                except WormOCRError as exc:
                    self._json(HTTPStatus.UNPROCESSABLE_ENTITY, {"error": str(exc)})
                    return
                except Exception as exc:  # noqa: BLE001
                    self._json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": f"recognition failed: {exc}"})
                    return
                self._json(HTTPStatus.OK, payload)
                return
            if path == "/api/ocr/stream":
                upload = self._read_upload()
                if upload is None:
                    return
                self._stream(*upload)
                return
            self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})

        def _stream(self, data: bytes, filename: str, mode: str) -> None:
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True
            clock = time.perf_counter()

            def emit(event: dict) -> None:
                event.setdefault("t", round((time.perf_counter() - clock) * 1000.0, 1))
                self.wfile.write((json.dumps(event) + "\n").encode("utf-8"))
                self.wfile.flush()

            try:
                for event in service.stream(data, filename, mode):
                    emit(event)
            except (BrokenPipeError, ConnectionResetError):
                return
            except WormOCRError as exc:
                emit({"type": "error", "message": str(exc)})
            except Exception as exc:  # noqa: BLE001
                emit({"type": "error", "message": f"recognition failed: {exc}"})

    return Handler


def make_server(engine, model_info: dict, host: str = "127.0.0.1", port: int = 8000) -> ThreadingHTTPServer:
    return ThreadingHTTPServer((host, port), _handler(OCRService(engine, model_info)))


def describe_engine(engine, name: str, device: str) -> dict:
    from wormocr.models.crnn import count_parameters

    model_cfg = engine.config.get("model", {})
    hidden = model_cfg.get("rnn_hidden")
    layers = model_cfg.get("rnn_layers")
    architecture = "CNN encoder, "
    architecture += f"{layers}-layer BiGRU ({hidden} hidden), " if hidden and layers else "BiGRU, "
    architecture += "CTC head"
    return {
        "name": name,
        "parameters": count_parameters(engine.model),
        "maturity": engine.maturity,
        "device": device,
        "architecture": architecture,
        "charset_size": engine.vocabulary.size - 1,
        "image_height": engine.image_height,
    }


def default_threads() -> int:
    """Half the logical CPUs, capped at 16.

    PyTorch defaults to every logical CPU, which oversubscribes the GRU on
    hybrid and hyper-threaded CPUs and makes recognition slower, not faster.
    """
    return max(1, min(16, (os.cpu_count() or 2) // 2))


def disable_power_throttling() -> bool:
    """Opt the process out of Windows EcoQoS execution-speed throttling.

    Windows moves windowless background processes such as this server onto
    efficiency cores after a few seconds, which makes OCR 2-3x slower.
    """
    if os.name != "nt":
        return False
    import ctypes
    from ctypes import wintypes

    class PowerThrottlingState(ctypes.Structure):
        _fields_ = [("Version", wintypes.ULONG), ("ControlMask", wintypes.ULONG), ("StateMask", wintypes.ULONG)]

    process_power_throttling = 4
    execution_speed = 0x1
    state = PowerThrottlingState(1, execution_speed, 0)
    try:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        kernel32.SetProcessInformation.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
        kernel32.SetProcessInformation.restype = wintypes.BOOL
        return bool(
            kernel32.SetProcessInformation(
                kernel32.GetCurrentProcess(), process_power_throttling, ctypes.byref(state), ctypes.sizeof(state)
            )
        )
    except (AttributeError, OSError):
        return False


def serve(
    model: str = "tiny",
    checkpoint: str | None = None,
    device: str | None = "cpu",
    host: str = "127.0.0.1",
    port: int = 8000,
    threads: int | None = None,
) -> None:
    from wormocr.api import OCR

    disable_power_throttling()
    torch.set_num_threads(int(threads) if threads else default_threads())
    engine = OCR(model=model, checkpoint=checkpoint, device=device).engine
    engine.batch_size = 16
    info = describe_engine(engine, f"wormOCR {model}", str(engine.device))
    info["threads"] = torch.get_num_threads()
    server = make_server(engine, info, host, port)
    print(f"{info['name']} ({info['parameters'] / 1e6:.2f}M parameters) ready at http://{host}:{port}", flush=True)
    print("Press Ctrl+C to stop.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
