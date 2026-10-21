"""Python API for line and simple-page OCR."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

from wormocr.config import repo_root
from wormocr.detection.lines import segment_page
from wormocr.exceptions import CheckpointError, RecognitionError
from wormocr.inference.engine import LineEngine
from wormocr.preprocessing import coerce_image, looks_like_line


@dataclass
class LineResult:
    text: str
    confidence: float
    bbox: tuple[int, int, int, int]
    order: int
    page: int = 0

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "confidence": self.confidence,
            "bbox": {"x": self.bbox[0], "y": self.bbox[1], "width": self.bbox[2], "height": self.bbox[3]},
            "order": self.order,
            "page": self.page,
        }


@dataclass
class OCRResult:
    text: str
    lines: list[LineResult]
    image_width: int
    image_height: int
    warnings: list[str] = field(default_factory=list)
    deskew_degrees: float = 0.0
    layout: str = "line"
    maturity: str = "untrained"
    pages: list["OCRResult"] | None = None

    def to_dict(self) -> dict:
        payload = {
            "text": self.text,
            "lines": [line.to_dict() for line in self.lines],
            "image_width": self.image_width,
            "image_height": self.image_height,
            "warnings": list(self.warnings),
            "deskew_degrees": self.deskew_degrees,
            "layout": self.layout,
            "maturity": self.maturity,
        }
        if self.pages is not None:
            payload["pages"] = [page.to_dict() for page in self.pages]
        return payload


def resolve_checkpoint(model: str, checkpoint: str | Path | None) -> Path:
    if checkpoint:
        path = Path(checkpoint)
        if not path.exists():
            raise CheckpointError(
                f"checkpoint not found: {path}. wormOCR does not download weights."
            )
        return path
    relative = Path("artifacts") / "checkpoints" / model / "best.pt"
    candidates = [
        Path.cwd() / relative,
        repo_root() / relative,
        Path.cwd() / "artifacts" / "checkpoints" / model / "best_weights.pt",
        repo_root() / "artifacts" / "checkpoints" / model / "best_weights.pt",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise CheckpointError(
        f"No local checkpoint for {model!r}. wormOCR does not download weights. "
        f"Train one with: wormocr train --config configs/{model}.yaml --mode baseline"
    )


class OCR:
    """Local recognizer. Constructing it never downloads weights."""

    def __init__(
        self,
        model: str = "tiny",
        checkpoint: str | Path | None = None,
        device: str | None = None,
        beam_width: int = 0,
    ) -> None:
        self.model_name = model
        self.checkpoint_path = resolve_checkpoint(model, checkpoint)
        self.engine = LineEngine.load(self.checkpoint_path, device=device, beam_width=beam_width)

    def predict(
        self,
        image: str | Path | Image.Image,
        mode: str = "auto",
        correct_orientation: bool = True,
    ) -> OCRResult:
        if mode not in {"auto", "line", "page"}:
            raise RecognitionError("mode must be auto, line, or page")
        if isinstance(image, (str, Path)) and Path(image).suffix.lower() == ".pdf":
            return self._predict_pdf(Path(image), mode, correct_orientation)
        pil = coerce_image(image)
        if mode == "line" or (mode == "auto" and looks_like_line(pil)):
            return self._recognize_line(pil)
        return self._recognize_page(pil, correct_orientation)

    def predict_batch(self, images: list, mode: str = "auto", correct_orientation: bool = True) -> list[OCRResult]:
        return [self.predict(image, mode=mode, correct_orientation=correct_orientation) for image in images]

    def _recognize_line(self, image: Image.Image) -> OCRResult:
        recognized = self.engine.recognize([image])[0]
        width, height = image.size
        line = LineResult(
            text=str(recognized["text"]),
            confidence=float(recognized["confidence"]),
            bbox=(0, 0, width, height),
            order=0,
        )
        return OCRResult(
            text=line.text,
            lines=[line],
            image_width=width,
            image_height=height,
            layout="line",
            maturity=self.engine.maturity,
        )

    def _recognize_page(self, image: Image.Image, correct_orientation: bool) -> OCRResult:
        warnings: list[str] = []
        layout = segment_page(image, correct_orientation=correct_orientation)
        if not layout.crops:
            warnings.append("no text lines detected; recognized the whole image as one line")
            result = self._recognize_line(image)
            result.warnings = warnings
            result.layout = "fallback-line"
            return result
        if len(layout.crops) > 400:
            warnings.append("more than 400 regions detected; kept the first 400 in reading order")
            layout.boxes = layout.boxes[:400]
            layout.crops = layout.crops[:400]
        recognized = self.engine.recognize(layout.crops)
        lines: list[LineResult] = []
        for order, (box, item) in enumerate(zip(layout.boxes, recognized)):
            lines.append(
                LineResult(
                    text=str(item["text"]),
                    confidence=float(item["confidence"]),
                    bbox=box.as_xywh(),
                    order=order,
                )
            )
        return OCRResult(
            text="\n".join(line.text for line in lines),
            lines=lines,
            image_width=image.width,
            image_height=image.height,
            warnings=warnings,
            deskew_degrees=layout.angle,
            layout=layout.layout,
            maturity=self.engine.maturity,
        )

    def _predict_pdf(self, path: Path, mode: str, correct_orientation: bool) -> OCRResult:
        pages = [_page_result(self, page, mode, correct_orientation) for page in load_pdf(path)]
        lines: list[LineResult] = []
        warnings: list[str] = []
        for index, page in enumerate(pages):
            warnings.extend(page.warnings)
            for line in page.lines:
                line.page = index
                lines.append(line)
        return OCRResult(
            text="\n\n".join(page.text for page in pages),
            lines=lines,
            image_width=pages[0].image_width if pages else 0,
            image_height=pages[0].image_height if pages else 0,
            warnings=warnings,
            layout="pdf",
            maturity=self.engine.maturity,
            pages=pages,
        )


def _page_result(ocr: OCR, image: Image.Image, mode: str, correct_orientation: bool) -> OCRResult:
    if mode == "line":
        return ocr._recognize_line(image)
    if mode == "page" or not looks_like_line(image):
        return ocr._recognize_page(image, correct_orientation)
    return ocr._recognize_line(image)


def load_pdf(path: Path, scale: float = 2.0) -> list[Image.Image]:
    try:
        import pypdfium2 as pdfium
    except ImportError as exc:
        raise RecognitionError(
            "Reading PDFs requires the optional package pypdfium2. "
            "Install it with: pip install pypdfium2"
        ) from exc
    document = pdfium.PdfDocument(str(path))
    images: list[Image.Image] = []
    try:
        for index in range(len(document)):
            page = document[index]
            bitmap = page.render(scale=scale)
            images.append(bitmap.to_pil().convert("RGB"))
    finally:
        document.close()
    if not images:
        raise RecognitionError(f"PDF has no pages: {path}")
    return images
