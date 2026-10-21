"""Locate installed fonts that can draw Latin letters.

wormOCR does not ship font files. Rendering uses fonts already on the machine.
"""

from __future__ import annotations

import functools
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from wormocr.exceptions import DataError

PREFERRED = (
    "arial.ttf",
    "arialbd.ttf",
    "ariali.ttf",
    "arialbi.ttf",
    "calibri.ttf",
    "calibrib.ttf",
    "calibrii.ttf",
    "calibriz.ttf",
    "times.ttf",
    "timesbd.ttf",
    "timesi.ttf",
    "timesbi.ttf",
    "verdana.ttf",
    "verdanab.ttf",
    "verdanai.ttf",
    "georgia.ttf",
    "georgiab.ttf",
    "georgiai.ttf",
    "trebuc.ttf",
    "trebucbd.ttf",
    "segoeui.ttf",
    "segoeuib.ttf",
    "segoeuii.ttf",
    "consola.ttf",
    "consolab.ttf",
    "cour.ttf",
    "courbd.ttf",
    "cambria.ttf",
    "cambriab.ttf",
    "candara.ttf",
    "candarab.ttf",
    "corbel.ttf",
    "constan.ttf",
    "constanb.ttf",
    "tahoma.ttf",
    "tahomabd.ttf",
    "comic.ttf",
    "comicbd.ttf",
    "gara.ttf",
    "garabd.ttf",
    "pala.ttf",
    "palab.ttf",
    "lucon.ttf",
    "bahnschrift.ttf",
    "micross.ttf",
    "framd.ttf",
    "century.ttf",
    "bookos.ttf",
    "bookosb.ttf",
    "DejaVuSans.ttf",
    "DejaVuSans-Bold.ttf",
    "LiberationSans-Regular.ttf",
    "LiberationSans-Bold.ttf",
    "FreeSans.ttf",
)

SKIP_NAME_PARTS = (
    "wingding",
    "webding",
    "symbol",
    "marlett",
    "emoji",
    "seguihis",
    "segmdl2",
    "holomdl",
    "reference",
    "mtsort",
    "mtextra",
    "jsmath",
    "symbol",
)


def font_directories(extra: str | None = None) -> list[Path]:
    home = Path.home()
    candidates = [
        Path(extra) if extra else None,
        Path(r"C:\Windows\Fonts"),
        home / "AppData/Local/Microsoft/Windows/Fonts",
        Path("/usr/share/fonts"),
        Path("/usr/local/share/fonts"),
        home / ".fonts",
        home / ".local/share/fonts",
        Path("/Library/Fonts"),
        Path("/System/Library/Fonts"),
        home / "Library/Fonts",
    ]
    return [path for path in candidates if path is not None and path.exists()]


def _renders_latin(path: Path) -> bool:
    try:
        font = ImageFont.truetype(str(path), 28)
    except OSError:
        return False
    image = Image.new("L", (96, 48), 255)
    ImageDraw.Draw(image).text((4, 4), "Ag", font=font, fill=0)
    ink = int((np.asarray(image) < 250).sum())
    return ink > 15


def _iter_files(directory: Path) -> list[Path]:
    found: list[Path] = []
    if not directory.exists():
        return found
    for pattern in ("*.ttf", "*.otf", "*.ttc"):
        found.extend(directory.glob(pattern))
        found.extend(path for path in directory.rglob(pattern) if path.parent != directory)
    unique = {path.resolve(): path for path in found if path.is_file()}
    return sorted(unique.values(), key=lambda item: item.name.lower())


@functools.lru_cache(maxsize=8)
def discover_fonts(font_dir: str = "") -> tuple[str, ...]:
    directories = font_directories(font_dir or None)
    if not directories:
        raise DataError(
            "No font directory was found. Install a Latin font such as Arial or "
            "DejaVu Sans, or set data.font_dir to a folder of .ttf files."
        )
    chosen: list[Path] = []
    seen: set[Path] = set()
    for name in PREFERRED:
        for directory in directories:
            path = directory / name
            if not path.exists():
                continue
            resolved = path.resolve()
            if resolved in seen:
                continue
            if _renders_latin(path):
                chosen.append(path)
                seen.add(resolved)
            break
    if len(chosen) < 3:
        available = {
            path.resolve(): path
            for directory in directories
            for path in _iter_files(directory)
        }
        for path in available.values():
            if path.resolve() in seen:
                continue
            lowered = path.name.lower()
            if any(part in lowered for part in SKIP_NAME_PARTS):
                continue
            if _renders_latin(path):
                chosen.append(path)
                seen.add(path.resolve())
            if len(chosen) >= 12:
                break
    if not chosen:
        raise DataError(
            "No installed font could draw Latin letters. Install Arial, Calibri, "
            "or DejaVu Sans, or set data.font_dir."
        )
    return tuple(str(path) for path in chosen)
