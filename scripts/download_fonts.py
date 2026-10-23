"""Download open-licensed handwriting fonts from github.com/google/fonts.

Only families under ofl/ (SIL Open Font License) or apache/ (Apache 2.0) are
used. Fonts go to artifacts/fonts/handwriting/ (gitignored) together with
each family's license file and a FONTS.json inventory.

    python scripts/download_fonts.py      # set GITHUB_TOKEN to avoid API rate limits
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from pathlib import Path

DEST = Path(__file__).resolve().parents[1] / "artifacts" / "fonts" / "handwriting"

FAMILIES = [
    # neat print-style handwriting
    "patrickhand", "kalam", "handlee", "neucha", "architectsdaughter", "indieflower", "gochihand",
    "schoolbell", "comingsoon", "shortstack", "sriracha", "itim", "mali", "gaegu", "delius",
    "shantellsans", "deliciushandrawn", "fuzzybubbles", "grapenuts", "mynerve", "gamjaflower",
    "himelody", "annieuseyourtelescope", "justanotherhand", "sunshiney", "craftygirls", "walterturncoat",
    "permanentmarker", "rocksalt", "caveat", "caveatbrush", "shadowsintolight", "shadowsintolighttwo",
    "gloriahallelujah", "coveredbyyourgrace", "reeniebeanie", "nothingyoucoulddo", "waitingforthesunrise",
    "sueellenfrancisco", "overtherainbow", "justmeagaindownhere", "kristi", "zeyada", "dawningofanewday",
    "labelleaurore", "homemadeapple", "cedarvillecursive", "nanumpenscript", "nanumbrushscript",
    "calligraffitti", "loversquarrel", "swankyandmoomoo", "badscript", "marckscript", "oooohbaby",
    "squarepeg", "bethellen", "mrdafoe", "comforter", "cookie", "satisfy", "yellowtail", "damion",
    "dancingscript", "pacifico", "sacramento", "allura", "greatvibes", "parisienne", "alexbrush",
    "herrvonmuellerhoff", "mrssaintdelafield", "meddon", "lacquer", "rubikdoodleshadow",
    # school handwriting models (print and cursive)
    "edusabeginner", "edunswactfoundation", "eduqldbeginner", "eduvicwantbeginner", "edutasbeginner",
    "playwriteus", "playwriteustrad", "playwritegbs", "playwritegbj", "playwriteau", "playwritenz",
    "playwriteie", "playwriteca", "playwritein", "playwriteza", "playwritefr", "playwritede",
]

API = "https://api.github.com/repos/google/fonts/contents"


def _get(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "wormocr-font-fetch"})
    token = os.environ.get("GITHUB_TOKEN")
    if token and "api.github.com" in url:
        request.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    inventory: list[dict] = []
    missing: list[str] = []
    for family in FAMILIES:
        listing = None
        for license_dir in ("ofl", "apache"):
            try:
                listing = json.loads(_get(f"{API}/{license_dir}/{family}"))
                break
            except urllib.error.HTTPError as exc:
                if exc.code != 404:
                    raise
        if listing is None:
            missing.append(family)
            continue
        folder = DEST / family
        folder.mkdir(exist_ok=True)
        files = []
        for entry in listing:
            name = entry["name"]
            if not (name.endswith(".ttf") or name in {"OFL.txt", "LICENSE.txt"}):
                continue
            target = folder / name
            if not target.exists():
                target.write_bytes(_get(entry["download_url"]))
            files.append(name)
        license_name = "SIL OFL 1.1" if license_dir == "ofl" else "Apache 2.0"
        inventory.append({"family": family, "license": license_name, "files": files})
        print(f"{family}: {license_name}, {sum(f.endswith('.ttf') for f in files)} font file(s)", flush=True)
    (DEST / "FONTS.json").write_text(
        json.dumps({"source": "https://github.com/google/fonts", "families": inventory, "missing": missing}, indent=2),
        encoding="utf-8",
    )
    print(f"{len(inventory)} families downloaded; not found: {', '.join(missing) or 'none'}")


if __name__ == "__main__":
    main()
