import io
import json
import threading
import urllib.error
import urllib.request

import pytest
import torch
from PIL import Image, ImageDraw

from wormocr.config import load_config
from wormocr.inference.engine import LineEngine
from wormocr.models.crnn import build_model
from wormocr.vocabulary import Vocabulary
from wormocr.web import describe_engine, make_server


@pytest.fixture(scope="module")
def server():
    config = load_config("configs/debug.yaml")
    vocabulary = Vocabulary()
    model = build_model(config, vocabulary.size)
    engine = LineEngine(model, vocabulary, torch.device("cpu"), config, "untrained")
    httpd = make_server(engine, describe_engine(engine, "wormOCR test", "cpu"), port=0)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()
    httpd.server_close()


def _page(fmt: str = "PNG") -> bytes:
    image = Image.new("RGB", (600, 260), "white")
    draw = ImageDraw.Draw(image)
    for y in (40, 110, 180):
        draw.rectangle((40, y, 520, y + 26), fill="black")
    buffer = io.BytesIO()
    image.save(buffer, format=fmt)
    return buffer.getvalue()


def _post(url: str, data: bytes):
    return urllib.request.urlopen(urllib.request.Request(url, data=data, method="POST"))


def test_index_assets_and_health(server):
    page = urllib.request.urlopen(server + "/").read().decode("utf-8")
    assert "wormOCR" in page and "/static/app.js" in page
    assert urllib.request.urlopen(server + "/static/style.css").status == 200
    health = json.loads(urllib.request.urlopen(server + "/api/health").read())
    assert health["model"]["name"] == "wormOCR test"
    assert health["model"]["parameters"] > 0


def test_page_upload_returns_lines_boxes_and_preview(server):
    payload = json.loads(_post(server + "/api/ocr?mode=page&filename=a.png", _page()).read())
    assert payload["kind"] == "image"
    page = payload["pages"][0]
    assert page["preview"].startswith("data:image/jpeg;base64,")
    assert len(page["lines"]) == 3
    assert payload["line_count"] == 3
    assert all(line["frames"] > 0 and line["bbox"]["width"] > 0 for line in page["lines"])
    assert set(payload["timings"]) == {"decode", "layout", "recognize"}


def test_stream_emits_pipeline_events_in_order(server):
    body = _post(server + "/api/ocr/stream?mode=page&filename=a.png", _page()).read().decode("utf-8")
    events = [json.loads(line) for line in body.splitlines() if line.strip()]
    types = [event["type"] for event in events]
    assert types[0] == "start" and types[-1] == "done"
    assert types.index("page") < types.index("layout") < types.index("line")
    assert types.count("line") == 3
    stages = {event["stage"] for event in events if event["type"] == "log"}
    assert {"upload", "model", "decode", "layout", "recognize", "done"} <= stages
    assert all("t" in event for event in events)


def test_pdf_upload_streams_each_page(server):
    pytest.importorskip("pypdfium2")
    first = Image.open(io.BytesIO(_page()))
    buffer = io.BytesIO()
    first.save(buffer, format="PDF", save_all=True, append_images=[first.copy()])
    body = _post(server + "/api/ocr/stream?mode=page&filename=doc.pdf", buffer.getvalue()).read().decode("utf-8")
    events = [json.loads(line) for line in body.splitlines() if line.strip()]
    assert [event["pages"] for event in events if event["type"] == "document"] == [2]
    assert [event["page"] for event in events if event["type"] == "page"] == [0, 1]
    assert events[-1]["type"] == "done"


def test_bad_uploads_and_path_traversal_are_rejected(server):
    with pytest.raises(urllib.error.HTTPError) as error:
        _post(server + "/api/ocr?filename=a.png", b"not an image")
    assert error.value.code == 422
    body = _post(server + "/api/ocr/stream?filename=a.png", b"not an image").read().decode("utf-8")
    assert json.loads(body.strip().splitlines()[-1])["type"] == "error"
    with pytest.raises(urllib.error.HTTPError) as error:
        _post(server + "/api/ocr?mode=sideways&filename=a.png", _page())
    assert error.value.code == 400
    with pytest.raises(urllib.error.HTTPError) as error:
        urllib.request.urlopen(server + "/static/../server.py")
    assert error.value.code == 404
