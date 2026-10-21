import numpy as np
from PIL import Image, ImageDraw, ImageFont

from wormocr.data.fonts import discover_fonts
from wormocr.detection.lines import Box, line_boxes, map_box_to_original, pil_rotate_matrix, reading_order, segment_page


def test_reading_order_columns_and_header():
    columns = [
        Box(10, 10, 100, 30),
        Box(10, 50, 100, 70),
        Box(300, 10, 420, 30),
        Box(300, 50, 420, 70),
    ]
    ordered = reading_order(columns)
    assert [(box.x0, box.y0) for box in ordered] == [(10, 10), (10, 50), (300, 10), (300, 50)]
    header = [
        Box(10, 10, 430, 40),
        Box(10, 70, 120, 100),
        Box(10, 120, 120, 150),
        Box(300, 70, 430, 100),
        Box(300, 120, 430, 150),
    ]
    ordered = reading_order(header)
    assert [(box.x0, box.y0) for box in ordered] == [
        (10, 10),
        (10, 70),
        (10, 120),
        (300, 70),
        (300, 120),
    ]


def test_blank_page_has_no_lines():
    assert line_boxes(np.full((80, 120), 255, dtype=np.uint8)) == []


def test_rendered_page_finds_lines_in_order():
    font = ImageFont.truetype(discover_fonts("")[0], 32)
    image = Image.new("L", (520, 240), 255)
    draw = ImageDraw.Draw(image)
    draw.text((24, 20), "Alpha line", font=font, fill=0)
    draw.text((24, 90), "Beta line", font=font, fill=0)
    draw.text((24, 160), "Gamma line", font=font, fill=0)
    layout = segment_page(image, correct_orientation=False)
    assert len(layout.boxes) == 3
    assert layout.boxes[0].y0 < layout.boxes[1].y0 < layout.boxes[2].y0
    assert layout.layout == "single-column"


def test_two_columns_read_down_each_column():
    font = ImageFont.truetype(discover_fonts("")[0], 28)
    image = Image.new("L", (640, 220), 255)
    draw = ImageDraw.Draw(image)
    draw.text((20, 30), "Left one", font=font, fill=0)
    draw.text((20, 110), "Left two", font=font, fill=0)
    draw.text((400, 30), "Right one", font=font, fill=0)
    draw.text((400, 110), "Right two", font=font, fill=0)
    layout = segment_page(image, correct_orientation=False)
    assert layout.layout == "multi-column"
    assert len(layout.boxes) == 4
    assert layout.boxes[0].x0 < 200
    assert layout.boxes[1].x0 < 200
    assert layout.boxes[2].x0 > 300


def test_deskew_and_box_mapping():
    image = Image.new("L", (220, 160), 255)
    ImageDraw.Draw(image).rectangle((40, 70, 170, 86), fill=0)
    skewed = image.rotate(3, resample=Image.Resampling.BILINEAR, expand=True, fillcolor=255)
    layout = segment_page(skewed, correct_orientation=True)
    assert abs(layout.angle - (-3)) <= 1.0
    matrix, size = pil_rotate_matrix(image.width, image.height, 3)
    rotated = image.rotate(3, resample=Image.Resampling.BILINEAR, expand=True, fillcolor=255)
    assert rotated.size == size
    ink = np.asarray(rotated)
    ys, xs = np.where(ink < 128)
    found = Box(int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
    mapped = map_box_to_original(found, matrix, image.size)
    assert mapped.x0 < 50 and mapped.x1 > 150
    assert mapped.y0 < 80 and mapped.y1 > 70
