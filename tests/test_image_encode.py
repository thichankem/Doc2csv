"""OCR pre-processing: oversized images are downscaled + JPEG-compressed."""
import base64
import io

import pytest

from src.extractors.image_extractor import (
    _DEFAULT_MAX_SIDE,
    _encode_image,
)

Image = pytest.importorskip("PIL.Image", reason="Pillow not installed")


def _make_png(tmp_path, w, h, color=(123, 200, 50)):
    p = tmp_path / f"img_{w}x{h}.png"
    Image.new("RGB", (w, h), color).save(p, format="PNG")
    return p


def test_oversized_image_is_downscaled(tmp_path):
    src = _make_png(tmp_path, 4000, 3000)
    b64 = _encode_image(src)
    out = Image.open(io.BytesIO(base64.b64decode(b64)))
    assert max(out.size) <= _DEFAULT_MAX_SIDE          # capped on the long side
    assert out.size[0] / out.size[1] == pytest.approx(4000 / 3000, rel=0.02)
    # JPEG-compressed payload is far smaller than the raw PNG bytes.
    assert len(base64.b64decode(b64)) < src.stat().st_size


def test_small_image_not_upscaled(tmp_path):
    src = _make_png(tmp_path, 800, 600)
    b64 = _encode_image(src)
    out = Image.open(io.BytesIO(base64.b64decode(b64)))
    assert out.size == (800, 600)                      # untouched dimensions
    assert out.format == "JPEG"


def test_custom_max_side(tmp_path):
    src = _make_png(tmp_path, 4000, 2000)
    b64 = _encode_image(src, max_side=1000)
    out = Image.open(io.BytesIO(base64.b64decode(b64)))
    assert max(out.size) == 1000


def test_grayscale_stays_grayscale(tmp_path):
    p = tmp_path / "gray.png"
    Image.new("L", (3000, 1000), 128).save(p, format="PNG")
    b64 = _encode_image(p)
    out = Image.open(io.BytesIO(base64.b64decode(b64)))
    assert out.mode == "L"
    assert max(out.size) <= _DEFAULT_MAX_SIDE
