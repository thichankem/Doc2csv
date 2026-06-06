"""Image text extraction (OCR) via an Ollama vision model.

An image is turned into plain text by asking a vision-capable model to
faithfully transcribe everything it sees. The resulting text then flows
through the normal pipeline (chunk → instruction → JSON → CSV), exactly like
a PDF/DOCX would — so the `input` column stays real text, ideal for training.
"""
import base64
import io
from pathlib import Path
from typing import Callable, Optional

from ..ollama_client import DEFAULT_BASE, generate

# Đuôi ảnh được hỗ trợ (giữ đồng bộ với app.py / pipeline.py).
IMAGE_EXTS = {
    ".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tif", ".tiff",
}

# Prompt OCR: chỉ chép chữ, không bình luận / dịch / tóm tắt.
_OCR_PROMPT = (
    "Bạn là công cụ OCR. Hãy ĐỌC và CHÉP LẠI toàn bộ nội dung chữ có trong ảnh, "
    "giữ nguyên thứ tự đọc, xuống dòng và cấu trúc (đoạn văn, danh sách, bảng). "
    "Với bảng: mỗi hàng một dòng, các ô ngăn cách bằng ' | '. "
    "TUYỆT ĐỐI không thêm bình luận, không dịch, không tóm tắt — chỉ chép đúng "
    "chữ nhìn thấy. Nếu ảnh không có chữ nào, trả về chuỗi rỗng."
)


def _encode_image(path: Path) -> str:
    """Đọc ảnh → base64. Nếu có Pillow thì chuẩn hoá sang PNG (hỗ trợ mọi
    định dạng lạ như bmp/tiff); không có thì gửi nguyên bytes (png/jpg/webp
    vốn đã chạy tốt với vision model)."""
    data = path.read_bytes()
    try:
        from PIL import Image  # optional dependency

        with Image.open(io.BytesIO(data)) as im:
            if im.mode not in ("RGB", "L"):
                im = im.convert("RGB")
            buf = io.BytesIO()
            im.save(buf, format="PNG")
            data = buf.getvalue()
    except Exception:
        pass  # fallback: raw bytes
    return base64.b64encode(data).decode("ascii")


def extract_image(
    path: str,
    model: str,
    base_url: str = DEFAULT_BASE,
    keep_alive: str = "30m",
    num_ctx: int = 4096,
    num_predict: int = 4096,
    on_token: Optional[Callable[[str, int], None]] = None,
    should_stop: Optional[Callable[[], bool]] = None,
) -> str:
    """OCR một ảnh bằng vision model của Ollama, trả về text thô.

    `model` phải là model nhìn được ảnh (vd: llama3.2-vision, llava,
    minicpm-v, qwen2.5vl). Nếu để trống sẽ báo lỗi rõ ràng.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File không tồn tại: {path}")
    if not model:
        raise RuntimeError(
            "Chưa chọn model đọc ảnh (vision/OCR). Hãy cài & chọn một model "
            "nhìn được ảnh, vd: llama3.2-vision, llava, minicpm-v, qwen2.5vl."
        )

    b64 = _encode_image(p)
    text = generate(
        model=model,
        prompt=_OCR_PROMPT,
        base_url=base_url,
        temperature=0.0,
        num_ctx=num_ctx,
        options={"num_predict": num_predict},
        images=[b64],
        keep_alive=keep_alive,
        on_token=on_token,
        should_stop=should_stop,
    )
    return text.strip()
