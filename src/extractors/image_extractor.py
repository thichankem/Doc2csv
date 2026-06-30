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


# OCR mặc định: hạ ảnh quá khổ về cạnh dài tối đa rồi nén JPEG. Vision model
# vốn tự chia ảnh thành ô (tile) ở độ phân giải cố định, nên ảnh 12MP gửi
# nguyên chỉ tốn băng thông + bộ nhớ chứ không sắc nét hơn. 2048px vẫn đủ rõ
# cho chữ tài liệu, mà payload nhỏ đi nhiều lần → upload + tiền xử lý nhanh hơn.
_DEFAULT_MAX_SIDE = 2048
_DEFAULT_JPEG_QUALITY = 90


def _encode_image(
    path: Path,
    max_side: int = _DEFAULT_MAX_SIDE,
    jpeg_quality: int = _DEFAULT_JPEG_QUALITY,
) -> str:
    """Đọc ảnh → base64. Có Pillow thì hạ kích thước ảnh quá khổ (chỉ thu nhỏ,
    không phóng to) và nén JPEG để giảm payload/token cho vision model; không có
    Pillow thì gửi nguyên bytes (png/jpg/webp vốn chạy tốt)."""
    data = path.read_bytes()
    try:
        from PIL import Image  # optional dependency

        with Image.open(io.BytesIO(data)) as im:
            # Giữ grayscale là 'L' (JPEG xám nhỏ hơn nhiều); còn lại ép RGB để
            # JPEG hoá được (bỏ alpha/palette).
            if im.mode not in ("RGB", "L"):
                im = im.convert("RGB")
            w, h = im.size
            longest = max(w, h)
            if max_side and longest > max_side:
                scale = max_side / longest
                im = im.resize(
                    (max(1, round(w * scale)), max(1, round(h * scale))),
                    Image.LANCZOS,
                )
            buf = io.BytesIO()
            im.save(buf, format="JPEG", quality=jpeg_quality, optimize=True)
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
    max_side: int = _DEFAULT_MAX_SIDE,
    jpeg_quality: int = _DEFAULT_JPEG_QUALITY,
    on_token: Optional[Callable[[str, int], None]] = None,
    should_stop: Optional[Callable[[], bool]] = None,
) -> str:
    """OCR một ảnh bằng vision model của Ollama, trả về text thô.

    `model` phải là model nhìn được ảnh (vd: llama3.2-vision, llava,
    minicpm-v, qwen2.5vl). Nếu để trống sẽ báo lỗi rõ ràng. Ảnh quá khổ được
    thu nhỏ về `max_side` px (cạnh dài) + nén JPEG để OCR nhanh hơn.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File không tồn tại: {path}")
    if not model:
        raise RuntimeError(
            "Chưa chọn model đọc ảnh (vision/OCR). Hãy cài & chọn một model "
            "nhìn được ảnh, vd: llama3.2-vision, llava, minicpm-v, qwen2.5vl."
        )

    b64 = _encode_image(p, max_side=max_side, jpeg_quality=jpeg_quality)
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
