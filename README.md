# Doc2CSV-AI

Phần mềm desktop trích xuất dữ liệu từ file **PDF / DOCX / DOC / TXT / ảnh** (hỗ trợ tài liệu lên tới ~1 triệu từ) thành **CSV dạng Alpaca** (`instruction / input / output`) để fine-tune mô hình AI. Sử dụng model **Ollama chạy local**, không gửi dữ liệu ra ngoài.

---

## Tính năng

- Đọc PDF, DOCX, DOC (Word cũ), TXT, MD
- **Đọc ảnh (OCR)**: PNG / JPG / WEBP / GIF / BMP / TIFF — dùng model **vision** của Ollama (llama3.2-vision, llava, minicpm-v, qwen2.5vl…) để chép chữ trong ảnh thành text, rồi xử lý như tài liệu thường
- **Thêm cả thư mục**: chọn 1 folder → tự duyệt đệ quy mọi file con được hỗ trợ (hoặc kéo-thả thư mục vào)
- Chia tài liệu lớn thành chunks theo ranh giới đoạn / câu (paragraph-aware)
- **Output luôn là JSON** (canonical: compact, sorted keys, UTF-8 preserved) — sẵn sàng cho fine-tune
- **Tốc độ tối ưu**:
  - Ollama `format=json` — grammar-constrained, không cần regex parse
  - `keep_alive=30m` — model giữ trong VRAM, chunk thứ 2 trở đi nhanh ~3-4×
  - `num_predict` cap — chặn output dài lê thê
  - HTTP session reuse
  - Preload model trước khi tính chunk → ETA chuẩn ngay từ đầu
- Streaming token + sub-status realtime
- **ETA prediction**: rolling 5-chunk average
- **System monitor**: CPU / RAM / GPU / VRAM realtime (NVIDIA NVML)
- DPI-aware Tkinter: sắc nét trên màn hình HiDPI
- Append-mode CSV: dừng giữa chừng không mất dữ liệu

---

## Yêu cầu

- **Python 3.10+** (Windows / macOS / Linux)
- **Ollama** đã cài đặt và đang chạy: <https://ollama.com>
- Ít nhất 1 model Ollama đã pull về (ví dụ: `qwen3.5:9b`, `llama3.1:8b`, `deepseek-r1:8b`...)
- **Nếu muốn đọc ảnh**: cần thêm 1 model *vision* — ví dụ `ollama pull llama3.2-vision` (hoặc `llava`, `minicpm-v`, `qwen2.5vl`)

---

## Cài đặt

```powershell
# 1. (Tuỳ chọn) Tạo virtualenv
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. Cài thư viện
pip install -r requirements.txt
```

Trên Windows, gói `pywin32` sẽ được cài tự động để hỗ trợ file `.doc` cũ qua Microsoft Word COM.

---

## Cách dùng

### 1. Mở GUI

**Cách đơn giản nhất** (Windows) — double-click hoặc gõ trong CMD:

```cmd
run.bat
```

Launcher này tự gọi Python ở `%USERPROFILE%\anaconda3\python.exe` (nơi đã cài đầy đủ deps).

Hoặc trong PowerShell:

```powershell
.\run.ps1
```

**Chạy trực tiếp** (nếu bạn biết Python nào có đủ deps):

```powershell
# Dùng Anaconda Python trực tiếp:
& "$env:USERPROFILE\anaconda3\python.exe" app.py

# Hoặc nếu `python` của bạn đã có pdfplumber/python-docx/requests:
python app.py
```

> ⚠ **Lưu ý**: Trên máy có nhiều Python (MSYS2, Windows Store, Anaconda...), `python app.py` có thể gọi nhầm Python không có deps. Cứ dùng `run.bat` là chắc chắn nhất.

### 2. Các bước trong GUI

1. **Thêm file** (.pdf / .docx / .doc / .txt / ảnh) hoặc **Thêm thư mục** (duyệt đệ quy cả folder). Cũng có thể kéo-thả file/thư mục vào cửa sổ.
2. **Nhập Instruction** — BẮT BUỘC. Mô tả task + JSON schema mong muốn. Ví dụ:
   ```
   Trích xuất Q&A dạng JSON: {"question": "...", "answer": "..."}
   ```
3. Chọn **Model Ollama** + đường dẫn **Output CSV**. Nếu có file ảnh, chọn thêm **Model đọc ảnh** (vision/OCR) — để trống nếu không xử lý ảnh.
4. Tham số (đã default tối ưu cho test nhanh):
   - **Chunk size**: số từ mỗi chunk (mặc định 200)
   - **num_predict**: chặn output tối đa N tokens (mặc định 256 — tăng nếu output bị cụt)
   - **Temp**: độ ngẫu nhiên (0.0 = deterministic, 0.3 = ổn định)
   - **num_ctx**: context window (4096 đủ cho chunk 200 từ)
   - **keep_alive**: thời gian giữ model trong VRAM (30m mặc định — tăng nếu test liên tục)
5. Bấm **▶ Bắt đầu trích xuất**

Mỗi chunk → 1 dòng CSV:

| instruction | input | output |
|---|---|---|
| (text bạn nhập) | (nội dung chunk) | `{"question":"...","answer":"..."}` |

### 3. Định dạng CSV output

| Cột | Mô tả |
|---|---|
| `instruction` | Câu lệnh/câu hỏi (bắt buộc) |
| `input` | Ngữ cảnh hỗ trợ (có thể rỗng) |
| `output` | Câu trả lời đúng (bắt buộc) |
| `source` | Tên file gốc |
| `chunk_id` | ID chunk (để truy vết) |

File CSV dùng UTF-8 BOM — mở Excel hiển thị tiếng Việt đúng ngay.

---

## Ước lượng thời gian (1 triệu từ)

- 1.000.000 từ ÷ 1500 từ/chunk ≈ **667 chunks**
- Tốc độ tuỳ phần cứng & model:
  - `llama3.1:8b` trên RTX 3060: ~5-8s/chunk → ~1-1.5 giờ
  - `deepseek-r1:14b` trên RTX 4090: ~6-10s/chunk → ~1.5-2 giờ
  - CPU-only: chậm hơn 5-10 lần
- Mỗi chunk sinh 3 mẫu → khoảng **~2000 mẫu training** cho 1 triệu từ

> Mẹo: chạy thử với 1 file nhỏ trước để kiểm tra chất lượng output. Nếu samples không đạt, điều chỉnh `samples_per_chunk` hoặc chuyển sang model mạnh hơn.

---

## Cấu trúc dự án

```
Doc2csv-ai/
├── app.py                  # GUI entry point
├── requirements.txt
├── README.md
└── src/
    ├── extractors/
    │   ├── pdf_extractor.py    # pdfplumber
    │   ├── docx_extractor.py   # python-docx + win32com (.doc)
    │   └── image_extractor.py  # OCR ảnh qua Ollama vision model
    ├── text_chunker.py         # Paragraph/sentence-aware chunking
    ├── ollama_client.py        # Ollama HTTP client
    ├── csv_writer.py           # Alpaca CSV writer (append-mode)
    └── pipeline.py             # Orchestrator
```

---

## FAQ

**Hỏi: Ollama không kết nối được?**
- Kiểm tra: `ollama serve` đang chạy, hoặc cài Ollama Desktop và mở nó lên
- URL mặc định: `http://localhost:11434`

**Hỏi: File .doc cũ không đọc được?**
- Cần Microsoft Word cài trên máy (cho `pywin32` COM tự động)
- Hoặc dùng Word/LibreOffice mở file rồi Save as `.docx`

**Hỏi: Đọc ảnh như thế nào?**
- Ảnh được OCR bằng **model vision của Ollama** (ảnh → text), sau đó chạy qua pipeline như tài liệu thường nên cột `input` vẫn là text thật.
- Cần chọn **Model đọc ảnh**: pull một model nhìn được ảnh, vd `ollama pull llama3.2-vision` (hoặc `llava`, `minicpm-v`, `qwen2.5vl`). App tự gợi ý nếu phát hiện model phù hợp.
- Có file ảnh nhưng chưa chọn model vision → app sẽ nhắc trước khi chạy.
- Chất lượng OCR phụ thuộc model vision; ảnh mờ/nghiêng có thể đọc thiếu. `Pillow` (cài kèm `requirements.txt`) giúp chuẩn hoá mọi định dạng ảnh.

**Hỏi: Dừng giữa chừng có mất dữ liệu không?**
- Không. Mỗi chunk được flush ngay vào CSV sau khi xử lý. Chạy lại sẽ append tiếp.

**Hỏi: Một số chunk báo "không parse được JSON"?**
- Thường gặp với model nhỏ (<7B). Hạ `temperature` xuống 0.1-0.2 hoặc đổi model mạnh hơn.

---

## License

MIT
