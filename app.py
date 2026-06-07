"""Doc2CSV-AI - Tkinter desktop GUI (DPI-aware, with system monitor)."""
import os
import sys
import threading
import tkinter as tk
import tkinter.font as tkfont
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Optional

from src.csv_flatten import flatten_csv
from src.llm_backends import LLMRouter, OllamaBackend
from src.ollama_client import is_running, list_models
from src.pipeline import Pipeline
from src.providers import DEFAULT_PATH as PROVIDERS_PATH
from src.providers import load_backends
from src.sysmon import SystemMonitor

# Optional drag-and-drop support via tkinterdnd2. If missing, app still
# works — user must use the "Thêm file..." button as before.
try:
    from tkinterdnd2 import DND_FILES, TkinterDnD  # type: ignore
    _DND_AVAILABLE = True
except ImportError:
    _DND_AVAILABLE = False

_IMG_GLOB = "*.png *.jpg *.jpeg *.webp *.gif *.bmp *.tif *.tiff"
SUPPORTED_FILETYPES = [
    ("Tất cả hỗ trợ", f"*.pdf *.docx *.doc *.txt *.md {_IMG_GLOB}"),
    ("Documents", "*.pdf *.docx *.doc *.txt *.md"),
    ("Ảnh", _IMG_GLOB),
    ("PDF", "*.pdf"),
    ("Word", "*.docx *.doc"),
    ("Text", "*.txt *.md"),
    ("All files", "*.*"),
]

DOC_EXTS = {".pdf", ".docx", ".doc", ".txt", ".md"}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tif", ".tiff"}
SUPPORTED_EXTS = DOC_EXTS | IMAGE_EXTS

PREFERRED_MODELS = (
    "llama3.1:8b", "gemma4:e4b", "qwen3.5:9b",
    "deepseek-r1:8b", "deepseek-r1:14b",
)

# Heuristic nhận diện model nhìn được ảnh (dựa trên tên) để tự gợi ý.
VISION_HINTS = (
    "llava", "vision", "minicpm-v", "moondream", "bakllava", "pixtral",
    "qwen2-vl", "qwen2.5vl", "qwen2.5-vl", "qwen3-vl", "qwen3vl",
    "gemma3", "llama3.2-vision", "llama4", "granite3.2-vision",
)

INSTRUCTION_PLACEHOLDER = (
    'Ví dụ: Tóm tắt đoạn văn dưới dạng JSON {"summary": "...", "keywords": [...]}\n'
    'Hoặc: Trích xuất Q&A: {"question": "...", "answer": "..."}\n'
    "(Output LUÔN là JSON — Ollama format=json bắt buộc)"
)


# ---------------------------------------------------------------------------
# DPI awareness — must run before Tk() to get sharp rendering on HiDPI screens.
# ---------------------------------------------------------------------------
def enable_dpi_awareness() -> None:
    if sys.platform != "win32":
        return
    import ctypes
    for fn in (
        lambda: ctypes.windll.shcore.SetProcessDpiAwareness(2),   # Per-Monitor v2
        lambda: ctypes.windll.shcore.SetProcessDpiAwareness(1),   # Per-Monitor v1
        lambda: ctypes.windll.user32.SetProcessDPIAware(),         # System
    ):
        try:
            fn()
            return
        except (AttributeError, OSError):
            continue


def format_duration(seconds: Optional[float]) -> str:
    if seconds is None or seconds <= 0:
        return "--"
    s = int(seconds)
    if s < 60:
        return f"{s}s"
    if s < 3600:
        m, sec = divmod(s, 60)
        return f"{m}m {sec:02d}s"
    h, rem = divmod(s, 3600)
    m = rem // 60
    return f"{h}h {m:02d}m"


class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("Doc2CSV-AI · Trích xuất dữ liệu training")
        root.geometry("980x860")
        root.minsize(880, 740)

        self.files: list[str] = []
        self.worker: threading.Thread | None = None
        self.stop_flag = False
        self.online_backends: list = []   # rotation pool loaded from providers.json

        self.sysmon = SystemMonitor()
        self._sysmon_job: Optional[str] = None
        self._placeholder_on = True
        self._pending_logs: list[str] = []
        self._log_flush_job: Optional[str] = None

        self._build_ui()
        self.refresh_models()
        self.load_providers(silent=True)
        self._schedule_sysmon()

        root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ------------------------------------------------------------------ UI
    def _build_ui(self) -> None:
        pad = {"padx": 10, "pady": 6}
        main = ttk.Frame(self.root, padding=12)
        main.pack(fill="both", expand=True)

        # ===== Section 1: Files =====
        dnd_hint = "  ·  kéo-thả file vào đây" if _DND_AVAILABLE else ""
        fbox = ttk.LabelFrame(
            main,
            text=f"  1. Files đầu vào (.pdf / .docx / .doc / .txt / ảnh) — có thể thêm cả thư mục{dnd_hint}  ",
        )
        fbox.pack(fill="x", pady=(0, 10))

        toolbar = ttk.Frame(fbox)
        toolbar.pack(fill="x", **pad)
        ttk.Button(toolbar, text="Thêm file...", command=self.add_files).pack(side="left", padx=(0, 6))
        ttk.Button(toolbar, text="Thêm thư mục...", command=self.add_folder).pack(side="left", padx=(0, 6))
        ttk.Button(toolbar, text="Xóa đã chọn", command=self.remove_selected).pack(side="left", padx=(0, 6))
        ttk.Button(toolbar, text="Xóa hết", command=self.clear_files).pack(side="left")
        self.lbl_count = ttk.Label(toolbar, text="0 file")
        self.lbl_count.pack(side="right")

        list_frame = ttk.Frame(fbox)
        list_frame.pack(fill="both", expand=False, **pad)
        self.lst_files = tk.Listbox(
            list_frame, height=5, selectmode="extended",
            activestyle="dotbox",
            font=("Segoe UI", 10),
            highlightthickness=0, borderwidth=1, relief="solid",
        )
        sb = ttk.Scrollbar(list_frame, orient="vertical", command=self.lst_files.yview)
        self.lst_files.config(yscrollcommand=sb.set)
        self.lst_files.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")

        # Đăng ký drop target cho listbox và toàn bộ cửa sổ
        # (thả ở đâu cũng nhận, miễn là tkinterdnd2 sẵn sàng).
        if _DND_AVAILABLE:
            for w in (self.lst_files, self.root):
                try:
                    w.drop_target_register(DND_FILES)
                    w.dnd_bind("<<Drop>>", self._on_drop_files)
                except (tk.TclError, AttributeError):
                    pass

        # ===== Section 2: Instruction =====
        ibox = ttk.LabelFrame(main, text="  2. Instruction (cùng dùng cho mỗi chunk → cột 'instruction')  ")
        ibox.pack(fill="x", pady=(0, 10))

        instr_frame = ttk.Frame(ibox)
        instr_frame.pack(fill="x", **pad)
        self.txt_instr = tk.Text(
            instr_frame, height=4, wrap="word",
            font=("Segoe UI", 10),
            highlightthickness=0, borderwidth=1, relief="solid",
            padx=8, pady=6,
        )
        sb_i = ttk.Scrollbar(instr_frame, orient="vertical", command=self.txt_instr.yview)
        self.txt_instr.config(yscrollcommand=sb_i.set)
        self.txt_instr.pack(side="left", fill="both", expand=True)
        sb_i.pack(side="right", fill="y")

        self.txt_instr.insert("1.0", INSTRUCTION_PLACEHOLDER)
        self.txt_instr.config(foreground="gray")
        self.txt_instr.bind("<FocusIn>", self._instr_focus_in)
        self.txt_instr.bind("<FocusOut>", self._instr_focus_out)

        # Schema mẫu — ép Ollama dùng đúng keys cho MỌI chunk (Structured Outputs).
        schema_frame = ttk.Frame(ibox)
        schema_frame.pack(fill="x", **pad)
        ttk.Label(
            schema_frame,
            text='Schema mẫu (khuyên dùng): JSON ví dụ — vd  {"bệnh":[], "triệu chứng":[]}',
            foreground="#444",
        ).pack(anchor="w")
        self.var_schema = tk.StringVar(value='')
        ttk.Entry(
            schema_frame, textvariable=self.var_schema,
            font=("Consolas", 10),
        ).pack(fill="x", pady=(4, 0))

        # ===== Section 3: Config =====
        cbox = ttk.LabelFrame(main, text="  3. Cấu hình  ")
        cbox.pack(fill="x", pady=(0, 10))

        r1 = ttk.Frame(cbox); r1.pack(fill="x", **pad)
        ttk.Label(r1, text="Model Ollama:", width=14).pack(side="left")
        self.cmb_model = ttk.Combobox(r1, state="readonly", width=28)
        self.cmb_model.pack(side="left", padx=8)
        ttk.Button(r1, text="↻ Refresh", command=self.refresh_models).pack(side="left")
        self.lbl_ollama = ttk.Label(r1, text="", foreground="gray")
        self.lbl_ollama.pack(side="left", padx=12)

        rv = ttk.Frame(cbox); rv.pack(fill="x", **pad)
        ttk.Label(rv, text="Model đọc ảnh:", width=14).pack(side="left")
        self.cmb_vision = ttk.Combobox(rv, state="readonly", width=28)
        self.cmb_vision.pack(side="left", padx=8)
        ttk.Label(
            rv, text="(vision/OCR — để trống nếu không xử lý ảnh)",
            foreground="gray",
        ).pack(side="left", padx=4)

        r2 = ttk.Frame(cbox); r2.pack(fill="x", **pad)
        ttk.Label(r2, text="Output CSV:").pack(side="left")
        self.var_out = tk.StringVar(value=str(Path.cwd() / "output" / "training_data.csv"))
        ttk.Entry(r2, textvariable=self.var_out, font=("Segoe UI", 10)).pack(
            side="left", fill="x", expand=True, padx=8
        )
        ttk.Button(r2, text="...", command=self.choose_output, width=4).pack(side="left")

        r3 = ttk.Frame(cbox); r3.pack(fill="x", **pad)
        ttk.Label(r3, text="Chunk size (từ):").pack(side="left")
        self.var_chunk = tk.IntVar(value=200)
        ttk.Spinbox(r3, from_=50, to=8000, increment=50,
                    textvariable=self.var_chunk, width=8).pack(side="left", padx=6)

        ttk.Label(r3, text="num_predict:").pack(side="left", padx=(18, 0))
        self.var_npredict = tk.IntVar(value=512)
        ttk.Spinbox(r3, from_=32, to=4096, increment=32,
                    textvariable=self.var_npredict, width=8).pack(side="left", padx=6)

        ttk.Label(r3, text="Temp:").pack(side="left", padx=(18, 0))
        self.var_temp = tk.DoubleVar(value=0.1)
        ttk.Spinbox(r3, from_=0.0, to=1.5, increment=0.05,
                    textvariable=self.var_temp, width=6, format="%.2f").pack(side="left", padx=6)

        ttk.Label(r3, text="num_ctx:").pack(side="left", padx=(18, 0))
        self.var_ctx = tk.IntVar(value=4096)
        ttk.Spinbox(r3, from_=2048, to=32768, increment=1024,
                    textvariable=self.var_ctx, width=8).pack(side="left", padx=6)

        ttk.Label(r3, text="keep_alive:").pack(side="left", padx=(18, 0))
        self.var_keep = tk.StringVar(value="30m")
        ttk.Entry(r3, textvariable=self.var_keep, width=6).pack(side="left", padx=6)

        r4 = ttk.Frame(cbox); r4.pack(fill="x", **pad)
        ttk.Label(r4, text="Retry JSON:").pack(side="left")
        self.var_retries = tk.IntVar(value=1)
        ttk.Spinbox(r4, from_=0, to=5, increment=1,
                    textvariable=self.var_retries, width=6).pack(side="left", padx=6)
        ttk.Label(
            r4, text="(thử lại khi output chưa đạt JSON/schema)",
            foreground="gray",
        ).pack(side="left", padx=(2, 0))

        self.var_resume = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            r4, text="Resume (ghi tiếp file output, bỏ qua chunk đã có)",
            variable=self.var_resume,
        ).pack(side="left", padx=(24, 0))

        # Chế độ chạy: offline (Ollama) / online (API xoay vòng) / mix (cả hai)
        r5 = ttk.Frame(cbox); r5.pack(fill="x", **pad)
        ttk.Label(r5, text="Chế độ:", width=14).pack(side="left")
        self.var_mode = tk.StringVar(value="offline")
        for val, label in (
            ("offline", "Offline (Ollama)"),
            ("online", "Online (API xoay vòng)"),
            ("mix", "Mix (Ollama + API)"),
        ):
            ttk.Radiobutton(
                r5, text=label, value=val, variable=self.var_mode,
                command=self._on_mode_change,
            ).pack(side="left", padx=(0, 10))
        ttk.Button(r5, text="↻ Tải providers", command=self.load_providers).pack(side="left", padx=(8, 6))
        self.lbl_providers = ttk.Label(r5, text="", foreground="gray")
        self.lbl_providers.pack(side="left")

        # ===== Action buttons =====
        abox = ttk.Frame(main)
        abox.pack(fill="x", pady=(0, 10))
        self.btn_start = ttk.Button(abox, text="▶  Bắt đầu trích xuất", command=self.start, style="Accent.TButton")
        self.btn_start.pack(side="left", padx=(0, 8), ipadx=6, ipady=2)
        self.btn_stop = ttk.Button(abox, text="⏹  Dừng", command=self.stop, state="disabled")
        self.btn_stop.pack(side="left", ipadx=4, ipady=2)
        ttk.Button(abox, text="📂  Mở thư mục output", command=self.open_output_dir).pack(
            side="right", ipadx=4, ipady=2
        )
        ttk.Button(abox, text="📁  Mở thư mục flat", command=self.open_flat_dir).pack(
            side="right", padx=(0, 8), ipadx=4, ipady=2
        )

        # ===== Bottom split: Progress (left) + System monitor (right) =====
        bottom = ttk.Frame(main)
        bottom.pack(fill="x", pady=(0, 10))
        bottom.columnconfigure(0, weight=3)
        bottom.columnconfigure(1, weight=2)

        # Progress
        pbox = ttk.LabelFrame(bottom, text="  4. Tiến trình  ")
        pbox.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        self.pb = ttk.Progressbar(pbox, mode="determinate")
        self.pb.pack(fill="x", padx=10, pady=(10, 6))
        self.lbl_status = ttk.Label(pbox, text="Sẵn sàng.", font=("Segoe UI", 10, "bold"))
        self.lbl_status.pack(anchor="w", padx=10, pady=2)
        self.lbl_eta = ttk.Label(pbox, text="ETA: --", foreground="#0066cc")
        self.lbl_eta.pack(anchor="w", padx=10, pady=2)
        self.lbl_substatus = ttk.Label(pbox, text="", foreground="gray", font=("Consolas", 9))
        self.lbl_substatus.pack(anchor="w", padx=10, pady=(2, 10))

        # System monitor
        sbox = ttk.LabelFrame(bottom, text="  5. Tài nguyên hệ thống  ")
        sbox.grid(row=0, column=1, sticky="nsew", padx=(6, 0))
        self._build_sysmon(sbox)

        # ===== Section 6: Log =====
        lbox = ttk.LabelFrame(main, text="  6. Log  ")
        lbox.pack(fill="both", expand=True)
        log_frame = ttk.Frame(lbox)
        log_frame.pack(fill="both", expand=True, padx=10, pady=8)
        self.txt_log = tk.Text(
            log_frame, height=10, wrap="word",
            font=("Consolas", 9),
            highlightthickness=0, borderwidth=1, relief="solid",
            padx=8, pady=6,
        )
        sb2 = ttk.Scrollbar(log_frame, orient="vertical", command=self.txt_log.yview)
        self.txt_log.config(yscrollcommand=sb2.set, state="disabled")
        self.txt_log.pack(side="left", fill="both", expand=True)
        sb2.pack(side="right", fill="y")

    def _build_sysmon(self, parent: ttk.LabelFrame) -> None:
        wrap = ttk.Frame(parent, padding=(10, 10))
        wrap.pack(fill="both", expand=True)
        wrap.columnconfigure(1, weight=1)

        def row(r: int, label: str) -> tuple[ttk.Progressbar, ttk.Label]:
            ttk.Label(wrap, text=label, width=6).grid(row=r, column=0, sticky="w", pady=3)
            pb = ttk.Progressbar(wrap, mode="determinate", maximum=100)
            pb.grid(row=r, column=1, sticky="ew", padx=(8, 8))
            val = ttk.Label(wrap, text="--", width=16, anchor="e", font=("Consolas", 9))
            val.grid(row=r, column=2, sticky="e")
            return pb, val

        self.pb_cpu, self.lbl_cpu = row(0, "CPU")
        self.pb_ram, self.lbl_ram = row(1, "RAM")
        self.pb_gpu, self.lbl_gpu = row(2, "GPU")
        self.pb_vram, self.lbl_vram = row(3, "VRAM")

        gpu_name = self.sysmon.gpu_name or "Không phát hiện GPU NVIDIA"
        self.lbl_gpu_name = ttk.Label(wrap, text=gpu_name, foreground="gray", font=("Segoe UI", 9))
        self.lbl_gpu_name.grid(row=4, column=0, columnspan=3, sticky="w", pady=(8, 0))

    # ---------------------------------------------------------------- sysmon
    def _schedule_sysmon(self) -> None:
        try:
            s = self.sysmon.sample()
            self.pb_cpu["value"] = s.cpu_pct
            self.lbl_cpu.config(text=f"{s.cpu_pct:5.1f} %")
            self.pb_ram["value"] = s.ram_pct
            self.lbl_ram.config(text=f"{s.ram_used_gb:4.1f}/{s.ram_total_gb:.1f} GB")
            if s.gpu_pct is not None:
                self.pb_gpu["value"] = s.gpu_pct
                self.lbl_gpu.config(text=f"{s.gpu_pct:5.1f} %")
                self.pb_vram["value"] = s.vram_pct or 0.0
                self.lbl_vram.config(text=f"{s.vram_used_gb:4.2f}/{s.vram_total_gb:.2f} GB")
            else:
                self.lbl_gpu.config(text="n/a")
                self.lbl_vram.config(text="n/a")
        except Exception:
            pass
        self._sysmon_job = self.root.after(2000, self._schedule_sysmon)

    # ------------------------------------------------------------ placeholder
    def _instr_focus_in(self, _evt) -> None:
        if self._placeholder_on:
            self.txt_instr.delete("1.0", "end")
            self.txt_instr.config(foreground="black")
            self._placeholder_on = False

    def _instr_focus_out(self, _evt) -> None:
        if not self.txt_instr.get("1.0", "end-1c").strip():
            self.txt_instr.delete("1.0", "end")
            self.txt_instr.insert("1.0", INSTRUCTION_PLACEHOLDER)
            self.txt_instr.config(foreground="gray")
            self._placeholder_on = True

    def _get_instruction(self) -> str:
        if self._placeholder_on:
            return ""
        return self.txt_instr.get("1.0", "end-1c").strip()

    # ------------------------------------------------------------ file ops
    def _add_one(self, path_str: str) -> tuple[int, int]:
        """Thêm một file (đã chắc là file). Trả về (added, skipped)."""
        path = Path(path_str)
        if path.suffix.lower() not in SUPPORTED_EXTS:
            return 0, 1
        s = str(path)
        if s in self.files:
            return 0, 0
        self.files.append(s)
        self.lst_files.insert("end", s)
        return 1, 0

    @staticmethod
    def _iter_dir_files(root: str) -> list[str]:
        """Duyệt đệ quy thư mục, trả về mọi file có đuôi hỗ trợ (đã sort)."""
        out: list[str] = []
        for dirpath, _dirs, names in os.walk(root):
            for n in names:
                p = Path(dirpath) / n
                if p.suffix.lower() in SUPPORTED_EXTS:
                    out.append(str(p))
        return sorted(out)

    def _add_paths(self, paths) -> tuple[int, int]:
        """Lọc & thêm đường dẫn hợp lệ (file hoặc thư mục) vào self.files +
        listbox. Thư mục được duyệt đệ quy. Trả về (thêm mới, bỏ qua)."""
        added = skipped = 0
        for raw in paths:
            p = str(raw).strip().strip("{}").strip('"').strip("'")
            if not p:
                continue
            path = Path(p)
            if path.is_dir():
                for fp in self._iter_dir_files(p):
                    a, s = self._add_one(fp)
                    added += a
                    skipped += s
            elif path.is_file():
                a, s = self._add_one(p)
                added += a
                skipped += s
            else:
                skipped += 1
        self.lbl_count.config(text=f"{len(self.files)} file")
        return added, skipped

    def add_files(self) -> None:
        paths = filedialog.askopenfilenames(title="Chọn file", filetypes=SUPPORTED_FILETYPES)
        if paths:
            self._add_paths(paths)

    def add_folder(self) -> None:
        d = filedialog.askdirectory(
            title="Chọn thư mục (đọc tất cả file con, kể cả thư mục con)"
        )
        if not d:
            return
        added, skipped = self._add_paths([d])
        msg = f"📁 Thêm thư mục: +{added} file"
        if skipped:
            msg += f" ({skipped} bỏ qua — không hỗ trợ)"
        self.log(msg)

    def _on_drop_files(self, event) -> None:
        """Handler cho sự kiện <<Drop>> của tkinterdnd2.
        event.data là chuỗi Tcl list, có thể chứa các path trong dấu {} khi có khoảng trắng."""
        try:
            raw_list = self.root.tk.splitlist(event.data)
        except (tk.TclError, AttributeError):
            raw_list = [event.data]
        added, skipped = self._add_paths(raw_list)
        if added or skipped:
            msg = f"📥 Đã kéo-thả: +{added} file"
            if skipped:
                msg += f" ({skipped} bỏ qua — không hỗ trợ)"
            self.log(msg)

    def remove_selected(self) -> None:
        for idx in reversed(self.lst_files.curselection()):
            self.lst_files.delete(idx)
            del self.files[idx]
        self.lbl_count.config(text=f"{len(self.files)} file")

    def clear_files(self) -> None:
        self.files.clear()
        self.lst_files.delete(0, "end")
        self.lbl_count.config(text="0 file")

    def choose_output(self) -> None:
        p = filedialog.asksaveasfilename(
            defaultextension=".csv",
            initialfile="training_data.csv",
            filetypes=[("CSV", "*.csv")],
        )
        if p:
            self.var_out.set(p)

    def open_output_dir(self) -> None:
        out = Path(self.var_out.get()).parent
        out.mkdir(parents=True, exist_ok=True)
        try:
            os.startfile(str(out))
        except AttributeError:
            messagebox.showinfo("Output dir", str(out))

    def open_flat_dir(self) -> None:
        """Mở thư mục `<output>_flat` — nơi nút 'JSON → CSV phẳng' ghi kết quả."""
        out_parent = Path(self.var_out.get()).parent
        flat = out_parent.parent / f"{out_parent.name}_flat"
        flat.mkdir(parents=True, exist_ok=True)
        try:
            os.startfile(str(flat))
        except AttributeError:
            messagebox.showinfo("Flat dir", str(flat))

    def _auto_flatten(self, src_csv: str) -> None:
        """Sau khi pipeline xong, tự động phẳng hoá file vừa ghi ra
        <output_dir>_flat/<stem>_flat.csv (bao gồm explode array)."""
        src = Path(src_csv)
        if not src.exists():
            return
        dst_dir = src.parent.parent / f"{src.parent.name}_flat"
        dst = dst_dir / f"{src.stem}_flat.csv"
        self.log("")
        self.log(f"📊 Tự động phẳng hoá → {dst}")
        try:
            flatten_csv(str(src), str(dst), on_log=self.log)
        except Exception as e:
            self.log(f"   ❌ Lỗi phẳng hoá: {e}")

    @staticmethod
    def _timestamped_path(path: str) -> str:
        """Insert _YYYYMMDD_HHMMSS before extension so each run gets a unique file."""
        p = Path(path)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        suffix = p.suffix or ".csv"
        return str(p.with_name(f"{p.stem}_{ts}{suffix}"))

    # ------------------------------------------------------------- Ollama
    def refresh_models(self) -> None:
        if not is_running():
            self.lbl_ollama.config(text="⚠ Ollama không chạy (localhost:11434)", foreground="red")
            self.cmb_model["values"] = []
            self.cmb_model.set("")
            self.cmb_vision["values"] = []
            self.cmb_vision.set("")
            return
        models = list_models()
        self.cmb_model["values"] = models
        # Vision: cho phép để trống + tự gợi ý model có vẻ nhìn được ảnh.
        self.cmb_vision["values"] = [""] + models
        vchosen = ""
        for m in models:
            if any(h in m.lower() for h in VISION_HINTS):
                vchosen = m
                break
        self.cmb_vision.set(vchosen)
        if not models:
            self.cmb_model.set("")
            self.lbl_ollama.config(text="⚠ Chưa có model nào", foreground="orange")
            return
        chosen = ""
        for pref in PREFERRED_MODELS:
            if pref in models:
                chosen = pref
                break
        self.cmb_model.set(chosen or models[0])
        self.lbl_ollama.config(text=f"✓ {len(models)} model có sẵn", foreground="#0a8a3a")

    # ---------------------------------------------------- providers / mode
    def load_providers(self, silent: bool = False) -> None:
        """(Re)load the online rotation pool from providers.json."""
        try:
            self.online_backends = load_backends(
                PROVIDERS_PATH, on_log=(None if silent else self.log)
            )
        except Exception as e:
            self.online_backends = []
            if not silent:
                self.log(f"⚠ Lỗi nạp providers: {e}")
        self._update_providers_label()

    def _update_providers_label(self) -> None:
        n = len(self.online_backends)
        if n:
            self.lbl_providers.config(
                text=f"✓ {n} model online", foreground="#0a8a3a"
            )
        else:
            self.lbl_providers.config(
                text=f"chưa có (tạo {PROVIDERS_PATH})", foreground="orange"
            )

    def _on_mode_change(self) -> None:
        mode = self.var_mode.get()
        if mode in ("online", "mix") and not self.online_backends:
            self.log(
                f"ℹ Chế độ '{mode}' cần API online — copy providers.example.json "
                f"thành {PROVIDERS_PATH}, điền key rồi bấm '↻ Tải providers'."
            )

    def _build_router(self):
        """Build the LLMRouter for the current mode, or None for pure offline.

        Returns (router_or_None, error_message_or_None)."""
        mode = self.var_mode.get()
        if mode == "offline":
            return None, None

        if mode in ("online", "mix") and not self.online_backends:
            return None, (
                f"Chế độ '{mode}' cần ít nhất một model online.\n"
                f"Copy providers.example.json → {PROVIDERS_PATH}, điền API key "
                "(free), rồi bấm '↻ Tải providers'."
            )

        backends = list(self.online_backends)
        if mode == "mix":
            model = self.cmb_model.get().strip()
            if not model:
                return None, "Chế độ Mix cần chọn thêm một Model Ollama."
            backends = [OllamaBackend(model)] + backends

        return LLMRouter(backends, on_log=self.log), None

    # ---------------------------------------------------- logging & progress
    def _flush_logs(self) -> None:
        self._log_flush_job = None
        if not self._pending_logs:
            return

        self.txt_log.config(state="normal")
        for msg in self._pending_logs:
            self.txt_log.insert("end", msg + "\n")
        self.txt_log.see("end")
        self.txt_log.config(state="disabled")
        self._pending_logs.clear()

        try:
            line_count = int(self.txt_log.index("end-1c").split(".")[0])
        except Exception:
            return
        if line_count > 400:
            self.txt_log.delete("1.0", f"{line_count - 300}.0")

    def log(self, msg: str) -> None:
        self._pending_logs.append(msg)
        if self._log_flush_job is None:
            self._log_flush_job = self.root.after(120, self._flush_logs)

    def update_progress(self, cur: int, total: int, eta: Optional[float] = None) -> None:
        def _apply():
            self.pb["maximum"] = max(total, 1)
            self.pb["value"] = cur
            pct = (cur / total * 100) if total else 0.0
            self.lbl_status.config(text=f"Tổng: {cur}/{total} chunks  ({pct:.1f}%)")
            if eta is None or eta <= 0:
                self.lbl_eta.config(text="ETA: --")
            else:
                self.lbl_eta.config(text=f"ETA: {format_duration(eta)}")
        self.root.after(0, _apply)

    def update_status(self, msg: str) -> None:
        self.root.after(0, lambda: self.lbl_substatus.config(text=msg))

    # -------------------------------------------------------------- run/stop
    def start(self) -> None:
        if not self.files:
            messagebox.showwarning("Thiếu file", "Vui lòng thêm ít nhất một file.")
            return

        mode = self.var_mode.get()
        router, router_err = self._build_router()
        if router_err:
            messagebox.showwarning("Chế độ chạy", router_err)
            return

        model = self.cmb_model.get().strip()
        # Offline mode chạy trực tiếp self.model nên bắt buộc chọn. Online dùng
        # router (model bỏ trống được). Mix đã kiểm tra model trong _build_router.
        if mode == "offline" and not model:
            messagebox.showwarning("Thiếu model", "Vui lòng chọn model Ollama.")
            return

        vision_model = self.cmb_vision.get().strip()
        has_image = any(Path(f).suffix.lower() in IMAGE_EXTS for f in self.files)
        if has_image and not vision_model:
            messagebox.showwarning(
                "Thiếu model đọc ảnh",
                "Danh sách có file ảnh nhưng bạn chưa chọn 'Model đọc ảnh' "
                "(vision/OCR).\nHãy chọn một model nhìn được ảnh — vd: "
                "llama3.2-vision, llava, minicpm-v, qwen2.5vl — rồi thử lại.",
            )
            return

        out_template = self.var_out.get().strip()
        if not out_template:
            messagebox.showwarning("Thiếu output", "Vui lòng chọn đường dẫn output CSV.")
            return

        instruction = self._get_instruction()
        if not instruction:
            messagebox.showwarning(
                "Thiếu instruction",
                "Vui lòng nhập instruction. Output sẽ luôn là JSON nên cần mô tả task cụ thể.",
            )
            return

        # Resume → ghi tiếp vào đúng file người dùng chọn (không timestamp) để
        # tìm lại được chunk đã làm. Ngược lại mỗi run tạo file mới với timestamp
        # - tránh lock Excel/OneDrive.
        resume = bool(self.var_resume.get())
        out = out_template if resume else self._timestamped_path(out_template)
        if router is not None:
            self.log(f"\n=== Bắt đầu run | Chế độ: {mode} | {len(router)} backend xoay vòng ===")
        else:
            self.log(f"\n=== Bắt đầu run | Model: {model} ===")
        if has_image:
            self.log(f"🖼 Model đọc ảnh (OCR): {vision_model}")
        self.log(f"📁 Output file: {out}")

        self.btn_start.config(state="disabled")
        self.btn_stop.config(state="normal")
        self.stop_flag = False
        self.pb["value"] = 0
        self.lbl_eta.config(text="ETA: tính toán...")

        pipe = Pipeline(
            files=list(self.files),
            model=model,
            output_csv=out,
            instruction=instruction,
            chunk_words=int(self.var_chunk.get()),
            temperature=float(self.var_temp.get()),
            num_ctx=int(self.var_ctx.get()),
            num_predict=int(self.var_npredict.get()),
            keep_alive=self.var_keep.get().strip() or "30m",
            vision_model=vision_model,
            output_template=self.var_schema.get().strip(),
            max_json_retries=int(self.var_retries.get()),
            resume=resume,
            router=router,
            on_log=self.log,
            on_progress=self.update_progress,
            on_status=self.update_status,
            should_stop=lambda: self.stop_flag,
        )

        def runner():
            try:
                pipe.run()
                # Auto-flatten ngay sau khi pipeline ghi xong CSV gốc
                if not self.stop_flag:
                    self._auto_flatten(out)
            except Exception as e:
                self.log(f"❌ Lỗi không mong đợi: {e}")
            finally:
                self.root.after(0, self._on_done)

        self.worker = threading.Thread(target=runner, daemon=True)
        self.worker.start()

    def stop(self) -> None:
        self.stop_flag = True
        self.log("⏸ Đang dừng sau token / chunk hiện tại...")
        self.btn_stop.config(state="disabled")

    def _on_done(self) -> None:
        self.btn_start.config(state="normal")
        self.btn_stop.config(state="disabled")
        if self.stop_flag:
            self.lbl_status.config(text="Đã dừng.")
        else:
            self.lbl_status.config(text="Hoàn tất.")
        self.lbl_eta.config(text="ETA: --")

    def _on_close(self) -> None:
        self.stop_flag = True
        if self._sysmon_job is not None:
            try:
                self.root.after_cancel(self._sysmon_job)
            except Exception:
                pass
        try:
            self.sysmon.shutdown()
        except Exception:
            pass
        self.root.destroy()


def _setup_fonts() -> None:
    # Set the global Tk named fonts so all widgets render with Segoe UI / Consolas.
    for name in ("TkDefaultFont", "TkTextFont", "TkMenuFont",
                 "TkHeadingFont", "TkCaptionFont", "TkSmallCaptionFont",
                 "TkIconFont", "TkTooltipFont"):
        try:
            tkfont.nametofont(name).configure(family="Segoe UI", size=10)
        except tk.TclError:
            pass
    try:
        tkfont.nametofont("TkFixedFont").configure(family="Consolas", size=10)
    except tk.TclError:
        pass


def _setup_style(root: tk.Tk) -> None:
    style = ttk.Style()
    for theme in ("vista", "winnative", "clam"):
        if theme in style.theme_names():
            style.theme_use(theme)
            break

    style.configure("TLabelframe.Label", font=("Segoe UI", 10, "bold"))
    style.configure("TButton", padding=(10, 5))
    style.configure("Accent.TButton", padding=(14, 6), font=("Segoe UI", 10, "bold"))
    style.configure("TProgressbar", thickness=18)


def main() -> None:
    enable_dpi_awareness()
    # Dùng TkinterDnD.Tk() để nhận drop từ Explorer; fallback nếu lib chưa cài.
    root = TkinterDnD.Tk() if _DND_AVAILABLE else tk.Tk()
    _setup_fonts()
    _setup_style(root)
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
