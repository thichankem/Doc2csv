# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller build spec for Doc2CSV-AI.

Builds a single windowed .exe (no console) with the app icon. Run with:

    pyinstaller --noconfirm Doc2CSV-AI.spec

or just double-click `build_exe.bat`. Output: dist/Doc2CSV-AI.exe
"""
from PyInstaller.utils.hooks import collect_all

# Packages that ship data/binaries PyInstaller can't see by static analysis:
#   tkinterdnd2 -> the native tkdnd library (drag & drop)
#   pdfplumber / pdfminer -> CMap & font data files for PDF text extraction
datas, binaries, hiddenimports = [], [], []
for _pkg in ("tkinterdnd2", "pdfplumber", "pdfminer"):
    _d, _b, _h = collect_all(_pkg)
    datas += _d
    binaries += _b
    hiddenimports += _h

# Imported lazily (inside try/except), so declare them explicitly:
#   pynvml        -> GPU/VRAM monitor (nvidia-ml-py)
#   win32com/...  -> reading legacy .doc via Word COM (pywin32)
hiddenimports += [
    "pynvml",
    "win32com",
    "win32com.client",
    "win32timezone",
    "pythoncom",
    "psutil",
    "docx",
    "PIL",
]


a = Analysis(
    ["app.py"],
    pathex=["."],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=[
        "pytest", "_pytest",
        # Keep the exe lean / avoid Qt-binding clashes if the build env happens
        # to be a fat Anaconda install. None of these are used by the app.
        "matplotlib", "PyQt5", "PyQt6", "PySide2", "PySide6",
        "IPython", "jupyter", "notebook", "nbformat", "nbconvert",
        "scipy", "pandas", "numpy", "sphinx", "black", "tornado", "zmq",
    ],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="Doc2CSV-AI",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,          # GUI app — no console window
    disable_windowed_traceback=False,
    icon="assets/icon.ico",
)
