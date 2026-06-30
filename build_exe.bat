@echo off
REM ============================================================
REM  Doc2CSV-AI - build a standalone Windows .exe (PyInstaller)
REM  Output: dist\Doc2CSV-AI.exe  (double-click to run, no Python needed)
REM ============================================================
setlocal
cd /d "%~dp0"

REM --- Find a Python interpreter (prefer the project .venv, then py/python) ---
set "PY="
if exist ".venv\Scripts\python.exe" set "PY=.venv\Scripts\python.exe"
if not defined PY (
    where py >nul 2>nul && set "PY=py"
)
if not defined PY (
    where python >nul 2>nul && set "PY=python"
)
if not defined PY (
    echo [X] Khong tim thay Python. Cai Python 3.10+ roi chay lai.
    pause
    exit /b 1
)

echo [*] Dung Python: %PY%
echo [*] Cai/cap nhat thu vien (PyInstaller + deps)...
%PY% -m pip install -r requirements.txt >nul
%PY% -m pip install pyinstaller >nul

echo [*] Dang build... (lan dau co the mat vai phut)
%PY% -m PyInstaller --noconfirm --clean Doc2CSV-AI.spec
if errorlevel 1 (
    echo [X] Build that bai. Xem log o tren.
    pause
    exit /b 1
)

echo.
echo [OK] Xong! File chay o:  dist\Doc2CSV-AI.exe
echo      Copy providers.json (neu dung Online/Mix) vao cung thu muc voi .exe.
echo.
pause
