@echo off
REM ============================================================
REM  Doc2CSV-AI launcher (portable)
REM  - Tu tim Python, tu tao .venv + cai deps lan dau,
REM    sau do chay app. Hoat dong tren may moi sau khi clone.
REM ============================================================
setlocal
cd /d "%~dp0"

REM --- 1) Tim Python ---
set "PYEXE="
where py >nul 2>nul && set "PYEXE=py -3"
if not defined PYEXE (
    where python >nul 2>nul && set "PYEXE=python"
)
if not defined PYEXE if exist "%USERPROFILE%\anaconda3\python.exe" set "PYEXE=%USERPROFILE%\anaconda3\python.exe"
if not defined PYEXE (
    echo [LOI] Khong tim thay Python. Cai Python 3.10+ tu https://www.python.org/downloads/
    echo       (nho tick "Add Python to PATH") roi chay lai.
    pause
    exit /b 1
)

REM --- 2) Tao .venv lan dau ---
if not exist ".venv\Scripts\python.exe" (
    echo [SETUP] Lan dau chay - dang tao moi truong .venv ...
    %PYEXE% -m venv .venv
    if errorlevel 1 ( echo [LOI] Khong tao duoc venv & pause & exit /b 1 )
)

REM --- 3) Cai thu vien lan dau (danh dau bang .deps_ok de bo qua lan sau) ---
if not exist ".venv\.deps_ok" (
    echo [SETUP] Dang cai thu vien (chi lan dau, can mang)...
    ".venv\Scripts\python.exe" -m pip install --upgrade pip
    ".venv\Scripts\python.exe" -m pip install -r requirements.txt
    if errorlevel 1 ( echo [LOI] Cai thu vien that bai & pause & exit /b 1 )
    echo ok> ".venv\.deps_ok"
)

REM --- 4) Chay app (pythonw = khong hien cua so console) ---
start "" ".venv\Scripts\pythonw.exe" app.py
endlocal
