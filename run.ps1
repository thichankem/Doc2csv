# Doc2CSV-AI launcher (PowerShell, portable) - tu tao .venv + cai deps lan dau.
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

# 1) Tim Python
$pyexe = $null; $pyargs = @()
if (Get-Command py -ErrorAction SilentlyContinue) { $pyexe = "py"; $pyargs = @("-3") }
elseif (Get-Command python -ErrorAction SilentlyContinue) { $pyexe = "python" }
elseif (Test-Path "$env:USERPROFILE\anaconda3\python.exe") { $pyexe = "$env:USERPROFILE\anaconda3\python.exe" }
else {
    Write-Host "[LOI] Khong tim thay Python. Cai Python 3.10+ tu https://www.python.org/downloads/" -ForegroundColor Red
    Read-Host "Nhan Enter de thoat"; exit 1
}

# 2) Tao venv lan dau
if (-not (Test-Path ".venv\Scripts\python.exe")) {
    Write-Host "[SETUP] Tao moi truong .venv (lan dau)..." -ForegroundColor Cyan
    & $pyexe @pyargs -m venv .venv
}

# 3) Cai deps lan dau
if (-not (Test-Path ".venv\.deps_ok")) {
    Write-Host "[SETUP] Cai thu vien (lan dau, can mang)..." -ForegroundColor Cyan
    & ".venv\Scripts\python.exe" -m pip install --upgrade pip
    & ".venv\Scripts\python.exe" -m pip install -r requirements.txt
    "ok" | Out-File -Encoding ascii ".venv\.deps_ok"
}

# 4) Chay app
& ".venv\Scripts\pythonw.exe" app.py
