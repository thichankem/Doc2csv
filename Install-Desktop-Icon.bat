@echo off
REM Tao shortcut "Doc2CSV-AI" ra man hinh Desktop (icon dep, tro toi run.bat).
REM Chay 1 lan sau khi clone tu GitHub. Sau do bam icon la chay duoc.
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0Install-Desktop-Icon.ps1"
pause
