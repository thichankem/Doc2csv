# Tao shortcut Doc2CSV-AI tren Desktop, tro toi run.bat, dung assets\icon.ico.
$root = $PSScriptRoot
$target = Join-Path $root 'run.bat'
$icon = Join-Path $root 'assets\icon.ico'

if (-not (Test-Path $target)) {
    Write-Host "[LOI] Khong thay run.bat tai $target" -ForegroundColor Red
    exit 1
}

$desktop = [Environment]::GetFolderPath('Desktop')
$lnkPath = Join-Path $desktop 'Doc2CSV-AI.lnk'

$ws = New-Object -ComObject WScript.Shell
$sc = $ws.CreateShortcut($lnkPath)
$sc.TargetPath = $target
$sc.WorkingDirectory = $root
$sc.Description = 'Doc2CSV-AI - trich xuat du lieu training tu tai lieu'
$sc.WindowStyle = 7   # chay thu nho (console an gon)
if (Test-Path $icon) { $sc.IconLocation = $icon }
$sc.Save()

Write-Host "Da tao icon tren Desktop: $lnkPath" -ForegroundColor Green
Write-Host "Bam dup vao icon de chay (lan dau se tu cai thu vien)."
