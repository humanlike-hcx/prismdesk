@echo off
setlocal
start "PrismDesk" /min powershell.exe -NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass -File "%~dp0scripts\open-demo.ps1"
exit /b 0
