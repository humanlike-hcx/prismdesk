@echo off
setlocal
chcp 65001 >nul
title PrismDesk CLI
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\start-cli.ps1"
if errorlevel 1 pause
