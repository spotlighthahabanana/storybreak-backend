@echo off
chcp 65001 >nul
title AI Scene Detection Tool
echo.
echo ========================================
echo     AI Scene Detection Tool starting...
echo ========================================
echo.

rem Change to batch file directory
cd /d "%~dp0"

rem Suppress FFmpeg H.264 decode warnings
set AV_LOG_LEVEL=-8

echo Starting server...
echo Open the URL shown below in your browser
echo.
echo Press Ctrl+C to stop the server
echo ========================================
echo.

rem Prefer venv in this folder; otherwise use system Python
set "PYTHON_CMD="
if exist "venv\Scripts\python.exe" (
    set "PYTHON_CMD=venv\Scripts\python.exe"
) else (
    echo venv\Scripts\python.exe not found, using system Python...
    set "PYTHON_CMD=python"
)

%PYTHON_CMD% app.py

echo.
echo Server stopped. Press any key to close this window...
pause
