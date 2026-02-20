@echo off
chcp 65001 >nul
title Build FXCROWD Studio EXE
echo.
echo ========================================
echo   Building FXCROWD Studio (EXE)
echo ========================================
echo.

cd /d "%~dp0"

rem Check for venv
set "PYTHON_CMD=python"
if exist "venv\Scripts\python.exe" (
    set "PYTHON_CMD=venv\Scripts\python.exe"
    echo Using venv Python
) else (
    echo Using system Python
)
echo.

rem Install PyInstaller if needed
%PYTHON_CMD% -c "import PyInstaller" 2>nul
if errorlevel 1 (
    echo Installing PyInstaller...
    %PYTHON_CMD% -m pip install pyinstaller
)
echo.

rem Clean previous build
if exist "dist\FXCROWD_Studio" rd /s /q "dist\FXCROWD_Studio"
if exist "build" rd /s /q "build"
echo.

rem Build
echo Building (this may take several minutes)...
%PYTHON_CMD% -m PyInstaller --noconfirm FXCROWD_Studio.spec

if exist "dist\FXCROWD_Studio\FXCROWD_Studio.exe" (
    echo.
    echo ========================================
    echo   BUILD SUCCESS
    echo ========================================
    echo Output: dist\FXCROWD_Studio\
    echo Run: dist\FXCROWD_Studio\FXCROWD_Studio.exe
    echo.
    echo IMPORTANT: Copy these into dist\FXCROWD_Studio\ if needed:
    echo   - scene_detector.py, video_llava.py, ai_annotator.py (or ensure bundled)
    echo   - transnetv2-pytorch-weights.pth (if using TransNet)
    echo   - output\ folder for thumbnails/clips/exports
) else (
    echo BUILD FAILED
)
echo.
pause
