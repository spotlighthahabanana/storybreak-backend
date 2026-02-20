@echo off
chcp 65001 >nul
title Build StoryBreak EXE
echo.
echo ========================================
echo   Building StoryBreak (EXE)
echo ========================================
echo.

cd /d "%~dp0"

set "PYTHON_CMD=python"
if exist "venv\Scripts\python.exe" (
    set "PYTHON_CMD=venv\Scripts\python.exe"
    echo Using venv Python
) else (
    echo Using system Python
)
echo.

%PYTHON_CMD% -c "import PyInstaller" 2>nul
if errorlevel 1 (
    echo Installing PyInstaller...
    %PYTHON_CMD% -m pip install pyinstaller
)
echo.

if exist "dist\StoryBreak" rd /s /q "dist\StoryBreak"
if exist "build" rd /s /q "build"
echo.

echo Building (this may take several minutes)...
echo Optional: add assets\icon.ico and assets\splash.png for branded exe.
echo.
%PYTHON_CMD% -m PyInstaller --noconfirm StoryBreak.spec

if exist "dist\StoryBreak\StoryBreak.exe" (
    echo.
    echo ========================================
    echo   BUILD SUCCESS
    echo ========================================
    echo Output: dist\StoryBreak\
    echo Run: dist\StoryBreak\StoryBreak.exe
    echo.
    echo Copy into dist\StoryBreak\ if needed:
    echo   - scene_detector.py, video_llava.py, ai_annotator.py
    echo   - transnetv2-pytorch-weights.pth
    echo   - output\ for thumbnails/clips/exports
) else (
    echo BUILD FAILED
)
echo.
pause
