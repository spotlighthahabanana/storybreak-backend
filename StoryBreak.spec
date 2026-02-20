# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec for StoryBreak - The Ultimate Video Reference Breakdown Tool

import os

block_cipher = None

# Optional branding (place in assets/)
_assets = 'assets'
_splash = os.path.join(_assets, 'splash.png') if os.path.exists(os.path.join(_assets, 'splash.png')) else None
datas = [(_assets, _assets)] if os.path.exists(_assets) else []

a = Analysis(
    ['app.py'],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=[
        'scene_detector',
        'video_llava',
        'ai_annotator',
        'gradio',
        'cv2',
        'scenedetect',
        'PIL',
        'PIL.Image',
        'fpdf',
        'fpdf2',
        'openai',
        'numpy',
        'torch',
        'transformers',
        'imageio_ffmpeg',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'matplotlib',
        'tkinter',
        'PyQt5',
        'PyQt6',
        'PySide2',
        'PySide6',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='StoryBreak',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,
    splash=_splash,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='StoryBreak',
)
