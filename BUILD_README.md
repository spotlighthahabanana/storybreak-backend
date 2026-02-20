# Build FXCROWD Studio EXE

## Quick Start

1. Install dependencies:
   ```
   pip install -r requirements.txt
   pip install pyinstaller
   ```

2. Run the build:
   ```
   build_exe.bat
   ```

3. Output: `dist\FXCROWD_Studio\`
   - Run `FXCROWD_Studio.exe`
   - Browser will open at http://127.0.0.1:7860

## Notes

- **First build** may take 10–20 minutes (torch, transformers, gradio are large).
- **Output size** is typically 2–4 GB due to PyTorch/transformers.
- **TransNet weights**: If you use TransNet V2, copy `transnetv2-pytorch-weights.pth` into `dist\FXCROWD_Studio\` so it can be found.
- **imageio-ffmpeg**: FFmpeg is bundled via imageio-ffmpeg for video export.
- **output/** folder: The app creates `output/` for thumbnails, clips, exports. Run the exe from its folder or ensure a writable `output/` path.

## Single-File EXE (Optional)

For a single .exe instead of a folder, edit `FXCROWD_Studio.spec` and replace the `EXE`+`COLLECT` section with:

```python
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='FXCROWD_Studio',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
```

Then run: `pyinstaller --noconfirm FXCROWD_Studio.spec`

Note: Single-file exe will be larger and slower to start.
