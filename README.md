# 🎬 StoryBreak — AI Video Reference Breakdown Tool

Automatically detect scene cuts, extract key frames, and generate scene annotations with AI.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## ✨ Features

- 🎥 **Smart scene detection** — PySceneDetect and TransNet V2 for automatic cut detection
- 📸 **Auto thumbnails** — Extract a representative frame per scene
- 🤖 **AI annotation** — GPT-4 Vision (optional) for scene descriptions
- 🎨 **Annotation styles** — Detailed, cinematic, concise, script-style
- 💾 **Multi-format export** — JSON, CSV, Markdown, and branded PDF shot lists
- 🖥️ **Modern UI** — Gradio-based workstation (Projects, Assembly, AI Classify, New Task)

## 🚀 Quick start

### 1. Install dependencies

```bash
# Optional: use a virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

pip install -r requirements.txt
```

### 2. Run the app

```bash
python app.py
```

The browser will open automatically, or visit http://127.0.0.1:7860

## 📖 Usage

### Scene detection (New Task)

1. Upload a video (MP4, AVI, MKV, MOV, etc.).
2. Choose algorithm: **TransNet V2 (AI)**, Content, Adaptive, or Threshold.
3. Adjust **Sensitivity** and **Min frames**.
4. Click **Start Analysis**, then **Open in Workstation** to edit scenes.

### AI Classify

Run AI analysis to tag shot types (e.g. Wide, Close-up) and camera movement. Filter and edit tags in the gallery.

### Assembly

Drag scenes from **Source Scenes** into the right **Timeline** to build a sequence. Export to PDF (shot list) or video.

### Export

- **PDF** — Professional shot list with optional project name and director’s notes (Settings: user logo path).
- **JSON / CSV / Markdown** — For data and scripts.

## 📁 Project structure

```
movie_v4/
├── app.py              # Main app (Gradio UI)
├── scene_detector.py   # Scene detection
├── ai_annotator.py     # AI annotation
├── requirements.txt   # Dependencies
├── README.md           # This file
├── assets/             # Optional: splash.png, logo_text.png
└── output/             # Created at runtime (thumbnails, clips, exports)
```

## ⚙️ Configuration

### Environment variables

```bash
set OPENAI_API_KEY=sk-your-api-key-here  # Windows
# export OPENAI_API_KEY=sk-your-api-key-here  # Linux/Mac
```

### Settings (in app)

- Default export path  
- UI scale  
- Default tags (comma-separated) for Scene Tag dropdown  
- PDF: optional path to your/company logo (shown on every PDF page)

## 🔧 FAQ

**Q: Too many or too few scenes?**  
Adjust **Sensitivity**. Higher = fewer cuts; lower = more.

**Q: Can I use it without an OpenAI API key?**  
Yes. Scene detection and thumbnails work offline. Only AI annotation and some AI Classify features need a key.

**Q: Supported video formats?**  
Any format OpenCV can read: MP4, AVI, MKV, MOV, WMV, FLV, etc.

**Q: Slow processing?**  
Use an SSD, try a lower resolution, or reduce **Min frames**.

## 📝 Changelog

### v2.0.0 (2026-02-17)
- Rebranded as **StoryBreak** — *The Ultimate Video Reference Breakdown Tool*
- UI and workflow improvements; PDF shot list with project name and director’s notes

### v1.0.0 (2026-02-06)
- Initial release
- Multiple detection methods and export formats

## 📄 License

MIT License — Use, modify, and distribute freely.

---

Made with ❤️ for filmmakers and video creators
