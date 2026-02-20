# StoryBreak Pro — Railway 部署
# 內含 ffmpeg、Noto CJK 字體；持久化請掛載 Volume 到 /app/output

FROM python:3.10-slim

USER root

# 更新系統並安裝 FFmpeg、OpenCV 依賴與 Noto 中日韓字體
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsm6 \
    libxext6 \
    fonts-noto-cjk \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 複製並安裝 Python 依賴套件
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 複製所有程式碼到映像
COPY . .

# 建立輸出資料夾並開放權限（避免雲端 FFmpeg 寫入暫存檔失敗）
RUN mkdir -p /app/output /app/output/exports && chmod -R 777 /app/output

# Railway 會動態分配 PORT，main.py 已用 os.getenv("PORT", "7860") 對接
ENV GRADIO_SERVER_NAME="0.0.0.0"

# 啟動應用程式（FastAPI + Gradio 掛載於 main.py）
CMD ["python", "main.py"]
