# StoryBreak Pro — Railway 部署
# 內含 ffmpeg、Noto CJK 字體；持久化請掛載 Volume 到 /app/output

FROM python:3.10-slim

# 更新系統並安裝 FFmpeg 與 Noto 中日韓開源字體
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    fonts-noto-cjk \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 複製並安裝 Python 依賴套件
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 複製所有程式碼到映像
COPY . .

# 建立預設輸出資料夾（對應 BASE_DIR = Path("output")；exports 為 output/exports）
RUN mkdir -p /app/output /app/output/exports

# Railway 會動態分配 PORT，main.py 已用 os.getenv("PORT", "7860") 對接
ENV GRADIO_SERVER_NAME="0.0.0.0"

# 啟動應用程式
CMD ["python", "main.py"]
