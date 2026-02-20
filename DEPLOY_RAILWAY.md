# 部署 StoryBreak Pro 到 Railway

## 架構總覽：兩個專案、兩邊部署

| 要上線的內容 | 技術 | 建議 GitHub Repo | 部署平台 | 網址 |
|-------------|------|------------------|----------|------|
| **前端殼**（首頁 + iframe 嵌工具） | Vite + React | `storybreak-frontend` 或 `framelogic-app` | **Vercel** | `framelogic.app` |
| **後端工具**（Gradio + 影片處理） | Python + FastAPI | `storybreak-backend` 或 `movie_v4` | **Railway** | 例如 `xxx.up.railway.app` |

- **framelogic.app**：給使用者看的網域，跑的是 **React 專案**（Vercel）。React 裡用 iframe 載入 Railway 的網址。
- **Railway 網址**：跑的是 **這個 Python 專案**（main.py + app.py + Dockerfile）。要先把這個資料夾 push 到某個 GitHub repo，再讓 Railway 從該 repo 部署。

**你現在要選清楚：**

1. **要上線的是 React 前端** → 把 **Vite 專案** push 到 `storybreak-backend`（或新建 `storybreak-frontend`），Vercel 連這個 repo，Domains 綁 `framelogic.app`。
2. **要上線的是 Python 後端** → 把 **這個 movie_v4 資料夾**（含 main.py、Dockerfile、requirements.txt）push 到一個 GitHub repo（可叫 `storybreak-backend`），Railway 連這個 repo 部署，拿到 `xxx.up.railway.app`，再在 React 的 iframe `src` 填這個網址。

兩邊都做：React repo 給 Vercel，Python repo 給 Railway，framelogic.app 的 iframe 指到 Railway 網址。

---

## 專案結構（部署前請確認）

請確保所有檔案在同一層級，再推送到 GitHub：

```text
/你的專案資料夾 (例如 movie_v4)
 ├── main.py              # FastAPI + Gradio 入口（已讀取 PORT）
 ├── app.py               # UI 與業務邏輯
 ├── scene_detector.py    # 場景偵測
 ├── ai_annotator.py      # AI 註解
 ├── video_llava.py       # LLaVA 影片註解（可選）
 ├── requirements.txt
 ├── Dockerfile
 └── assets/              # 可選，例如 logo.png
```

## 一、前置

- 已安裝 [Railway CLI](https://docs.railway.app/develop/cli)（可選，用網頁也可）
- 專案根目錄有 `Dockerfile`、`requirements.txt`、`main.py`

## 二、用 Railway 網頁部署（推薦）

1. **登入** [railway.app](https://railway.app) → **New Project** → **Deploy from GitHub repo**（或 **Empty** 再手動連 GitHub）。
2. **選 Repo**：選你的 `movie_v4`（或含此專案的 repo）。
3. **Root Directory**：若專案不在 repo 根目錄，設成該資料夾（例如 `movie_v4`）。
4. **Build**：Railway 會偵測到 `Dockerfile`，用 Docker 建置；若沒有 Dockerfile 會用 Nixpacks。
5. **設定環境變數**：在專案 → **Variables** 新增：

   | 變數名 | 說明 | 必填 |
   |--------|------|------|
   | `PORT` | Railway 會自動注入，不需手動設 | 自動 |
   | `SESSION_SECRET_KEY` | Session 加密用，自訂一串亂碼 | 建議 |
   | `GOOGLE_CLIENT_ID` | Google OAuth Client ID | 若用 Google 登入 |
   | `GOOGLE_CLIENT_SECRET` | Google OAuth Client Secret | 若用 Google 登入 |
   | `STORYBREAK_LICENSE_KEYS` | License 白名單，多組用 `;` 分隔 | 選填 |
   | `STORYBREAK_USER` / `STORYBREAK_PASS` | Admin 後門帳密 | 選填 |

6. **Google OAuth**：在 Google Cloud Console 的 OAuth 憑證裡，把 **授權重新導向 URI** 加上：
   - `https://你的服務名稱.up.railway.app/auth`
   - 部署後從 Railway 的 **Settings → Domains** 複製網址。
7. **Volume（持久化）**：**強烈建議**設定，否則每次重啟/重新部署後 SQLite 與匯出檔會清空。
   - **Settings** → **Volumes** → **Add Volume**
   - **Mount Path** 填：`/app/output`
   - 對應程式中的 `BASE_DIR = Path("output")`，資料庫 `storybreak.db` 與 PDF/影片匯出都在此目錄下。
8. **Deploy**：推送 commit 或手動 **Deploy**，等 build 完成後用 **Generate Domain** 取得網址。

## 三、用 Railway CLI 部署

```bash
# 安裝 CLI 後登入
railway login

# 在專案目錄
cd movie_v4
railway init          # 選 Create new project 或 Link to existing
railway add -d docker # 若要用 Docker 建置（有 Dockerfile 通常會自動）
railway variables set SESSION_SECRET_KEY=你的亂碼
railway variables set GOOGLE_CLIENT_ID=xxx
railway variables set GOOGLE_CLIENT_SECRET=xxx
railway up             # 建置並部署
railway domain         # 產生對外網址
```

## 四、程式已配合的設定

- **Port**：`main.py` 會讀取環境變數 `PORT`（Railway 自動注入），本機預設 `7860`。
- **Host**：`0.0.0.0`，可對外連線。
- **Reload**：若存在 `RAILWAY_ENVIRONMENT=production` 會關閉 reload，避免雲端重載。

## 五、常見問題

- **Build 失敗**：確認 `requirements.txt` 與 `Dockerfile` 在 repo 內，且 Dockerfile 無語法錯誤。
- **502 / 無法連線**：確認服務有 listen `0.0.0.0` 且使用 Railway 給的 `PORT`。
- **Google 登入失敗**：檢查 redirect URI 是否為 `https://你的網址/auth`，且 Client ID/Secret 與 Variables 一致。
- **資料重啟後不見**：需加 Volume 並把 DB / 匯出目錄放在掛載路徑。

---

## 六、嵌入 framelogic.app（React）

後端在 Railway 跑起來並取得網址後，在 Vite + React 專案中修改：

**`src/App.tsx`**

```tsx
import './index.css';

export default function App() {
  // 替換成你在 Railway 產生的專屬網址
  const STORYBREAK_URL = "https://你的railway專屬網址.up.railway.app";

  return (
    <div style={{ width: '100vw', height: '100vh', backgroundColor: '#121212' }}>
      <iframe
        src={STORYBREAK_URL}
        title="StoryBreak Pro Workstation"
        style={{
          width: '100%',
          height: '100%',
          border: 'none',
          display: 'block'
        }}
        allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
        allowFullScreen
      />
    </div>
  );
}
```

**`src/index.css`**（避免多餘邊距、雙捲軸）

```css
body, html, #root {
  margin: 0;
  padding: 0;
  width: 100vw;
  height: 100vh;
  overflow: hidden;
}
```
