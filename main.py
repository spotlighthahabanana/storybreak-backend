import os
from fastapi import FastAPI, Request
from fastapi.responses import RedirectResponse
from starlette.middleware.sessions import SessionMiddleware
from authlib.integrations.starlette_client import OAuth
import uvicorn
import gradio as gr

# 引入 UI 以及與 app.py 相同的主題、CSS、JS
from app import create_ui, sci_fi_theme, css_pro, js_drag_drop

# 密鑰只從環境變數讀取，不上傳到程式碼（Railway 後台填 GOOGLE_CLIENT_ID / GOOGLE_CLIENT_SECRET）
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID", "")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET", "")
SESSION_SECRET_KEY = os.getenv("SESSION_SECRET_KEY", "storybreak-super-secret-session-key")

app = FastAPI()
app.add_middleware(SessionMiddleware, secret_key=SESSION_SECRET_KEY)

oauth = OAuth()
oauth.register(
    name='google',
    client_id=GOOGLE_CLIENT_ID,
    client_secret=GOOGLE_CLIENT_SECRET,
    server_metadata_url='https://accounts.google.com/.well-known/openid-configuration',
    client_kwargs={'scope': 'openid email profile'}
)

@app.get('/login/google')
async def login(request: Request):
    # 使用者點擊按鈕後，跳轉到 Google 授權畫面（路徑避開 Gradio 內建 /login）
    redirect_uri = request.url_for('auth')
    return await oauth.google.authorize_redirect(request, redirect_uri)

@app.get('/auth')
async def auth(request: Request):
    # Google 驗證成功後跳回這裡
    try:
        token = await oauth.google.authorize_access_token(request)
        user = token.get('userinfo')
        if user:
            # 將 Google 回傳的使用者資訊存進 Session
            request.session['user'] = dict(user)
    except Exception as e:
        print(f"登入失敗: {e}")
    # 登入完成，跳轉回 Gradio 首頁
    return RedirectResponse(url='/')

@app.get('/logout_route')
async def logout_route(request: Request):
    # 清除 Session
    request.session.pop('user', None)
    return RedirectResponse(url='/')

# 呼叫你原本的函數建立 UI
ui = create_ui()

# 掛載在根路徑 /；allowed_paths 授權 Gradio 讀取 Volume 內檔案，避免 403 Forbidden
app = gr.mount_gradio_app(
    app, ui, path="/",
    theme=sci_fi_theme, css=css_pro, js=js_drag_drop,
    allowed_paths=["/app/output"],
)

if __name__ == "__main__":
    # Railway 會動態分配 PORT，沒有則預設 7860（本地開發）
    port = int(os.environ.get("PORT", "7860"))
    reload = os.getenv("RAILWAY_ENVIRONMENT") != "production"
    print(f"啟動 StoryBreak 伺服器... (port={port})")
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=reload)
