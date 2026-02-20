"""
AI Scene Detection Tool - Pro Workstation Edition (UX Optimized + AI Classification)
PySceneDetect backend, Sci-Fi UI, and AI scene classification.
"""

import os
import subprocess
import sqlite3
import tempfile
import hashlib
# Suppress FFmpeg/libavcodec H.264 decode warnings
os.environ.setdefault("AV_LOG_LEVEL", "-8")
import json
import socket
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
import gradio as gr

# Backend modules
try:
    from scene_detector import SceneDetector, SceneInfo, get_video_info
    from ai_annotator import AIAnnotator, LocalAnnotator
except ImportError:
    print("Error: scene_detector.py or ai_annotator.py not found. Please ensure files exist.")
    # Mock for UI dev
    class SceneInfo:
        def __init__(self, scene_number, start_time, end_time, start_frame, end_frame, duration, thumbnail_path=None, annotation=None, video_clip_path=None, tag=None, group_id=None, movement=None):
            self.scene_number = scene_number
            self.start_time = start_time
            self.end_time = end_time
            self.start_frame = start_frame
            self.end_frame = end_frame
            self.duration = duration
            self.thumbnail_path = thumbnail_path
            self.annotation = annotation
            self.video_clip_path = video_clip_path
            self.tag = tag
            self.group_id = group_id
            self.movement = movement
        @property
        def scene_id(self): return f"{self.start_frame}-{self.end_frame}"

    def get_video_info(path): return {'duration': 0, 'width': 0, 'height': 0, 'fps': 0}
    class SceneDetector:
        def process_video(self, *args, **kwargs): return []
        def extract_video_clips(self, *args, **kwargs): return []
        def extract_thumbnails(self, *args, **kwargs): return []
        def extract_thumbnail_at_position(self, *args, **kwargs): return None
        def extract_cover(self, *args, **kwargs): return None
        class ClassifierMock:
            def classify_scenes(self, scenes, video_path, progress_callback=None): return scenes
        classifier = ClassifierMock()

    class AIAnnotator:
        def set_api_key(self, key): pass
        def annotate_scenes(self, scenes, **kwargs): return scenes
    class LocalAnnotator:
        def annotate_scenes(self, scenes, **kwargs): return scenes

try:
    from video_llava import generate_scene_note
except ImportError:
    generate_scene_note = None

# --- Settings & constants ---
BASE_DIR = Path("output")
VIDEO_LIBRARY_FILE = BASE_DIR / "video_library.json"
CONFIG_FILE = BASE_DIR / "config.json"
SQLITE_DB = BASE_DIR / "storybreak.db"
ASSETS_DIR = Path(__file__).resolve().parent / "assets"
BASE_DIR.mkdir(parents=True, exist_ok=True)


def _init_sqlite_db():
    """Create SQLite DB and users + projects tables if not exist."""
    conn = sqlite3.connect(str(SQLITE_DB), timeout=10)
    try:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA busy_timeout=5000;")
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                email TEXT PRIMARY KEY,
                password_hash TEXT NOT NULL,
                plan TEXT NOT NULL DEFAULT 'Free',
                joined_at TEXT NOT NULL,
                credits INTEGER NOT NULL DEFAULT 5
            );
            CREATE TABLE IF NOT EXISTS projects (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_email TEXT NOT NULL DEFAULT 'guest',
                name TEXT NOT NULL,
                path TEXT NOT NULL,
                scenes_count INTEGER NOT NULL DEFAULT 0,
                duration REAL DEFAULT 0,
                width INTEGER DEFAULT 0,
                height INTEGER DEFAULT 0,
                fps REAL DEFAULT 0,
                thumbnail TEXT DEFAULT '',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                scenes_data TEXT NOT NULL DEFAULT '[]'
            );
            CREATE INDEX IF NOT EXISTS idx_projects_user ON projects(user_email);
            CREATE UNIQUE INDEX IF NOT EXISTS ux_projects_user_path ON projects(user_email, path);
        """)
        conn.commit()
        # One-time migrate from users.json if it exists
        users_json = BASE_DIR / "users.json"
        if users_json.exists():
            try:
                with open(users_json, "r", encoding="utf-8") as f:
                    users = json.load(f)
                for email, data in users.items():
                    if not isinstance(data, dict):
                        continue
                    pw = data.get("password") or data.get("password_hash")
                    if not pw:
                        continue
                    conn.execute(
                        "INSERT OR IGNORE INTO users (email, password_hash, plan, joined_at, credits) VALUES (?, ?, ?, ?, ?)",
                        (email, pw, data.get("plan", "Free"), data.get("joined_at", ""), data.get("credits", 5)),
                    )
                conn.commit()
            except Exception:
                pass
    finally:
        conn.close()
_init_sqlite_db()

# UI return tuple length (avoid magic numbers in handlers)
WORKSTATION_VIEW_OUTPUTS = 22   # load_video_to_workstation, handle_library_gallery_select (last 2 = welcome_screen, library_content)
SCENE_SELECT_OUTPUTS = 9        # scene select / prev / next

def _gr_updates(n: int):
    """Return tuple of n gr.update() for multi-output handlers. Use with WORKSTATION_VIEW_OUTPUTS / SCENE_SELECT_OUTPUTS."""
    return (gr.update(),) * n

DEFAULT_CONFIG = {
    "default_export_path": str(BASE_DIR / "exports"),
    "ui_scale": 100,
    "custom_tags": ["Indoor", "Outdoor", "Action", "Dialogue", "VFX", "Wide", "Close-up"],
    "user_logo_path": "",
    "export_fps": 30,  # Assembly export video: 24 | 25 | 30
}

def load_config() -> dict:
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                return {**DEFAULT_CONFIG, **json.load(f)}
        except Exception:
            pass
    return DEFAULT_CONFIG.copy()

def save_config(cfg: dict):
    try:
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

# --- Data management (SQLite projects table) ---
class VideoLibraryManager:
    def __init__(self, db_path: Path, legacy_json_path: Optional[Path] = None):
        self.db_path = db_path
        self.legacy_json_path = legacy_json_path or (BASE_DIR / "video_library.json")
        self._migrated = False

    def _conn(self):
        conn = sqlite3.connect(str(self.db_path), timeout=10)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA busy_timeout=5000;")
        return conn

    def _migrate_from_json_if_needed(self, user_email: str):
        if self._migrated or user_email != "guest":
            return
        if not self.legacy_json_path.exists():
            self._migrated = True
            return
        try:
            conn = self._conn()
            try:
                n = conn.execute("SELECT COUNT(*) FROM projects WHERE user_email = ?", ("guest",)).fetchone()[0]
                if n > 0:
                    self._migrated = True
                    return
                with open(self.legacy_json_path, "r", encoding="utf-8") as f:
                    library = json.load(f)
                for item in library:
                    path_val = item.get("path", "")
                    path_resolved = str(Path(path_val).resolve()) if path_val else ""
                    conn.execute(
                        """INSERT INTO projects (user_email, name, path, scenes_count, duration, width, height, fps, thumbnail, created_at, updated_at, scenes_data)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (
                            "guest",
                            item.get("name", ""),
                            path_resolved,
                            item.get("scenes_count", 0),
                            item.get("duration", 0),
                            item.get("width", 0),
                            item.get("height", 0),
                            item.get("fps", 0),
                            item.get("thumbnail", ""),
                            item.get("created_at", ""),
                            item.get("updated_at", ""),
                            json.dumps(item.get("scenes_data", []), ensure_ascii=False),
                        ),
                    )
                conn.commit()
            finally:
                conn.close()
            self._migrated = True
        except Exception:
            pass

    def load(self, user_email: str = "guest") -> List[Dict]:
        self._migrate_from_json_if_needed(user_email)
        conn = self._conn()
        try:
            rows = conn.execute(
                "SELECT name, path, scenes_count, duration, width, height, fps, thumbnail, created_at, updated_at, scenes_data FROM projects WHERE user_email = ? ORDER BY updated_at DESC",
                (user_email,),
            ).fetchall()
            out = []
            for r in rows:
                name, path, scenes_count, duration, width, height, fps, thumbnail, created_at, updated_at, scenes_data = r
                try:
                    scenes_list = json.loads(scenes_data) if isinstance(scenes_data, str) else (scenes_data or [])
                except Exception:
                    scenes_list = []
                out.append({
                    "name": name,
                    "path": path,
                    "scenes_count": scenes_count,
                    "duration": duration,
                    "width": width,
                    "height": height,
                    "fps": fps,
                    "thumbnail": thumbnail or "",
                    "created_at": created_at or "",
                    "updated_at": updated_at or "",
                    "scenes_data": scenes_list,
                })
            return out
        finally:
            conn.close()

    def add_or_update(self, video_path: str, video_info: dict, scenes: List[SceneInfo], thumbnail: str, user_email: str = "guest"):
        video_name = Path(video_path).name
        scenes_data = [s.to_dict() if hasattr(s, "to_dict") and callable(getattr(s, "to_dict")) else _scene_to_dict(s) for s in scenes]
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        path_resolved = str(Path(video_path).resolve())
        conn = self._conn()
        try:
            row = conn.execute(
                "SELECT id, created_at FROM projects WHERE user_email = ? AND path = ?",
                (user_email, path_resolved),
            ).fetchone()
            if row:
                created_at = row[1]
                conn.execute(
                    "UPDATE projects SET name=?, scenes_count=?, duration=?, width=?, height=?, fps=?, thumbnail=?, updated_at=?, scenes_data=? WHERE id=?",
                    (video_name, len(scenes), video_info.get("duration", 0), video_info.get("width", 0), video_info.get("height", 0), video_info.get("fps", 0), str(thumbnail), timestamp, json.dumps(scenes_data, ensure_ascii=False), row[0]),
                )
            else:
                created_at = timestamp
                conn.execute(
                    """INSERT INTO projects (user_email, name, path, scenes_count, duration, width, height, fps, thumbnail, created_at, updated_at, scenes_data)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (user_email, video_name, path_resolved, len(scenes), video_info.get("duration", 0), video_info.get("width", 0), video_info.get("height", 0), video_info.get("fps", 0), str(thumbnail), created_at, timestamp, json.dumps(scenes_data, ensure_ascii=False)),
                )
            conn.commit()
        finally:
            conn.close()

    def delete_at_index(self, index: int, user_email: str = "guest") -> bool:
        library = self.load(user_email)
        if index < 0 or index >= len(library):
            return False
        path_to_remove = str(Path(library[index]["path"]).resolve())
        conn = self._conn()
        try:
            conn.execute("DELETE FROM projects WHERE user_email = ? AND path = ?", (user_email, path_to_remove))
            conn.commit()
            return True
        finally:
            conn.close()

# Singleton
lib_manager = VideoLibraryManager(SQLITE_DB, VIDEO_LIBRARY_FILE)
detector = SceneDetector()
ai_annotator = AIAnnotator()

# --- User Management & Subscription System (SQLite) ---
def _hash_password_bcrypt(password: str) -> str:
    """bcrypt 加鹽雜湊，正式環境請用此。"""
    try:
        import bcrypt
        return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
    except Exception:
        return hashlib.sha256(password.encode()).hexdigest()


def _allowed_license_keys() -> set:
    """環境變數 STORYBREAK_LICENSE_KEYS 允許多把 key，分號分隔。若未設則回傳空 set（不強制白名單）。"""
    raw = (os.getenv("STORYBREAK_LICENSE_KEYS", "") or "").strip()
    return {k.strip() for k in raw.split(";") if k.strip()}


def _verify_password(stored_hash: str, password: str) -> bool:
    """驗證密碼：支援 bcrypt、舊版 SHA256；Google-only 帳號不可用密碼登入。"""
    if not stored_hash:
        return False
    if stored_hash in ("__google_only__", "__license__"):
        return False
    try:
        import bcrypt
        if stored_hash.startswith("$2") or stored_hash.startswith("$argon2"):
            return bcrypt.checkpw(password.encode("utf-8"), stored_hash.encode("utf-8"))
    except Exception:
        pass
    legacy = hashlib.sha256(password.encode()).hexdigest()
    return legacy == stored_hash


class UserManager:
    def __init__(self, db_path: Path):
        self.db_path = db_path

    def _conn(self):
        conn = sqlite3.connect(str(self.db_path), timeout=10)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA busy_timeout=5000;")
        return conn

    def _hash_password(self, password: str) -> str:
        return _hash_password_bcrypt(password)

    def register(self, email: str, password: str):
        try:
            conn = self._conn()
            try:
                conn.execute(
                    "INSERT INTO users (email, password_hash, plan, joined_at, credits) VALUES (?, ?, ?, ?, ?)",
                    (email, self._hash_password(password), "Free", datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 5),
                )
                conn.commit()
                return True, "Account created! Please sign in."
            except sqlite3.IntegrityError:
                return False, "Email already registered."
            finally:
                conn.close()
        except Exception:
            return False, "Registration failed. Please try again."

    @staticmethod
    def _looks_like_license(s: str) -> bool:
        """判斷字串是否像 License Key（用來決定是否走 license 登入）。"""
        s = (s or "").strip()
        if len(s) < 8:
            return False
        if "-" in s and all(part for part in s.split("-")):
            return True
        return len(s) >= 16

    def login(self, email: str, password: str, license_key: str = ""):
        email = (email or "").strip()
        password = (password or "").strip()

        env_user = os.getenv("STORYBREAK_USER")
        env_pass = os.getenv("STORYBREAK_PASS")
        if env_user and email == env_user and password == env_pass:
            return True, {"email": email, "plan": "Admin", "credits": 9999}

        # License 登入：用使用者輸入的 password 當 key（只要看起來像 key）
        if email and password and self._looks_like_license(password):
            user_data = self.ensure_license_user(email, password)
            if user_data:
                return True, user_data
            return False, "Invalid License Key."

        # 若呼叫端有傳入 license_key 參數也可當 key 用
        if (license_key or "").strip() and email and password == (license_key or "").strip():
            user_data = self.ensure_license_user(email, (license_key or "").strip())
            if user_data:
                return True, user_data
            return False, "Invalid License Key."

        # 一般帳密登入
        conn = self._conn()
        try:
            row = conn.execute(
                "SELECT password_hash, plan, joined_at, credits FROM users WHERE email = ?",
                (email,),
            ).fetchone()
            if not row:
                return False, "User not found."
            password_hash, plan, joined_at, credits = row
            if not _verify_password(password_hash, password):
                return False, "Invalid password."
            return True, {"email": email, "plan": plan, "joined_at": joined_at, "credits": credits}
        finally:
            conn.close()

    def upgrade_to_pro(self, email: str, license_key: str):
        key = (license_key or "").strip()
        if not key or len(key) < 8:
            return False, "Invalid License Key."
        allowed = _allowed_license_keys()
        if allowed and key not in allowed:
            return False, "Invalid License Key."
        conn = self._conn()
        try:
            cur = conn.execute("UPDATE users SET plan = 'Pro', credits = 9999 WHERE email = ?", (email,))
            conn.commit()
            if cur.rowcount:
                return True, "Upgraded to Pro! Welcome to StoryBreak Pro."
            return False, "User not found."
        finally:
            conn.close()

    def get_user_by_email(self, email: str) -> Optional[dict]:
        """從 DB 讀取使用者，權威來源。若不存在回傳 None。"""
        if not email or email == "guest":
            return None
        conn = self._conn()
        try:
            row = conn.execute(
                "SELECT plan, joined_at, credits FROM users WHERE email = ?",
                (email,),
            ).fetchone()
            if not row:
                return None
            plan, joined_at, credits = row
            return {"email": email, "plan": plan, "joined_at": joined_at, "credits": credits}
        finally:
            conn.close()

    def ensure_google_user(self, email: str, name: str = "", picture: str = "") -> dict:
        """Google 登入後：若 DB 無此 email 則寫入（密碼佔位），再從 DB 讀回 plan/credits。回傳完整 user_data 供 state 使用。"""
        if not email or email == "guest":
            return {"email": "guest", "plan": "Pro", "credits": 9999}
        conn = self._conn()
        try:
            existing = conn.execute("SELECT plan, credits FROM users WHERE email = ?", (email,)).fetchone()
            if not existing:
                conn.execute(
                    "INSERT INTO users (email, password_hash, plan, joined_at, credits) VALUES (?, ?, ?, ?, ?)",
                    (email, "__google_only__", "Free", datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 5),
                )
                conn.commit()
            row = conn.execute(
                "SELECT plan, joined_at, credits FROM users WHERE email = ?",
                (email,),
            ).fetchone()
            plan, joined_at, credits = row
            return {
                "email": email,
                "name": name,
                "picture": picture,
                "plan": plan,
                "credits": credits,
                "joined_at": joined_at,
            }
        finally:
            conn.close()

    def ensure_license_user(self, email: str, license_key: str) -> Optional[dict]:
        """License Key 登入：若 DB 無此 email 則寫入，再 upgrade to Pro，回傳 user_data（權威寫回 users 表）。"""
        email = (email or "").strip()
        license_key = (license_key or "").strip()

        if not email or not license_key:
            return None

        conn = self._conn()
        try:
            existing = conn.execute(
                "SELECT 1 FROM users WHERE email = ?",
                (email,)
            ).fetchone()

            if not existing:
                conn.execute(
                    """
                    INSERT INTO users (email, password_hash, plan, joined_at, credits)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (email, "__license__", "Free", datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 5),
                )
                conn.commit()
        finally:
            conn.close()

        # upgrade_to_pro 回傳 (success, msg)，必須 unpack
        success, _msg = self.upgrade_to_pro(email, license_key)
        if not success:
            return None

        user = self.get_user_by_email(email)
        if user:
            return user

        return {"email": email, "plan": "Pro", "credits": 9999}


user_manager = UserManager(SQLITE_DB)

# --- Helpers ---
TAG_SEP = " | "
TAG_SEP_ALTS = (" | ", "|", "、", "，")

def normalize_tag(tag_text: Optional[str]) -> str:
    """Normalize tag string: split by any of TAG_SEP_ALTS, join with TAG_SEP."""
    if not tag_text or not str(tag_text).strip():
        return ""
    text = str(tag_text).strip()
    parts = []
    for sep in TAG_SEP_ALTS:
        if sep in text:
            parts = [p.strip() for p in text.split(sep) if p.strip()]
            break
    if not parts:
        return text
    return TAG_SEP.join(parts)

def _scene_to_dict(s) -> dict:
    """Fallback when SceneInfo has no to_dict (e.g. mock)."""
    return {
        "scene_number": s.scene_number,
        "scene_id": getattr(s, "scene_id", None) or f"{getattr(s, 'start_frame', 0)}-{getattr(s, 'end_frame', 0)}",
        "start_time": getattr(s, "start_time", 0),
        "end_time": getattr(s, "end_time", 0),
        "duration": getattr(s, "duration", 0),
        "start_frame": getattr(s, "start_frame", 0),
        "end_frame": getattr(s, "end_frame", 0),
        "thumbnail_path": str(s.thumbnail_path) if getattr(s, "thumbnail_path", None) else None,
        "video_clip_path": str(s.video_clip_path) if getattr(s, "video_clip_path", None) else None,
        "annotation": getattr(s, "annotation", None),
        "tag": getattr(s, "tag", None),
        "group_id": getattr(s, "group_id", None),
        "movement": getattr(s, "movement", None),
    }

def format_time(seconds: float) -> str:
    if seconds is None: return "00:00.00"
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes:02d}:{secs:05.2f}"


def _is_file_path(p: Optional[str]) -> bool:
    """可用的檔案路徑（非 data-uri、且檔案存在）。"""
    if not p:
        return False
    s = str(p).strip()
    if s.startswith("data:"):
        return False
    return os.path.exists(s)

def get_library_gallery_data(user_email: Optional[str] = None):
    library = lib_manager.load(user_email or "guest")
    gallery = []
    for item in library:
        thumb = item.get('thumbnail', '')
        res = f"{item.get('width')}x{item.get('height')}"
        caption = f"{item.get('name')}\n{item.get('scenes_count')} Scenes | {format_time(item.get('duration', 0))} | {res}"
        gallery.append((thumb, caption))
    return gallery


def get_library_gallery_data_and_visibility(state=None):
    """Return (gallery_data, welcome_screen visible, library_content visible) for refresh."""
    user_email = (state.get("user", {}).get("email") if state else None) or "guest"
    data = get_library_gallery_data(user_email)
    w_vis, l_vis = _library_welcome_visibility(user_email)
    return data, w_vis, l_vis

def _get_gallery_items_from_scenes(scenes):
    return [(s.thumbnail_path, f"#{s.scene_number} {s.tag or ''}") for s in scenes if s.thumbnail_path]


def get_all_scenes_flat(user_email: Optional[str] = None):
    """Get all projects' scenes (gallery items + metadata) for Assembly. Never raises — returns ([], []) on error to avoid UI error toasts."""
    try:
        library = lib_manager.load(user_email or "guest")
        items = []
        meta = []
        for item in library:
            video_name = item.get("name", "Unknown")
            video_path = item.get("path", "")
            for s in item.get("scenes_data", []):
                thumb = s.get("thumbnail_path")
                if _is_file_path(thumb):
                    tag = s.get("tag") or ""
                    caption = f"{video_name} | #{s.get('scene_number', 0)} {tag}"
                    items.append((thumb, caption))
                    meta.append({
                        "video_name": video_name,
                        "video_path": video_path,
                        "scene_data": s,
                        "thumbnail_path": thumb,
                        "caption": caption,
                    })
        return items, meta
    except Exception:
        return [], []


def get_all_scenes_gallery_data():
    items, _ = get_all_scenes_flat()
    return items


NUM_ASSEMBLY_SLOTS = 30

_PLACEHOLDER_PATH = None
# 1x1 grey PNG (avoids Gallery error toast when placeholder file is missing)
_FALLBACK_IMAGE_DATA = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="

def _get_placeholder_path() -> str:
    """Create and return placeholder image path for empty slots. Never returns empty string to avoid Gallery load errors."""
    global _PLACEHOLDER_PATH
    if _PLACEHOLDER_PATH and os.path.exists(_PLACEHOLDER_PATH):
        return _PLACEHOLDER_PATH
    out = BASE_DIR / "assembly_placeholder.png"
    try:
        from PIL import Image, ImageDraw, ImageFont
        w, h = 200, 120
        img = Image.new("RGB", (w, h), color=(45, 55, 72))
        draw = ImageDraw.Draw(img)
        draw.rectangle([2, 2, w-3, h-3], outline=(100, 116, 139), width=2)
        try:
            font = ImageFont.truetype("arial.ttf", 14)
        except Exception:
            font = ImageFont.load_default()
        text = "+"
        try:
            bbox = draw.textbbox((0, 0), text, font=font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        except AttributeError:
            tw, th = 12, 14
        draw.text(((w - tw) // 2, (h - th) // 2 - 5), text, fill=(148, 163, 184), font=font)
        draw.text((w//2 - 40, h//2 + 15), "Click to add", fill=(100, 116, 139), font=font)
        img.save(str(out))
        _PLACEHOLDER_PATH = str(out)
        return _PLACEHOLDER_PATH
    except Exception:
        try:
            from PIL import Image
            img = Image.new("RGB", (200, 120), color=(38, 38, 38))
            img.save(str(out))
            _PLACEHOLDER_PATH = str(out)
            return _PLACEHOLDER_PATH
        except Exception:
            pass
    if out.exists():
        _PLACEHOLDER_PATH = str(out)
        return _PLACEHOLDER_PATH
    try:
        with open(out, "wb") as f:
            f.write(b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x05\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82')
        _PLACEHOLDER_PATH = str(out)
        return _PLACEHOLDER_PATH
    except Exception:
        pass
    return ""


def _assembly_to_gallery(assembly: List) -> List[tuple]:
    """Convert assembly list to gallery items; empty slots show placeholder. Avoid empty image path to prevent Gallery error toasts."""
    try:
        ph = _get_placeholder_path()
        if not ph and (BASE_DIR / "assembly_placeholder.png").exists():
            ph = str(BASE_DIR / "assembly_placeholder.png")
        a = _normalize_assembly(assembly)
        result = []
        safe_placeholder = ph or _FALLBACK_IMAGE_DATA
        for i in range(NUM_ASSEMBLY_SLOTS):
            if i < len(a) and a[i] is not None:
                m = a[i]
                thumb = m.get("thumbnail_path") or ph
                if not _is_file_path(thumb):
                    thumb = safe_placeholder
                atype = m.get("asset_type", "clip")
                result.append((thumb, f"#{i+1} {atype.upper()} | {m.get('video_name','')} | {m.get('scene_data',{}).get('tag','')}"))
            else:
                result.append((safe_placeholder, f"Slot {i+1}"))
        return result
    except Exception:
        return [(_FALLBACK_IMAGE_DATA, f"Slot {i+1}") for i in range(NUM_ASSEMBLY_SLOTS)]


def _assembly_to_gallery_legacy(assembly: List[Dict]) -> List[tuple]:
    return [(m["thumbnail_path"], f"{m['video_name']} | #{m['scene_data'].get('scene_number', 0)} {m['scene_data'].get('tag', '')}") for m in assembly]


def _normalize_assembly(a: List) -> List:
    """Normalize assembly to length NUM_ASSEMBLY_SLOTS, pad with None."""
    a = list(a) if a else []
    while len(a) < NUM_ASSEMBLY_SLOTS:
        a.append(None)
    return a[:NUM_ASSEMBLY_SLOTS]


def assembly_add_scene(all_meta: List, assembly: List, idx: int):
    if not all_meta or idx < 0 or idx >= len(all_meta):
        return _normalize_assembly(assembly)
    m = dict(all_meta[idx])
    m["asset_type"] = "clip"  # 來自場景的片段，可 Export Video/EDL
    a = _normalize_assembly(assembly)
    for i in range(len(a)):
        if a[i] is None:
            a[i] = m
            return a
    return a


def assembly_upload_handler(assembly: List, uploaded_file):
    """Add uploaded file to assembly and clear upload field."""
    asm = assembly_add_upload(assembly, uploaded_file)
    e, t = _assembly_empty_visibility(asm)
    return asm, _assembly_to_gallery(asm), None, e, t

def assembly_add_upload(assembly: List, files) -> List:
    """Add uploaded file to assembly (first empty slot)."""
    if not files:
        return _normalize_assembly(assembly)
    path = files.get("path", files) if isinstance(files, dict) else (files[0] if isinstance(files, (list, tuple)) else files)
    path = str(path)
    if not path or not os.path.exists(path):
        return _normalize_assembly(assembly)
    m = {"video_name": Path(path).stem, "video_path": None, "scene_data": {"scene_number": 0, "tag": "Upload", "annotation": ""}, "thumbnail_path": path, "caption": Path(path).stem, "asset_type": "image"}
    a = _normalize_assembly(assembly)
    for i in range(len(a)):
        if a[i] is None:
            a[i] = m
            return a
    return a


def assembly_remove_scene(assembly: List, slot_idx: int):
    """Clear slot at slot_idx (0-based)."""
    a = _normalize_assembly(assembly)
    if slot_idx < 0 or slot_idx >= len(a):
        return a
    a[slot_idx] = None
    filled = [x for x in a if x is not None]
    return filled + [None] * (NUM_ASSEMBLY_SLOTS - len(filled))


def assembly_move_scene(assembly: List, slot_idx: int, direction: int):
    """Move scene in slot slot_idx by delta positions."""
    a = _normalize_assembly(assembly)
    if slot_idx < 0 or slot_idx >= len(a):
        return a
    new_idx = slot_idx + direction
    if new_idx < 0 or new_idx >= len(a):
        return a
    a[slot_idx], a[new_idx] = a[new_idx], a[slot_idx]
    return a


def assembly_refresh_fn(state=None):
    user_email = (state.get("user", {}).get("email") if state else None) or "guest"
    items, meta = get_all_scenes_flat(user_email)
    return items, meta


def assembly_add_handler_from_drop(idx, all_meta: List, my_asm: List):
    """On drop: add scene to Assembly by source index."""
    try:
        i = int(float(idx)) if idx is not None else -1
    except (TypeError, ValueError):
        i = -1
    if i < 0:
        asm = my_asm or [None] * NUM_ASSEMBLY_SLOTS
        e, t = _assembly_empty_visibility(asm)
        return asm, gr.update(), "-1", e, t
    new_asm = assembly_add_scene(all_meta or [], my_asm or [], i)
    e, t = _assembly_empty_visibility(new_asm)
    return new_asm, _assembly_to_gallery(new_asm), "-1", e, t


def assembly_add_handler(evt: gr.SelectData, all_meta: List, my_asm: List):
    my_asm = assembly_add_scene(all_meta, my_asm or [], evt.index)
    e, t = _assembly_empty_visibility(my_asm)
    return my_asm, _assembly_to_gallery(my_asm), e, t


def assembly_select_handler(evt: gr.SelectData, my_asm: List):
    idx = evt.index
    filled = sum(1 for x in (my_asm or []) if x is not None)
    status = f"Selected slot {idx + 1} ({filled}/{NUM_ASSEMBLY_SLOTS} used)"
    return idx, status


def assembly_remove_handler(my_asm: List, sel_idx: int):
    my_asm = assembly_remove_scene(my_asm or [], sel_idx)
    e, t = _assembly_empty_visibility(my_asm)
    return my_asm, _assembly_to_gallery(my_asm), -1, "Removed", e, t


def assembly_move_up_handler(my_asm: List, sel_idx: int):
    my_asm = assembly_move_scene(my_asm or [], sel_idx, -1)
    new_idx = max(0, sel_idx - 1) if sel_idx > 0 else -1
    e, t = _assembly_empty_visibility(my_asm)
    return my_asm, _assembly_to_gallery(my_asm), new_idx, f"Moved up to position {new_idx + 1}", e, t


def assembly_move_down_handler(my_asm: List, sel_idx: int):
    my_asm = assembly_move_scene(my_asm or [], sel_idx, 1)
    new_idx = min(len(my_asm or []) - 1, sel_idx + 1) if sel_idx >= 0 else -1
    e, t = _assembly_empty_visibility(my_asm)
    return my_asm, _assembly_to_gallery(my_asm), new_idx, f"Moved down to position {new_idx + 1}", e, t


def assembly_clear_handler():
    empty = [None] * NUM_ASSEMBLY_SLOTS
    return empty, _assembly_to_gallery(empty), -1, "Cleared", gr.update(visible=True), gr.update(visible=False)


def _assembly_empty_visibility(assembly: List):
    """Return (show_empty_state, show_timeline) from assembly list. Use gr.update so Gradio correctly toggles visibility."""
    n = sum(1 for x in (assembly or []) if x is not None)
    return gr.update(visible=(n == 0)), gr.update(visible=(n > 0))


def assembly_export_handler(my_asm: List, project_name="", director_notes=""):
    filled = [x for x in (my_asm or []) if x is not None]
    path, err = export_assembly_pdf(filled, project_name=project_name or None, director_notes=director_notes or None)
    if path:
        return gr.update(value=path, visible=True), gr.update(value="")
    if err:
        return gr.update(visible=False), gr.update(value=f"⚠️ {err}", visible=True)
    return gr.update(visible=False), gr.update(value="")


def export_assembly_pdf(assembly: List, project_name: Optional[str] = None, director_notes: Optional[str] = None) -> tuple:
    """Export Assembly timeline to PDF. Returns (path, error_message); error_message is None on success."""
    assembly = [x for x in (assembly or []) if x is not None]
    if not assembly:
        return None, None
    try:
        from fpdf import FPDF
    except ImportError:
        return None, "Assembly PDF export requires: pip install fpdf2"

    out_dir = BASE_DIR / "exports"
    out_dir.mkdir(parents=True, exist_ok=True)
    base_name = (project_name or "Assembly").strip() or "Assembly"
    out_path = out_dir / f"{base_name}_shotlist_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    _cjk_font_path = None
    for p in [
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        Path("/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"),
        Path("C:/Windows/Fonts/msyh.ttf"),
        Path("C:/Windows/Fonts/msyhbd.ttf"),
        Path("C:/Windows/Fonts/simhei.ttf"),
        Path("C:/Windows/Fonts/msjh.ttf"),
        Path("C:/Windows/Fonts/msjh.ttc"),
        Path("C:/Windows/Fonts/msjhbd.ttc"),
        Path("C:/Windows/Fonts/mingliu.ttc"),
    ]:
        if p.exists():
            _cjk_font_path = str(p)
            break

    _user_logo = (load_config().get("user_logo_path") or "").strip()
    _logo_path = Path(_user_logo).resolve() if _user_logo and Path(_user_logo).exists() else None

    class PDF(FPDF):
        def __init__(self, logo_path=None):
            super().__init__()
            self._logo_path = logo_path
            self.set_auto_page_break(auto=True, margin=15)
            self.set_margins(15, 15, 15)
            self.add_page()
            if _cjk_font_path:
                try:
                    self.add_font("CJK", "", _cjk_font_path)
                    self._use_cjk = True
                except Exception:
                    self._use_cjk = False
            else:
                self._use_cjk = False

        def header(self):
            if self._logo_path and os.path.exists(self._logo_path):
                try:
                    self.image(str(self._logo_path), 10, 8, w=33)
                except Exception:
                    pass

        def footer(self):
            self.set_y(-12)
            self.set_font("Helvetica", "I", 7)
            self.set_text_color(180, 180, 180)
            self.cell(0, 8, "Generated by StoryBreak", 0, 0, "R")

    pdf = PDF(logo_path=_logo_path)
    pdf.set_margins(15, 15, 15)
    font_name = "CJK" if pdf._use_cjk else "Helvetica"

    def _safe(t):
        return (t or "") if pdf._use_cjk else (t or "").encode("ascii", "replace").decode("ascii")

    # --- Shot list header (first page only) ---
    header_h = 0
    if project_name or director_notes:
        pdf.set_font(font_name, "B", 14)
        pdf.set_xy(15, 15)
        title = _safe((project_name or "Assembly").strip() or "Assembly Shot List")
        pdf.cell(0, 8, title)
        y_cur = 26
        if director_notes and (director_notes := director_notes.strip()):
            pdf.set_font(font_name, "", 9)
            pdf.set_xy(15, y_cur)
            pdf.cell(0, 5, _safe("Director's Notes"))
            pdf.set_font(font_name, "", 8)
            pdf.set_xy(15, y_cur + 6)
            pdf.multi_cell(180, 4, _safe(director_notes[:500]))
            y_cur += 6 + max(4 * ((len(director_notes) // 45) + 1), 8)
        pdf.set_draw_color(180, 180, 180)
        pdf.line(15, y_cur + 2, 195, y_cur + 2)
        header_h = y_cur + 8

    block_h = (297 - 15 - 15 - header_h) / 3 if header_h else (297 - 30) / 3
    img_h = block_h * 0.65
    img_w = min(210 - 30, img_h * 16 / 9)
    y_base = 15 + header_h

    for i, m in enumerate(assembly):
        if i > 0 and i % 3 == 0:
            pdf.add_page()
            y_base = 15
            block_h = (297 - 30) / 3
            img_h = block_h * 0.65
            img_w = min(210 - 30, img_h * 16 / 9)
        y = y_base + (i % 3) * block_h
        s = m["scene_data"]
        if _is_file_path(m.get("thumbnail_path")):
            try:
                pdf.image(m["thumbnail_path"], 15, y, w=img_w, h=img_h)
            except Exception:
                pass
        txt_y = y + img_h + 3
        pdf.set_font(font_name, "", 9)
        pdf.set_xy(15, txt_y)
        pdf.cell(0, 5, f"#{i+1} {m['video_name']} | #{s.get('scene_number',0)}")
        tag = (s.get("tag") or "")[:50]
        if tag:
            pdf.set_xy(15, txt_y + 6)
            pdf.cell(0, 5, f"Tag: {_safe(tag)}")
        anno = (s.get("annotation") or "").replace("\n", " ")[:80]
        if anno:
            pdf.set_xy(15, txt_y + 12)
            pdf.multi_cell(180, 5, _safe(anno))
    pdf.output(str(out_path))
    return str(out_path), None


def _assembly_has_images(assembly: List) -> bool:
    """Timeline 中是否含有圖片（僅圖片無法 Export Video/EDL）。"""
    for m in (assembly or []):
        if m is None:
            continue
        at = m.get("asset_type")
        if at == "image":
            return True
        if at != "clip":
            s = m.get("scene_data") or {}
            if not s.get("video_clip_path"):
                return True
    return False


def export_assembly_video(assembly: List, progress=gr.Progress()) -> tuple:
    """Normalize and concatenate Assembly clips to video. Returns (path, status_msg). FPS from config (export_fps: 24/25/30)."""
    filled = [x for x in (assembly or []) if x is not None]
    if not filled:
        return None, "Timeline is empty. Please add scenes first."
    if _assembly_has_images(filled):
        return None, "Timeline contains image(s). Export Video only supports video clips. Use Export PDF for images."

    cfg = load_config()
    export_fps = int(cfg.get("export_fps", 30))
    if export_fps not in (24, 25, 30):
        export_fps = 30

    try:
        import imageio_ffmpeg
        ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        ffmpeg_path = "ffmpeg"

    out_dir = BASE_DIR / "exports"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"Assembly_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"

    temp_dir = Path(tempfile.mkdtemp())
    valid_clips = []

    try:
        # 1. Collect valid clips (must have video_clip_path)
        for m in filled:
            s = m.get("scene_data") or {}
            cp = s.get("video_clip_path")
            if cp and os.path.exists(str(cp)):
                valid_clips.append(str(Path(cp).resolve()))

        if not valid_clips:
            return None, "No exportable scene clips. Ensure: 1) Scenes are from New Task detection with clips generated. 2) Not uploaded images (only pre-extracted video clips are supported)."

        # 2. Build filter: 1080p, unified FPS and audio
        input_args = []
        filter_v = ""
        filter_a = ""
        creation_flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0

        for i, path in enumerate(valid_clips):
            input_args.extend(["-i", path])
            filter_v += (
                f"[{i}:v]scale=1920:1080:force_original_aspect_ratio=decrease,"
                f"pad=1920:1080:(ow-iw)/2:(oh-ih)/2,setsar=1,fps={export_fps}[v{i}];"
            )
            filter_a += f"[{i}:a]aresample=44100,aformat=sample_fmts=fltp:channel_layouts=stereo[a{i}];"

        concat_inputs = "".join([f"[v{i}][a{i}]" for i in range(len(valid_clips))])
        final_filter = f"{filter_v}{filter_a}{concat_inputs}concat=n={len(valid_clips)}:v=1:a=1[outv][outa]"

        cmd = [
            ffmpeg_path, "-y",
            *input_args,
            "-filter_complex", final_filter,
            "-map", "[outv]", "-map", "[outa]",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
            "-c:a", "aac", "-b:a", "128k",
            "-movflags", "+faststart",
            str(out_path)
        ]

        if progress:
            progress(0.5, "Rendering and normalizing video...")

        result = subprocess.run(cmd, capture_output=True, text=True, creationflags=creation_flags)

        if result.returncode != 0:
            stderr = result.stderr or ""
            if "Stream specifier" in stderr and ":a" in stderr:
                # Some clips have no audio, fallback to video-only concat
                if progress:
                    progress(0.6, "Some clips have no audio, concatenating video only...")
                filter_v_only = ""
                concat_v_only = ""
                for i in range(len(valid_clips)):
                    filter_v_only += (
                        f"[{i}:v]scale=1920:1080:force_original_aspect_ratio=decrease,"
                        f"pad=1920:1080:(ow-iw)/2:(oh-ih)/2,setsar=1,fps={export_fps}[v{i}];"
                    )
                    concat_v_only += f"[v{i}]"
                filter_v_only += f"{concat_v_only}concat=n={len(valid_clips)}:v=1:a=0[outv]"
                cmd2 = [
                    ffmpeg_path, "-y", *input_args,
                    "-filter_complex", filter_v_only,
                    "-map", "[outv]", "-an",
                    "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
                    "-movflags", "+faststart", str(out_path)
                ]
                r2 = subprocess.run(cmd2, capture_output=True, text=True, creationflags=creation_flags)
                if r2.returncode == 0:
                    return str(out_path), "Export done (some clips had no audio, skipped). Use the download link below."
            err = stderr[-400:]
            return None, f"FFmpeg error:\n{err}"

        return str(out_path), f"Export done. {len(valid_clips)} clip(s). Use the Download Video link below."

    except Exception as e:
        return None, f"Export failed: {str(e)}"
    finally:
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)


def export_assembly_edl(assembly: List, project_name: Optional[str] = None) -> tuple:
    """Export Assembly timeline to standard CMX3600 EDL format."""
    filled = [x for x in (assembly or []) if x is not None]
    if not filled:
        return None, "Timeline is empty. Please add scenes first."
    if _assembly_has_images(filled):
        return None, "Timeline contains image(s). EDL export only supports video clips. Use Export PDF for images."

    fps = int(load_config().get("export_fps", 30))

    def seconds_to_tc(seconds: float, fps: int) -> str:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        f = int(round((seconds - int(seconds)) * fps))
        return f"{h:02d}:{m:02d}:{s:02d}:{f:02d}"

    title = (project_name or "StoryBreak_Assembly").strip() or "Assembly"
    out_dir = BASE_DIR / "exports"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{title}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.edl"

    lines = [f"TITLE: {title}", "FCM: NON-DROP FRAME", ""]
    record_start = 0.0

    for i, m in enumerate(filled):
        s = m.get("scene_data") or {}
        clip_path = s.get("video_clip_path") or m.get("video_path") or f"Clip_{i}.mp4"
        clip_name = Path(clip_path).name

        event_num = f"{i+1:03d}"
        reel = "AX"

        src_start = float(s.get("start_time", 0.0))
        src_end = float(s.get("end_time", 5.0))
        duration = src_end - src_start
        rec_end = record_start + duration

        tc_src_in = seconds_to_tc(src_start, fps)
        tc_src_out = seconds_to_tc(src_end, fps)
        tc_rec_in = seconds_to_tc(record_start, fps)
        tc_rec_out = seconds_to_tc(rec_end, fps)

        lines.append(f"{event_num}  {reel}       V     C        {tc_src_in} {tc_src_out} {tc_rec_in} {tc_rec_out}")
        lines.append(f"* FROM CLIP NAME: {clip_name}")
        lines.append("")

        record_start = rec_end

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return str(out_path), "✅ EDL Exported! Ready for DaVinci Resolve."


def edl_export_handler(my_asm: List, project_name=""):
    path, msg = export_assembly_edl(my_asm, project_name=project_name or None)
    if path:
        return gr.update(value=path, visible=True), msg
    return gr.update(visible=False), f"⚠️ {msg}"


def export_pdf_handler(state, project_name="", director_notes=""):
    """Export scenes PDF; return File component update for download link."""
    path = export_scenes_to_pdf(state, project_name=project_name or None, director_notes=director_notes or None)
    if path:
        return gr.update(value=path, visible=True)
    return gr.update(visible=False)

def export_scenes_to_pdf(state, project_name: Optional[str] = None, director_notes: Optional[str] = None) -> Optional[str]:
    """Export scenes to PDF as professional shot list. Optional header: project name + director notes. 3 scenes per page."""
    if not state or not state.get("scenes"):
        return None
    scenes = state["scenes"]
    video_name = Path(state.get("video_path", "project")).stem
    try:
        from fpdf import FPDF
    except ImportError:
        return None

    out_dir = BASE_DIR / "exports"
    out_dir.mkdir(parents=True, exist_ok=True)
    base_name = (project_name or video_name).strip() or video_name
    out_path = out_dir / f"{base_name}_shotlist_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"

    _cjk_font_path = None
    for p in [
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        Path("/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"),
        Path("C:/Windows/Fonts/msyh.ttf"),
        Path("C:/Windows/Fonts/msyhbd.ttf"),
        Path("C:/Windows/Fonts/simhei.ttf"),
        Path("C:/Windows/Fonts/msjh.ttc"),
    ]:
        if p.exists():
            _cjk_font_path = str(p)
            break

    _user_logo = (load_config().get("user_logo_path") or "").strip()
    _logo_path = Path(_user_logo).resolve() if _user_logo and Path(_user_logo).exists() else None

    class PDF(FPDF):
        def __init__(self, logo_path=None):
            super().__init__()
            self._logo_path = logo_path
            self._cjk_path = _cjk_font_path
            self.set_auto_page_break(auto=True, margin=15)
            self.set_margins(15, 15, 15)
            self.add_page()
            if _cjk_font_path:
                try:
                    self.add_font("CJK", "", _cjk_font_path)
                    self._use_cjk = True
                except Exception:
                    self._use_cjk = False
            else:
                self._use_cjk = False

        def header(self):
            if self._logo_path and os.path.exists(self._logo_path):
                try:
                    self.image(str(self._logo_path), 10, 8, w=33)
                except Exception:
                    pass

        def footer(self):
            self.set_y(-12)
            self.set_font("Helvetica", "I", 7)
            self.set_text_color(180, 180, 180)
            self.cell(0, 8, "Generated by StoryBreak", 0, 0, "R")

    pdf = PDF(logo_path=_logo_path)
    pdf.set_margins(15, 15, 15)
    font_name = "CJK" if pdf._use_cjk else "Helvetica"
    font_size = 9

    def _safe_text(t: str) -> str:
        if pdf._use_cjk:
            return t
        return (t or "").encode("ascii", "replace").decode("ascii")

    # --- Professional shot list header (first page only) ---
    header_h = 0
    if project_name or director_notes:
        pdf.set_font(font_name, "B", 14)
        pdf.set_xy(15, 15)
        title = _safe_text((project_name or video_name).strip() or "Scene Breakdown")
        pdf.cell(0, 8, title)
        y_cur = 26
        if director_notes and (director_notes := director_notes.strip()):
            pdf.set_font(font_name, "", 9)
            pdf.set_xy(15, y_cur)
            pdf.cell(0, 5, _safe_text("Director's Notes"))
            pdf.set_font(font_name, "", 8)
            pdf.set_xy(15, y_cur + 6)
            pdf.multi_cell(180, 4, _safe_text(director_notes[:500]))
            y_cur += 6 + max(4 * ((len(director_notes) // 45) + 1), 8)
        pdf.set_draw_color(180, 180, 180)
        pdf.line(15, y_cur + 2, 195, y_cur + 2)
        header_h = y_cur + 8

    block_h = (297 - 15 - 15 - header_h) / 3 if header_h else (297 - 30) / 3
    img_h = block_h * 0.65
    img_w = min(210 - 30, img_h * 16 / 9)
    y_base = 15 + header_h

    for i, s in enumerate(scenes):
        if i > 0 and i % 3 == 0:
            pdf.add_page()
            y_base = 15
            block_h = (297 - 30) / 3
            img_h = block_h * 0.65
            img_w = min(210 - 30, img_h * 16 / 9)
        y = y_base + (i % 3) * block_h
        if _is_file_path(getattr(s, "thumbnail_path", None)):
            try:
                pdf.image(s.thumbnail_path, 15, y, w=img_w, h=img_h)
            except Exception:
                pass
        txt_y = y + img_h + 3
        pdf.set_font(font_name, "", font_size)
        pdf.set_xy(15, txt_y)
        pdf.cell(0, 5, f"#{s.scene_number}   {format_time(s.start_time)} - {format_time(s.end_time)}")
        if s.tag:
            tag_short = (s.tag or "")[:50] + ("..." if len(s.tag or "") > 50 else "")
            pdf.set_xy(15, txt_y + 6)
            pdf.cell(0, 5, f"Tag: {_safe_text(tag_short)}")
        if s.annotation:
            anno = (s.annotation or "").replace("\n", " ")[:100] + ("..." if len(s.annotation or "") > 100 else "")
            pdf.set_xy(15, txt_y + 12)
            pdf.multi_cell(180, 5, _safe_text(anno))

    pdf.output(str(out_path))
    return str(out_path)


# --- UI logic ---

def _library_welcome_visibility(user_email: Optional[str] = None):
    """Return (welcome_screen visible, library_content visible) from current library."""
    n = len(lib_manager.load(user_email or "guest"))
    return gr.update(visible=(n == 0)), gr.update(visible=(n > 0))


def _open_latest_project_in_workstation(state=None):
    """New Task 完成後「Open in Workstation」：載入 Library 最新一筆（index 0）到 Workstation。"""
    _evt = type("SelectData", (), {"index": 0})()
    return handle_library_gallery_select(_evt, False, state)


def handle_library_gallery_select(evt: gr.SelectData, delete_mode: bool, state=None):
    """Handle Project Library gallery click: delete if Delete Mode, else load into Workstation."""
    user_email = (state.get("user", {}).get("email") if state else None) or "guest"
    w_vis, l_vis = _library_welcome_visibility(user_email)

    if delete_mode:
        if not lib_manager.delete_at_index(evt.index, user_email):
            return _gr_updates(WORKSTATION_VIEW_OUTPUTS)
        w_vis, l_vis = _library_welcome_visibility(user_email)
        keep = gr.update()
        return (
            gr.update(),
            gr.update(visible=True),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            keep, keep, gr.HTML(""), gr.Markdown("### No Selection"),
            0.0, 0.0, "", "",
            gr.update(visible=False),
            keep,
            gr.update(choices=["All"], value="All"),
            gr.update(value=""),
            gr.update(interactive=False),
            gr.update(interactive=False),
            w_vis,
            l_vis,
        )

    result = load_video_to_workstation(evt, state)
    result_list = list(result)
    result_list[-2] = w_vis
    result_list[-1] = l_vis
    return tuple(result_list)

def load_video_to_workstation(evt: gr.SelectData, state=None):
    prev_state = state or {}
    user = prev_state.get("user", {"email": "guest", "plan": "Pro", "credits": 9999})
    user_email = user.get("email") or "guest"
    library = lib_manager.load(user_email)
    if evt.index < 0 or evt.index >= len(library):
        return list(_gr_updates(WORKSTATION_VIEW_OUTPUTS))

    item = library[evt.index]
    video_path = item.get('path', '')

    scenes = []
    for s in item.get('scenes_data', []):
        if hasattr(SceneInfo, 'from_dict') and callable(getattr(SceneInfo, 'from_dict')):
            scene = SceneInfo.from_dict(s)
        else:
            scene = SceneInfo(
                scene_number=s.get('scene_number', 0),
                start_time=s.get('start_time', 0),
                end_time=s.get('end_time', 0),
                duration=s.get('duration', 0),
                start_frame=s.get('start_frame', 0),
                end_frame=s.get('end_frame', 0),
                thumbnail_path=s.get('thumbnail_path'),
                annotation=s.get('annotation'),
                tag=s.get('tag'),
                group_id=s.get('group_id'),
                movement=s.get('movement')
            )
            if s.get('video_clip_path'):
                scene.video_clip_path = s['video_clip_path']
        scenes.append(scene)

    # 合併 state，保留 user，不覆蓋
    state = {
        **prev_state,
        "user": user,
        "video_path": video_path,
        "scenes": scenes,
        "info": item,
        "library_index": evt.index,
        "current_scene_idx": 0,
    }

    gallery_data = _get_gallery_items_from_scenes(scenes)

    first_scene = scenes[0] if scenes else None
    if first_scene:
        clip = getattr(first_scene, 'video_clip_path', None)
        if clip and os.path.exists(clip):
            video_update = gr.update(value=clip)
        else:
            video_update = gr.update(value=video_path)
    else:
        video_update = gr.update(value=video_path)

    s = first_scene
    info_md = f"""
    <div style='display: flex; align-items: center; gap: 10px;'>
        <div style='background:#3B82F6; width:4px; height:40px; border-radius:2px;'></div>
        <div>
            <div style='font-size: 1.1em; font-weight: bold; color: white;'>{item.get('name')}</div>
            <div style='font-size: 0.8em; color: #94a3b8;'>{item.get('width')}x{item.get('height')} | {item.get('fps'):.2f} FPS | {format_time(item.get('duration', 0))}</div>
        </div>
    </div>
    """

    t_start = s.start_time if s else 0
    t_end = s.end_time if s else 0
    tag_val = s.tag if s else ""
    anno_val = s.annotation if s else ""
    scene_status = f"Scene #{s.scene_number} / {len(scenes)}" if s else "No Scenes"

    classify_gallery_update = gr.update(value=gallery_data)
    classify_choices = ["All"]
    tags = set()
    for sc in scenes:
        if sc.tag:
            for t in sc.tag.split(" | "): tags.add(t.strip())
    classify_choices += sorted(list(tags))
    filter_radio_update = gr.update(choices=classify_choices, value="All")
    n_scenes = len(scenes)
    btn_prev_upd = gr.update(interactive=False)
    btn_next_upd = gr.update(interactive=(n_scenes > 1))

    return (
        state,
        gr.update(visible=False),
        gr.update(visible=True),
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(visible=False),
        gallery_data,
        video_update,
        info_md,
        scene_status,
        t_start, t_end, tag_val, anno_val,
        gr.update(visible=False),
        classify_gallery_update,
        filter_radio_update,
        gr.update(value=""),
        btn_prev_upd,
        btn_next_upd,
        gr.update(),
        gr.update(),
    )

def run_classification_analysis(state, progress=gr.Progress()):
    if not state or not state.get('scenes'):
        return state, "No scenes to analyze", gr.update(), gr.update()

    scenes = state['scenes']
    video_path = state['video_path']

    try:
        scenes = detector.classifier.classify_scenes(scenes, video_path, progress_callback=lambda p, m: progress(p, desc=m))
    except AttributeError:
        return state, "Error: Backend classifier not found. Please update scene_detector.py", gr.update(), gr.update()

    state['scenes'] = scenes
    user_email = (state.get("user", {}).get("email") or "guest")
    lib_manager.add_or_update(video_path, state['info'], scenes, state['info'].get('thumbnail'), user_email=user_email)

    tags = set()
    for s in scenes:
        if s.tag:
            for sep in (" | ", "|", "、", "，"):
                for part in s.tag.split(sep):
                    t = part.strip()
                    if t and len(t) <= 30:
                        tags.add(t)

    choices = ["All"] + sorted(list(tags))
    gallery_data = _get_gallery_items_from_scenes(scenes)

    return state, f"Analysis Complete. Found {len(tags)} tag types.", gr.update(choices=choices, value="All"), gr.update(value=gallery_data)

def filter_classification_gallery(state, tag):
    if not state or not state.get('scenes'): return []

    if tag == "All":
        return _get_gallery_items_from_scenes(state['scenes'])

    filtered = [s for s in state['scenes'] if s.tag and tag in s.tag]
    return _get_gallery_items_from_scenes(filtered)


def _get_filtered_scenes(state, tag):
    """Return filtered scene list for current tag filter."""
    if not state or not state.get('scenes'):
        return []
    if tag == "All":
        return state['scenes']
    return [s for s in state['scenes'] if s.tag and tag in s.tag]


def select_classify_scene_for_edit(evt: gr.SelectData, state, current_filter):
    """On gallery select: show selected scene's tag for editing."""
    if not state or not state.get('scenes'):
        return state, "", -1, "No project loaded."
    filtered = _get_filtered_scenes(state, current_filter)
    if evt.index < 0 or evt.index >= len(filtered):
        return state, "", -1, ""
    scene = filtered[evt.index]
    idx = state['scenes'].index(scene)
    return state, scene.tag or "", idx, f"Editing Scene #{scene.scene_number}"


def save_classify_tag_edit(state, sel_idx, new_tag, current_filter):
    """Save edited tag to scene and persist to library."""
    if not state or not state.get('scenes') or sel_idx < 0 or sel_idx >= len(state['scenes']):
        return state, gr.update(), gr.update(), "⚠️ Invalid selection."
    scene = state['scenes'][sel_idx]
    scene.tag = (new_tag or "").strip()
    video_path = state['video_path']
    user_email = (state.get("user", {}).get("email") or "guest")
    lib_manager.add_or_update(video_path, state['info'], state['scenes'], state['info'].get('thumbnail'), user_email=user_email)

    tags = set()
    for s in state['scenes']:
        if s.tag:
            for sep in (" | ", "|", "、", "，"):
                for part in s.tag.split(sep):
                    t = part.strip()
                    if t and len(t) <= 30:
                        tags.add(t)
    choices = ["All"] + sorted(list(tags))

    gallery_data = filter_classification_gallery(state, current_filter or "All")
    return state, gr.update(value=gallery_data), gr.update(choices=choices), "✅ Tag saved."

def _update_ui_for_scene(state, idx):
    scenes = state['scenes']
    if idx < 0: idx = 0
    if idx >= len(scenes): idx = len(scenes) - 1

    state['current_scene_idx'] = idx
    s = scenes[idx]

    clip_path = getattr(s, 'video_clip_path', None)
    if clip_path and os.path.exists(clip_path):
        player_upd = gr.update(value=clip_path, autoplay=True)
    else:
        player_upd = gr.update(value=state['video_path'])

    header = f"Scene #{s.scene_number} / {len(scenes)}"
    prev_state = gr.update(interactive=(idx > 0))
    next_state = gr.update(interactive=(idx < len(scenes) - 1))

    return (state, player_upd, header, s.start_time, s.end_time, s.tag or "", s.annotation or "", prev_state, next_state)

def select_scene_from_sidebar(state, evt: gr.SelectData):
    if not state or not state['scenes']: return list(_gr_updates(SCENE_SELECT_OUTPUTS))
    return _update_ui_for_scene(state, evt.index)

def nav_prev_scene(state):
    if not state: return list(_gr_updates(SCENE_SELECT_OUTPUTS))
    idx = state.get('current_scene_idx', 0) - 1
    return _update_ui_for_scene(state, idx)

def nav_next_scene(state):
    if not state: return list(_gr_updates(SCENE_SELECT_OUTPUTS))
    idx = state.get('current_scene_idx', 0) + 1
    return _update_ui_for_scene(state, idx)

def update_scene_attributes(state, start, end, tag, annotation):
    if not state or 'current_scene_idx' not in state:
        return state, gr.update(), "⚠️ No scene selected."
    idx = state['current_scene_idx']
    s = state['scenes'][idx]
    s.start_time = float(start)
    s.end_time = float(end)
    s.duration = s.end_time - s.start_time
    s.tag = normalize_tag(tag)
    s.annotation = annotation

    user_email = (state.get("user", {}).get("email") or "guest")
    lib_manager.add_or_update(state['video_path'], state['info'], state['scenes'], state['info'].get('thumbnail'), user_email=user_email)
    return state, gr.update(value=_get_gallery_items_from_scenes(state['scenes'])), f"✅ Scene #{s.scene_number} updated."

def merge_current_scene(state, progress=gr.Progress()):
    idx = state.get('current_scene_idx', 0)
    scenes = state.get('scenes', [])
    if idx >= len(scenes) - 1:
        return state, gr.update(), "⚠️ Cannot merge last scene."

    a, b = scenes[idx], scenes[idx + 1]
    merged = SceneInfo(
        scene_number=idx + 1,
        start_time=a.start_time, end_time=b.end_time,
        start_frame=a.start_frame, end_frame=b.end_frame,
        duration=b.end_time - a.start_time,
        thumbnail_path=a.thumbnail_path,
        annotation=((a.annotation or "") + " " + (b.annotation or "")).strip(),
        video_clip_path=None, tag=a.tag or b.tag, group_id=getattr(a, "group_id", None),
    )
    new_scenes = scenes[:idx] + [merged] + scenes[idx + 2:]
    for i, s in enumerate(new_scenes): s.scene_number = i + 1
    state['scenes'] = new_scenes
    user_email = (state.get("user", {}).get("email") or "guest")
    lib_manager.add_or_update(state['video_path'], state['info'], new_scenes, state['info'].get('thumbnail'), user_email=user_email)
    return state, gr.update(value=_get_gallery_items_from_scenes(new_scenes)), "✅ Scenes merged."

def split_current_scene(state, progress=gr.Progress()):
    if not state or 'current_scene_idx' not in state:
        return state, gr.update(), "⚠️ No scene selected."
    idx = state.get('current_scene_idx', 0)
    scenes = state['scenes']
    s = scenes[idx]
    mid_time = (s.start_time + s.end_time) / 2
    mid_frame = int((s.start_frame + s.end_frame) / 2)
    s1 = SceneInfo(0, s.start_time, mid_time, s.start_frame, mid_frame, mid_time - s.start_time, None, s.annotation, None, s.tag)
    s2 = SceneInfo(0, mid_time, s.end_time, mid_frame, s.end_frame, s.end_time - mid_time, None, s.annotation, None, s.tag)
    new_scenes = scenes[:idx] + [s1, s2] + scenes[idx+1:]
    for i, sc in enumerate(new_scenes): sc.scene_number = i + 1

    progress(0.2, desc="Generating thumbnails...")
    if state.get('video_path'):
        s1.thumbnail_path = detector.extract_thumbnail_at_position(state['video_path'], s1, 0.5)
        s2.thumbnail_path = detector.extract_thumbnail_at_position(state['video_path'], s2, 0.5)

    state['scenes'] = new_scenes
    user_email = (state.get("user", {}).get("email") or "guest")
    lib_manager.add_or_update(state['video_path'], state['info'], new_scenes, state['info'].get('thumbnail'), user_email=user_email)
    return state, gr.update(value=_get_gallery_items_from_scenes(new_scenes)), "✅ Scene split."

def generate_ai_note_handler(state, progress=gr.Progress()):
    """Generate AI note for current scene using LLaVA-NeXT (Video-LLaVA)."""
    if not state or "scenes" not in state or "current_scene_idx" not in state:
        return "[Please select a scene first]"
    if generate_scene_note is None:
        return "[Please install: pip install transformers av]"
    scenes = state["scenes"]
    idx = state["current_scene_idx"]
    if idx < 0 or idx >= len(scenes):
        return "[Invalid scene index]"
    scene = scenes[idx]
    video_path = state.get("video_path", "")
    clip_path = getattr(scene, "video_clip_path", None)

    progress(0.1, desc="Loading LLaVA-NeXT Video...")
    if clip_path and os.path.exists(clip_path):
        note = generate_scene_note(video_path=clip_path)
    elif video_path and os.path.exists(video_path):
        note = generate_scene_note(
            video_path=video_path,
            start_sec=scene.start_time,
            end_sec=scene.end_time,
        )
    elif _is_file_path(getattr(scene, "thumbnail_path", None)):
        note = generate_scene_note(image_path=scene.thumbnail_path)
    else:
        return "[Video or thumbnail not found]"

    scene.annotation = note
    user_email = (state.get("user", {}).get("email") or "guest")
    lib_manager.add_or_update(video_path, state["info"], scenes, state["info"].get("thumbnail"), user_email=user_email)
    return note

def resnap_thumbnail_handler(state):
    if not state or 'current_scene_idx' not in state:
        return state, gr.update(), "⚠️ No scene selected."
    idx = state['current_scene_idx']
    s = state['scenes'][idx]
    new_thumb = detector.extract_thumbnail_at_position(state['video_path'], s, 0.5)
    if new_thumb:
        s.thumbnail_path = new_thumb
        user_email = (state.get("user", {}).get("email") or "guest")
        lib_manager.add_or_update(state['video_path'], state['info'], state['scenes'], state['info'].get('thumbnail'), user_email=user_email)
        return state, gr.update(value=_get_gallery_items_from_scenes(state['scenes'])), "✅ Thumbnail updated."
    return state, gr.update(), "⚠️ Thumbnail update failed."

def _resolve_video_path(video_file):
    """Resolve Gradio File (str/dict/file://) to absolute path."""
    if not video_file:
        return None
    path = video_file.get("path", video_file) if isinstance(video_file, dict) else video_file
    path = str(path).replace("file:///", "").replace("file://", "").strip()
    return str(Path(path).resolve()) if path and Path(path).exists() else None

def _log_ts():
    return datetime.now().strftime("%H:%M:%S")


def run_detection_pipeline(video_file, method, threshold, min_len, state=None, progress=gr.Progress()):
    if not video_file:
        return "[%s] Please upload a video." % _log_ts(), gr.update()
    video_path = _resolve_video_path(video_file)
    if not video_path:
        return "[%s] Could not resolve video path. Confirm the file was uploaded." % _log_ts(), gr.update()
    log_lines = []
    try:
        log_lines.append("[%s] Analyzing video..." % _log_ts())
        progress(0.1, desc="Analyzing Video...")
        info = get_video_info(video_path)
        log_lines.append("[%s] Detecting scenes (%s)..." % (_log_ts(), method.split(" ")[0] if method else "content"))
        progress(0.2, desc="Detecting Scenes...")
        method_key = "content"
        if "TransNet" in method: method_key = "transnet"
        elif "Adaptive" in method: method_key = "adaptive"
        elif "Threshold" in method: method_key = "threshold"

        scenes = detector.process_video(
            video_path, method=method_key, threshold=threshold, min_scene_len=min_len,
            progress_callback=lambda p, m: progress(0.2 + p * 0.5, desc=m)
        )
        user_plan = (state or {}).get("user", {}).get("plan", "Pro")
        # Pro 版本：不限制場景數
        if user_plan == "Free" and len(scenes) > 10:
            scenes = scenes[:10]
            log_lines.append("[%s] Free Plan: limited to 10 scenes. Upgrade to Pro for full analysis." % _log_ts())
        log_lines.append("[%s] Found %d scene(s)." % (_log_ts(), len(scenes)))

        if scenes:
            log_lines.append("[%s] Extracting clips..." % _log_ts())
            progress(0.7, desc="Extracting clips...")
            scenes = detector.extract_video_clips(
                video_path, scenes,
                progress_callback=lambda p, m: progress(0.7 + p * 0.2, desc=m)
            )
            log_lines.append("[%s] Clips extracted." % _log_ts())

        cover_path = detector.extract_cover(video_path) or (scenes[0].thumbnail_path if scenes else None)
        user_email = (state.get("user", {}).get("email") if state else None) or "guest"
        lib_manager.add_or_update(video_path, info, scenes, cover_path, user_email=user_email)
        log_lines.append("[%s] Done. Detected %d scenes. Open in Workstation to edit." % (_log_ts(), len(scenes)))
        return "\n".join(log_lines), gr.update(visible=True)
    except Exception as e:
        import traceback
        traceback.print_exc()
        log_lines.append("[%s] Error: %s" % (_log_ts(), str(e)))
        return "\n".join(log_lines), gr.update()

# --- CSS & Theme (Pro / DaVinci-style neutral grey) ---
def _get_pro_theme():
    Slate = getattr(gr.themes, "Slate", None)
    try:
        if Slate:
            base_theme = Slate(
                primary_hue="blue",
                secondary_hue="neutral",
                neutral_hue="neutral",
                font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "sans-serif"],
                font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "monospace"],
            )
        else:
            base_theme = gr.themes.Base()
    except (AttributeError, TypeError):
        base_theme = gr.themes.Base()

    overrides = dict(
        body_background_fill="#121212",
        body_text_color="#E5E5E5",
        block_background_fill="#1E1E1E",
        block_border_width="0px",
        block_border_color="#2A2A2A",
        block_label_text_color="#A3A3A3",
        block_title_text_color="#D4D4D4",
        input_background_fill="#262626",
        input_border_color="#404040",
        input_placeholder_color="#737373",
        button_primary_background_fill="#3B82F6",
        button_primary_background_fill_hover="#2563EB",
        button_primary_text_color="#FFFFFF",
        button_primary_border_color="#3B82F6",
        button_secondary_background_fill="#262626",
        button_secondary_background_fill_hover="#404040",
        button_secondary_text_color="#E5E5E5",
        button_secondary_border_color="#404040",
        block_radius="6px",
        container_radius="6px",
        input_radius="4px",
    )
    return base_theme.set(**overrides)

sci_fi_theme = _get_pro_theme()

js_drag_drop = """
(function() {
    function findSourceGallery() {
        return document.querySelector('[id*="asm_source_gallery"]');
    }
    function findTargetTimeline() {
        var zone = document.getElementById('asm_drop_zone');
        if (zone) return zone;
        return document.querySelector('[id*="asm_target_timeline"]');
    }
    function findDropInput() {
        var c = document.querySelector('[id*="asm_drop_val"]');
        return c ? (c.querySelector('input') || c.querySelector('textarea')) : null;
    }

    document.addEventListener('mouseover', function(e) {
        var src = findSourceGallery();
        if (src && e.target.closest('[id*="asm_source_gallery"]')) {
            var imgs = src.querySelectorAll('img');
            imgs.forEach(function(img, idx) {
                var p = img.parentElement;
                if (p && !p.getAttribute('draggable')) {
                    p.setAttribute('draggable', 'true');
                    p.dataset.asmIdx = idx;
                }
            });
        }
    });

    document.addEventListener('dragstart', function(e) {
        var src = findSourceGallery();
        if (!src) return;
        var el = e.target.closest('[draggable]');
        if (!el || !src.contains(el)) return;
        var imgs = Array.from(src.querySelectorAll('img'));
        var idx = -1;
        var targetImg = el.querySelector('img') || (el.tagName === 'IMG' ? el : null);
        if (targetImg) idx = imgs.indexOf(targetImg);
        else if (el.dataset.asmIdx !== undefined) idx = parseInt(el.dataset.asmIdx, 10);
        if (idx >= 0) {
            e.dataTransfer.setData('text/plain', String(idx));
            e.dataTransfer.effectAllowed = 'copy';
            el.style.opacity = '0.5';
        }
    });

    document.addEventListener('dragend', function(e) {
        if (e.target.style) e.target.style.opacity = '1';
    });

    document.addEventListener('dragover', function(e) {
        var tgt = findTargetTimeline();
        if (tgt && tgt.contains(e.target)) {
            e.preventDefault();
            e.dataTransfer.dropEffect = 'copy';
            tgt.style.border = '2px dashed #3B82F6';
            tgt.style.borderRadius = '8px';
        }
    });

    document.addEventListener('dragleave', function(e) {
        var tgt = findTargetTimeline();
        if (tgt && !tgt.contains(e.relatedTarget)) {
            tgt.style.border = 'none';
        }
    });

    document.addEventListener('drop', function(e) {
        var tgt = findTargetTimeline();
        if (tgt && tgt.contains(e.target)) {
            e.preventDefault();
            tgt.style.border = 'none';
            var index = e.dataTransfer.getData('text/plain');
            if (index !== '') {
                var input = findDropInput();
                if (input) {
                    input.value = index;
                    input.dispatchEvent(new Event('input', { bubbles: true }));
                    input.dispatchEvent(new Event('change', { bubbles: true }));
                }
            }
        }
    });

    document.addEventListener('keydown', function(e) {
        if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA' || e.target.isContentEditable) return;
        var buttons = Array.from(document.querySelectorAll('button'));
        function clickByText(t) { var b = buttons.find(function(el) { return el.textContent.trim() === t; }); if (b) b.click(); }
        if (e.key === 'Delete' || e.key === 'Backspace') {
            var removeBtn = buttons.find(function(el) { return el.textContent.trim() === 'Remove'; });
            if (removeBtn) { removeBtn.click(); e.preventDefault(); }
        }
        if (e.key === 'ArrowLeft') clickByText('Prev');
        if (e.key === 'ArrowRight') clickByText('Next');
    });
})();
"""

css_pro = """
/* 強制全頁深色（FastAPI 掛載時若 theme 未載入，此處仍能保證黑底） */
body, .gradio-container, .contain, main, [class*="gradio-container"] {
    background-color: #121212 !important;
    background: #121212 !important;
    color: #e5e5e5 !important;
}
.gradio-container .wrap, .gradio-container .block, .gradio-container > div {
    background: transparent !important;
    background-color: transparent !important;
    border-color: #333 !important;
}
/* Tabs、按鈕、輸入框等預設淺色強制改深色 */
.gradio-container .tabs, .gradio-container .tab-nav,
.gradio-container .form, .gradio-container .panel {
    background-color: #1e1e1e !important;
    border-color: #404040 !important;
    color: #e5e5e5 !important;
}
.gradio-container input, .gradio-container textarea, .gradio-container select {
    background-color: #262626 !important;
    border-color: #404040 !important;
    color: #fff !important;
}
.gradio-container label, .gradio-container .label-wrap, .gradio-container span {
    color: #d4d4d4 !important;
}

footer {display: none !important;}
body { font-family: 'Inter', sans-serif; background-color: #121212; }
.header-row { background-color: #121212 !important; padding: 10px 5px; border-bottom: 1px solid #262626; border-width: 0; }
.header-row .block, .header-row .wrap, .header-row > div { background: transparent !important; border: none !important; box-shadow: none !important; }
h1.app-title {
    font-family: 'JetBrains Mono', monospace;
    color: #3B82F6;
    font-weight: 700; font-size: 28px !important; letter-spacing: 1px;
    margin: 0 !important;
}

/* Subtle caption overlay: less obtrusive, more visible on hover */
.caption-label {
    background-color: rgba(0, 0, 0, 0.6) !important;
    backdrop-filter: blur(2px);
    border: none !important;
    padding: 4px 8px !important;
    font-size: 11px !important;
    color: #94a3b8 !important;
    font-weight: 500 !important;
    opacity: 0.85;
    transition: opacity 0.2s, background-color 0.2s;
}
.gallery-item:hover .caption-label {
    opacity: 1;
    background-color: rgba(0, 0, 0, 0.75) !important;
}
.caption-label span { color: #cbd5e1 !important; font-size: 10px !important; display: block; margin-top: 1px; }

/* --- Gallery items: force 16:9 movie ratio --- */
.gallery-item {
    border: 1px solid #404040 !important;
    background-color: #1E1E1E !important;
    transition: all 0.2s;
    position: relative !important;
    aspect-ratio: 16 / 9 !important;
    display: flex !important;
    align-items: center;
    justify-content: center;
    overflow: hidden;
    border-radius: 6px !important;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.5);
}

.gallery-item img {
    width: 100% !important;
    height: 100% !important;
    object-fit: cover !important;
}

.gallery-item:hover {
    border-color: #3B82F6 !important;
    transform: translateY(-2px);
    z-index: 10;
}

.project-gallery.delete-active .gallery-item {
    border-color: #EF4444 !important;
    cursor: url('data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24"><path fill="red" d="M19 6.41L17.59 5 12 10.59 6.41 5 5 6.41 10.59 12 5 17.59 6.41 19 12 13.41 17.59 19 19 17.59 13.41 12z"/></svg>') 12 12, pointer;
}
.project-gallery.delete-active .gallery-item::after {
    content: "\\2715";
    position: absolute;
    top: 6px;
    right: 6px;
    width: 28px;
    height: 28px;
    background: #EF4444;
    color: white;
    font-weight: bold;
    font-size: 18px;
    border-radius: 4px;
    display: flex;
    align-items: center;
    justify-content: center;
    box-shadow: 0 2px 4px rgba(0,0,0,0.5);
    z-index: 10;
    pointer-events: none;
    animation: popIn 0.2s cubic-bezier(0.175, 0.885, 0.32, 1.275);
}
@keyframes popIn {
    from { transform: scale(0); opacity: 0; }
    to { transform: scale(1); opacity: 1; }
}

button:active {
    transform: scale(0.98);
    filter: brightness(1.2);
}

.status-bar {
    background: #1E1E1E;
    border: none;
    border-left: 4px solid #3B82F6;
    padding: 12px;
    border-radius: 8px;
    margin-bottom: 16px;
    color: #E5E5E5;
}
button { font-weight: 600 !important; }

::-webkit-scrollbar { width: 8px; height: 8px; }
::-webkit-scrollbar-track { background: #121212; }
::-webkit-scrollbar-thumb { background: #404040; border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: #525252; }

#asm_target_timeline, [id*="asm_target_timeline"] {
    transition: all 0.3s ease;
}
#asm_target_timeline img, [id*="asm_target_timeline"] img {
    pointer-events: none;
}
#asm_source_gallery img, [id*="asm_source_gallery"] img {
    cursor: grab;
}
#asm_source_gallery img:active, [id*="asm_source_gallery"] img:active {
    cursor: grabbing;
}

/* Main container extends with content height */
.gradio-container {
    height: auto !important;
    min-height: 100vh;
}

#asm_target_timeline, [id*="asm_target_timeline"] {
    border-radius: 8px;
    background: #262626 !important;
    border: none !important;
}

/* Gallery scrollbar styling */
.grid-container {
    scrollbar-width: thin;
    scrollbar-color: #404040 #121212;
}

/* Terminal-style system log (DaVinci/Pro feel) */
.terminal-log, .terminal-log textarea, .terminal-log .container {
    background-color: #0d1117 !important;
    color: #3fb950 !important;
    font-family: 'JetBrains Mono', 'Consolas', monospace !important;
    font-size: 13px !important;
    border: 1px solid #21262d !important;
    border-radius: 6px !important;
}
.terminal-log .label-wrap {
    color: #8b949e !important;
}

/* Nav toolbar: single row, no wrap；去除白框與淺色背景 */
#nav-toolbar,
#nav-toolbar.wrap,
[id="nav-toolbar"] {
    flex-wrap: nowrap !important;
    overflow-x: auto !important;
    gap: 8px !important;
    align-items: center !important;
    background: transparent !important;
    background-color: transparent !important;
    border: none !important;
    box-shadow: none !important;
}
#nav-toolbar button {
    min-width: auto !important;
    padding-left: 12px !important;
    padding-right: 12px !important;
    white-space: nowrap !important;
}

/* --- 強制去除 Accordion 白框 --- */

/* 針對 Gradio 的 Accordion 容器 */
.gradio-accordion, .accordion {
    border: none !important;
    background: transparent !important;
    background-color: transparent !important;
    box-shadow: none !important;
    padding: 0 !important;
    margin: 0 !important;
}

/* 針對標題按鈕區域 (header) */
.gradio-accordion .label-wrap, .accordion .label-wrap {
    background: transparent !important;
    background-color: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 8px 0 !important;
}

/* 針對標題文字 */
.gradio-accordion span, .accordion span {
    color: #A3A3A3 !important;
    font-weight: normal !important;
}

/* 針對展開後的內容區域 (content) */
.gradio-accordion .accordion-content, .accordion .accordion-content {
    background: transparent !important;
    background-color: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
}

/* 移除預設的邊框線條 */
.gradio-accordion::before, .gradio-accordion::after,
.accordion::before, .accordion::after {
    display: none !important;
    border: none !important;
}

/* --- 強制 Gallery 項目保持固定比例與大小，不隨便放大 --- */

/* 針對 Library View 的 Gallery (我們有給它 elem_id="project-gallery") */
#project-gallery .grid-container,
#project-gallery .grid-wrap > div:first-child {
    display: grid !important;
    /* 強制每列固定 4 個，不管有沒有內容 */
    grid-template-columns: repeat(4, 1fr) !important;
    gap: 16px !important;
    align-content: start !important; /* 內容靠上對齊，不要垂直拉伸 */
}

/* 針對手機版或小螢幕的響應式調整 (選用) */
@media (max-width: 1000px) {
    #project-gallery .grid-container,
    #project-gallery .grid-wrap > div:first-child {
        grid-template-columns: repeat(3, 1fr) !important;
    }
}

/* 確保每個 Item 的大小限制 */
#project-gallery .gallery-item {
    width: 100% !important; /* 填滿格子就好，不要超過 */
    height: auto !important;
    aspect-ratio: 16 / 9 !important; /* 再次強調 16:9 */
    max-height: 250px !important;    /* 限制最大高度，防止單圖變巨無霸 */
}

/* 針對 Scene List (左側邊欄) */
#scene-bin .grid-container {
    grid-template-columns: repeat(2, 1fr) !important; /* 強制 2 列 */
}

/* --- 進度條脈衝流動特效 --- */

/* 針對進度條的填充部分 */
.progress-level .fill,
.progress-bar .fill,
.progress-level .progress-bar {
    /* 設定漸層背景：藍色 -> 青色 -> 藍色 */
    background: linear-gradient(90deg, #3B82F6 0%, #06b6d4 50%, #3B82F6 100%) !important;
    background-size: 200% 100% !important; /* 放大背景以進行移動 */
    animation: progress-pulse 1.5s linear infinite !important; /* 無限循環動畫 */
    box-shadow: 0 0 10px rgba(59, 130, 246, 0.6) !important; /* 添加發光暈影 */
}

/* 定義流動動畫關鍵影格 */
@keyframes progress-pulse {
    0% {
        background-position: 100% 0%;
    }
    100% {
        background-position: -100% 0%;
    }
}

/* --- Adobe-style Login System (修正：登入後消失 + 完美置中) --- */
#login_container {
    position: fixed !important;
    top: 0;
    left: 0;
    width: 100vw !important;
    height: 100vh !important;
    /* 移除 display 的 !important，讓 Gradio 的 visible=False 能生效 */
    display: flex; 
    align-items: center;
    justify-content: center;
    background-color: #0f0f0f !important;
    z-index: 99999 !important;
    margin: 0 !important;
    padding: 0 !important;
}

/* 強制修正：當 Gradio 隱藏元件時，確保它真的消失不擋路 */
#login_container[style*="display: none"],
#login_container[hidden] {
    display: none !important;
}

#login_card {
    background-color: #1E1E1E !important;
    border: 1px solid #333 !important;
    border-radius: 12px !important;
    box-shadow: 0 20px 50px rgba(0,0,0,0.8) !important;
    width: 850px !important;
    max-width: 90% !important;
    min-height: 480px !important;
    display: flex !important;
    flex-direction: row !important;
    /* 確保在 Fixed 容器內不偏移 */
    margin: 0 auto !important; 
}

/* 順便去除主程式上方的空白間距 */
.gradio-container {
    padding-top: 0 !important;
}

#main-app {
    margin-top: 0 !important;
    padding-top: 0 !important;
}

/* 左側品牌圖區域 */
#login_brand_side {
    background: linear-gradient(135deg, #1e1e1e 0%, #000000 100%);
    width: 45% !important;
    display: flex;
    align-items: center;
    justify-content: center;
    border-right: 1px solid #333;
    position: relative;
    overflow: hidden;
}

/* 右側表單區域（強制深色，避免 Gradio 預設白底） */
#login_form_side, #login_form_side .wrap, #login_form_side .block,
#login_form_side .tabs, #login_form_side .tab-nav, #login_form_side .form {
    background-color: #1e1e1e !important;
    background: #1e1e1e !important;
    color: #e5e5e5 !important;
    border-color: #333 !important;
}
#login_form_side {
    width: 55% !important;
    padding: 40px !important;
    display: flex !important;
    flex-direction: column;
    justify-content: center;
}
#login_form_side .tab-nav button, #login_form_side .tab-nav span {
    color: #94a3b8 !important;
}
#login_form_side .tab-nav .selected, #login_form_side .tab-nav [aria-selected="true"] {
    color: #3B82F6 !important;
}

/* 輸入框美化 */
#login_form_side input {
    background-color: #262626 !important;
    border: 1px solid #404040 !important;
    color: white !important;
    height: 45px !important;
}
"""

# --- UI build ---

def _header_logo_html(user_data=None):
    """user_data 可為 dict：至少含 plan；可選 name, picture, email。若為 str 則視為 plan（相容舊呼叫）。"""
    import html
    if user_data is None:
        user_data = {}
    if isinstance(user_data, str):
        user_data = {"plan": user_data}
    user_plan = user_data.get("plan", "Pro")
    user_name = (user_data.get("name") or user_data.get("email") or "").strip()
    picture_url = (user_data.get("picture") or "").strip()
    if user_name:
        user_name = html.escape(user_name, quote=True)
    if user_data.get("email"):
        email_display = html.escape((user_data.get("email") or ""), quote=True)
    else:
        email_display = ""

    import base64
    logo_path = ASSETS_DIR / "logo.png"
    badge_color = "#3B82F6" if user_plan in ("Pro", "Admin") else "#737373"
    plan_badge = f'<span style="background:{badge_color}; color:white; padding:2px 6px; border-radius:4px; font-size:10px; margin-left:8px; vertical-align: middle;">{user_plan.upper()}</span>'

    # 有 Google 大頭貼與姓名時：左側 logo + 右側頭像與姓名
    user_block = ""
    safe_picture = picture_url if picture_url.startswith("https://") else ""
    if safe_picture and user_name:
        user_block = f"""
        <div style="display: flex; align-items: center; gap: 10px; margin-left: auto;">
            <img src="{html.escape(safe_picture, quote=True)}" alt="" style="width: 36px; height: 36px; border-radius: 50%; object-fit: cover; border: 2px solid #404040;" />
            <div style="display: flex; flex-direction: column; align-items: flex-start;">
                <span style="font-size: 14px; font-weight: 600; color: #E5E5E5;">{user_name}</span>
                <span style="font-size: 11px; color: #737373;">{email_display}</span>
            </div>
        </div>
        """
    elif user_name:
        user_block = f"""
        <div style="display: flex; align-items: center; gap: 8px; margin-left: auto;">
            <span style="font-size: 14px; font-weight: 600; color: #E5E5E5;">{user_name}</span>
        </div>
        """

    if logo_path.exists():
        try:
            with open(logo_path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("utf-8")
            return f"""
    <div style="display: flex; align-items: center; gap: 12px; width: 100%; flex-wrap: wrap;">
        <div style="display: flex; align-items: center; gap: 12px;">
            <img src="data:image/png;base64,{b64}" alt="StoryBreak" style="height: 40px; width: auto; object-fit: contain;" />
            <div style="display: flex; flex-direction: column;">
                <div style="display: flex; align-items: center;">
                    <span style="font-family: 'JetBrains Mono', monospace; font-weight: 700; font-size: 20px; color: #E5E5E5;">StoryBreak</span>
                    {plan_badge}
                </div>
                <span style="font-family: Inter; font-size: 10px; color: #737373;">v2.1 Pro Workstation</span>
            </div>
        </div>
        {user_block}
    </div>
    <p style="margin: 2px 0 0 0; font-size: 11px; color: #737373;">The Ultimate Video Reference Breakdown Tool</p>
    """
        except Exception:
            pass
    return f"""
    <div style="display: flex; align-items: center; gap: 12px; width: 100%;">
        <div style="display: flex; align-items: center; gap: 10px;">
            <span style="font-family: 'JetBrains Mono', monospace; color: #3B82F6; font-weight: 700; font-size: 24px;">StoryBreak</span>
            {plan_badge}
        </div>
        {user_block}
    </div>
    <p style="margin: 0; font-size: 10px; color: #737373;">v2.1 Pro Workstation · The Ultimate Video Breakdown Tool</p>
    """


# --- Auth / login ---

def get_session_user(request) -> Optional[dict]:
    """從 FastAPI/Gradio request 讀取 session 中的 user（需 main.py SessionMiddleware）。抓不到時不拋錯，回傳 None。"""
    if not request:
        return None
    try:
        req = getattr(request, "request", request)
        session = getattr(req, "session", None)
        if session is None:
            return None
        return session.get("user")
    except Exception:
        return None


def authenticate(username: str, password: str):
    """
    [已棄用] 此函式未接到 UI，登入請用 UserManager.login()。
    License Key / Admin 邏輯已整合進 UserManager.login()，權威來源為 SQLite users 表。
    保留僅供相容或往後整合參考。
    """
    env_user = os.getenv("STORYBREAK_USER")
    env_pass = os.getenv("STORYBREAK_PASS")
    license_key = os.getenv("STORYBREAK_LICENSE_KEY")

    # 1) 正式帳密（來自環境變數）
    if env_user and env_pass:
        if username == env_user and password == env_pass:
            return True, ""

    # 2) License Key 登入（只檢查密鑰）
    if license_key:
        if username.strip() and password == license_key:
            return True, ""

    # 3) 開發預設帳密（環境變數未設定時的後備）
    if not (env_user or env_pass or license_key):
        if username == "admin" and password == "admin":
            return True, ""
        if username == "user" and password == "1234":
            return True, ""

    return False, "Invalid ID / Password or License Key. Please try again."


# 注入到 <head> 的關鍵深色樣式（確保 FastAPI 掛載時也能套用）
_critical_dark_head = """
<style>
body, .gradio-container, .contain, main { background: #121212 !important; color: #e5e5e5 !important; }
#login_container { background: #0f0f0f !important; }
#login_card, #login_form_side, #login_brand_side { background: #1e1e1e !important; color: #e5e5e5 !important; }
</style>
"""

def create_ui():
    # 必須在這裡直接套用 theme, css 和 js，FastAPI 掛載時才吃得到樣式！
    with gr.Blocks(
        title="StoryBreak Pro",
        theme=sci_fi_theme,
        css=css_pro,
        js=js_drag_drop,
        head=_critical_dark_head,
    ) as app:

        state = gr.State({"user": {"email": "guest", "plan": "Pro", "credits": 9999}})
        cfg = load_config()

        # --- 1. Login / Register Overlay ---
        with gr.Group(visible=True, elem_id="login_container") as login_view:
            with gr.Row(elem_id="login_card", equal_height=True):
                with gr.Column(elem_id="login_brand_side", scale=1):
                    import base64
                    logo_path = ASSETS_DIR / "logo.png"
                    if logo_path.exists():
                        try:
                            with open(logo_path, "rb") as f:
                                b64 = base64.b64encode(f.read()).decode("utf-8")
                            gr.HTML(f"""
                            <div style="width:100%; height:100%; display:flex; flex-direction:column; align-items:center; justify-content:center; padding:20px;">
                                <img src="data:image/png;base64,{b64}" style="width: 60%; opacity: 0.9; margin-bottom: 20px;">
                                <h2 style="color:white; margin:0;">StoryBreak</h2>
                                <p style="color:#9ca3af; font-size:14px;">Professional Video Analysis</p>
                                <div style="margin-top: 30px; text-align: left; color: #d1d5db; font-size: 13px;">
                                    <p>✓ AI Scene Detection</p>
                                    <p>✓ PDF Shot List Export</p>
                                    <p>✓ Collaborative Cloud (Pro)</p>
                                </div>
                            </div>
                            """)
                        except Exception:
                            gr.Markdown("## StoryBreak\nProfessional Video Analysis")
                    else:
                        gr.Markdown("## StoryBreak\nProfessional Video Analysis")

                with gr.Column(elem_id="login_form_side", scale=1):
                    with gr.Tabs():
                        with gr.Tab("Google Sign In"):
                            gr.Markdown("### 歡迎來到 StoryBreak Pro")
                            gr.HTML("""
    <a href="/login/google" style="
        display: flex; align-items: center; justify-content: center;
        background-color: white; color: #333; font-weight: bold; font-size: 16px;
        padding: 12px; border-radius: 6px; text-decoration: none; width: 100%;
        margin-top: 20px; border: 1px solid #ddd; transition: 0.2s;
    " onmouseover="this.style.backgroundColor='#f8f9fa'" onmouseout="this.style.backgroundColor='white'">
        <img src="https://upload.wikimedia.org/wikipedia/commons/c/c1/Google_%22G%22_logo.svg" style="width: 20px; margin-right: 12px;">
        使用 Google 帳戶繼續
    </a>
    """)
                        with gr.Tab("Sign In"):
                            login_email = gr.Textbox(label="Email", placeholder="name@company.com")
                            login_pass = gr.Textbox(label="Password / License Key", type="password")
                            login_btn = gr.Button("Sign In", variant="primary", size="lg")
                        with gr.Tab("Create Account"):
                            reg_email = gr.Textbox(label="Email", placeholder="name@company.com")
                            reg_pass = gr.Textbox(label="Password", type="password")
                            reg_btn = gr.Button("Create Free Account", variant="secondary", size="lg")
                    auth_msg = gr.Markdown("", visible=True)
                    gr.HTML("<div style='margin-top: 20px; color: #555; font-size: 11px; text-align: center;'>By continuing, you agree to StoryBreak Terms of Service.</div>")

        # --- 2. Main Application ---
        with gr.Group(visible=False, elem_id="main-app") as main_app:

            with gr.Row(elem_classes="header-row", variant="compact"):
                with gr.Column(scale=1, min_width=250):
                    header_html = gr.HTML(_header_logo_html("Free"))
                with gr.Column(scale=4, min_width=600):
                    with gr.Row(elem_id="nav-toolbar", variant="compact"):
                        nav_upgrade = gr.Button("⚡ Upgrade Pro", size="sm", variant="primary")
                        nav_about = gr.Button("About", size="sm", variant="secondary")
                        nav_lib = gr.Button("Projects", size="sm", variant="secondary")
                        nav_assembly = gr.Button("Assembly", size="sm", variant="secondary")
                        nav_classify = gr.Button("AI Classify", size="sm", variant="secondary")
                        nav_detect = gr.Button("New Task", size="sm", variant="secondary")
                        nav_logout = gr.Button("Logout", size="sm", variant="stop")
                    with gr.Group(visible=False) as about_panel:
                        gr.Markdown("""
**StoryBreak Workstation** v2.1 Pro

The Ultimate Video Reference Breakdown Tool — for editors, directors, and reference hunters.

- **Engine:** PySceneDetect · TransNet V2 · LLaVA-NeXT (optional)
- **License:** Pro Commercial License  
- **© 2026** All Rights Reserved.

*Export path and PDF options are available when exporting from a project.*
                    """)

            with gr.Group(visible=False) as upgrade_modal:
                gr.Markdown("### ⚡ Upgrade to StoryBreak Pro")
                gr.Markdown("Unlock unlimited exports, AI classification, and remove watermarks.")
                with gr.Row():
                    upgrade_key = gr.Textbox(label="License Key", placeholder="Paste your key here...", scale=3)
                    upgrade_confirm_btn = gr.Button("Activate Pro", variant="primary", scale=1)
                upgrade_msg = gr.Markdown("")
                close_upgrade_btn = gr.Button("Close", variant="secondary")

            with gr.Group(visible=True, elem_id="library-view") as library_view:
                with gr.Group(visible=True) as welcome_screen:
                    gr.HTML("""
                    <div style="text-align: center; padding: 60px 20px; color: #94a3b8;">
                        <div style="font-size: 80px; margin-bottom: 20px;">🎬</div>
                        <h2 style="color: #fff; font-size: 24px; margin-bottom: 10px;">Welcome to StoryBreak</h2>
                        <p style="font-size: 16px; max-width: 500px; margin: 0 auto 30px auto; line-height: 1.5;">
                            The professional tool for video reference breakdown.<br>
                            Start by analyzing your first video to extract scenes automatically.
                        </p>
                    </div>
                    """)
                    big_start_btn = gr.Button(" Start Your First Analysis", variant="primary", size="lg")
                with gr.Group(visible=False) as library_content:
                    with gr.Row():
                        gr.Markdown("### Project Library")
                        refresh_lib_btn = gr.Button("Refresh", size="sm")
                        delete_mode = gr.Checkbox(label="Delete Mode", value=False)
                    with gr.Column(elem_id="project-gallery", elem_classes=["project-gallery"]):
                        lib_gallery = gr.Gallery(
                            value=get_library_gallery_data,
                            columns=4, height=800,
                            allow_preview=False, label="Recent Projects", object_fit="cover"
                        )

            with gr.Group(visible=False, elem_id="workstation-view") as workstation_view:
                with gr.Row(elem_classes="status-bar"):
                    video_info_md = gr.HTML("Loading...")
                    export_pdf_btn = gr.Button("Export PDF", size="sm", variant="secondary")
                    pdf_download = gr.File(label="Download PDF", visible=False)
                    back_btn = gr.Button("Close Project", size="sm", variant="stop")
                with gr.Accordion("Shot list info (shown on PDF cover)", open=False):
                    with gr.Row():
                        pdf_project_name = gr.Textbox(
                            label="Project name",
                            placeholder="e.g. film / campaign name, shown on PDF cover",
                            scale=2
                        )
                        pdf_director_notes = gr.Textbox(
                            label="Director's notes",
                            placeholder="e.g. shooting focus, notes, reference style…",
                            lines=2,
                            scale=3
                        )

                with gr.Row():
                    with gr.Column(scale=1, min_width=280, elem_id="scene-bin"):
                        gr.Markdown("### Scenes")
                        scene_gallery = gr.Gallery(
                            columns=2, height=700, allow_preview=False, label="Scene List", interactive=True, show_label=False
                        )

                    with gr.Column(scale=3):
                        gr.Markdown("### Viewport")
                        with gr.Group(elem_classes="video-player-frame"):
                            main_player = gr.Video(label="Preview", height=500, autoplay=True, show_label=False)
                        with gr.Row():
                            btn_prev = gr.Button("Prev", variant="secondary")
                            btn_next = gr.Button("Next", variant="secondary")
                        with gr.Row():
                            merge_btn = gr.Button("Merge Next", variant="secondary")
                            split_btn = gr.Button("Split Here", variant="secondary")
                            snapshot_btn = gr.Button("Re-Snap", variant="secondary")

                    with gr.Column(scale=1, min_width=320):
                        gr.Markdown("### Inspector")
                        with gr.Tabs():
                            with gr.Tab("Properties"):
                                scene_header_md = gr.Markdown("### No Selection")
                                with gr.Row():
                                    inp_start = gr.Number(label="Start (s)", precision=2)
                                    inp_end = gr.Number(label="End (s)", precision=2)
                                inp_tag = gr.Dropdown(choices=cfg.get("custom_tags", DEFAULT_CONFIG["custom_tags"]), label="Tag", allow_custom_value=True)
                                save_attr_btn = gr.Button("Update Scene", variant="primary")
                                status_msg = gr.Markdown("", visible=True)
                            with gr.Tab("AI Notes"):
                                inp_annotation = gr.TextArea(label="Description", lines=8)
                                ai_gen_btn = gr.Button("Generate AI Note", size="sm")

            with gr.Group(visible=False) as assembly_view:
                with gr.Row(elem_classes="status-bar"):
                    with gr.Column(scale=4):
                        gr.Markdown("### Pro Sequence Builder")
                        gr.Markdown("**Tip:** Drag to add scenes from the left **Source Scenes** into the Timeline. Use **Move Prev / Move Next** to reorder.")
                    with gr.Column(scale=2):
                        with gr.Accordion("Shot list info (PDF)", open=False):
                            asm_pdf_project_name = gr.Textbox(
                                label="Project name",
                                placeholder="e.g. film / campaign name",
                                max_lines=1
                            )
                            asm_pdf_director_notes = gr.Textbox(
                                label="Director's notes",
                                placeholder="e.g. shooting focus, notes…",
                                lines=2
                            )
                        asm_thumb_scale = gr.Slider(50, 150, value=100, step=10, label="Thumbnail scale %", min_width=120)
                        with gr.Row():
                            assembly_back_btn = gr.Button("Back", variant="secondary", size="sm")
                            asm_export_edl_btn = gr.Button("Export EDL", variant="primary", size="sm")
                            asm_export_btn = gr.Button("Export PDF", variant="primary", size="sm")
                            asm_export_video_btn = gr.Button("Export Video", variant="primary", size="sm")
                        assembly_edl_download = gr.File(label="Download EDL", visible=False)
                        assembly_pdf_download = gr.File(label="Download PDF", visible=False)
                        assembly_video_download = gr.File(label="Download Video", visible=False)

                with gr.Row(equal_height=False):
                    # --- Left: Source + Upload (Input) ---
                    with gr.Column(scale=2):
                        with gr.Group():
                            with gr.Row():
                                gr.Markdown("#### Source Scenes")
                                assembly_refresh_btn = gr.Button("Refresh", size="sm", variant="secondary")
                            assembly_gallery = gr.Gallery(
                                value=get_all_scenes_gallery_data,
                                columns=4, height=600,
                                allow_preview=True, label="All Scenes", object_fit="cover", elem_id="asm_source_gallery"
                            )

                        with gr.Accordion("Upload External Media", open=False):
                            assembly_upload = gr.Image(label="Drop image to add", type="filepath")
                            asm_upload_btn = gr.Button("Add to Assembly", size="sm", variant="primary")

                    # --- Right: Assembly Timeline (Output) ---
                    # Drop target container: always visible so JS can detect dragover/drop even when timeline gallery is hidden
                    with gr.Column(scale=3, elem_id="asm_drop_zone"):
                        assembly_empty_state = gr.HTML(
                            value="""<div style="pointer-events: none; display: flex; flex-direction: column; align-items: center; justify-content: center; min-height: 400px; padding: 48px; text-align: center; background: rgba(15,23,42,0.5); border: 2px dashed #334155; border-radius: 12px;">
                                <div style="font-size: 64px; margin-bottom: 16px;">🎬</div>
                                <h3 style="color: #E2E8F0; margin: 0 0 8px 0;">Timeline is empty</h3>
                                <p style="color: #94a3b8; margin: 0 0 24px 0; max-width: 360px;">Drag to add scenes from the left. Use Move Prev / Move Next to reorder.</p>
                            </div>""",
                            visible=True
                        )
                        with gr.Group(visible=False) as assembly_timeline_group:
                            gr.Markdown("#### My Assembly Timeline (30 Slots)")
                            assembly_my_gallery = gr.Gallery(
                                value=_assembly_to_gallery([None] * NUM_ASSEMBLY_SLOTS),
                                columns=3, height=750,
                                allow_preview=True, label="Timeline", object_fit="cover", interactive=True,
                                elem_id="asm_target_timeline"
                            )
                            assembly_status = gr.Markdown("*Drag to add · Move Prev/Next to reorder*", visible=True)
                            with gr.Row():
                                asm_move_up_btn = gr.Button("Move Prev", size="sm")
                                asm_move_down_btn = gr.Button("Move Next", size="sm")
                                asm_remove_btn = gr.Button("Remove", size="sm", variant="stop")
                                asm_clear_btn = gr.Button("Clear All", size="sm", variant="stop")

                drop_trigger_idx = gr.Textbox(value="-1", visible=False, elem_id="asm_drop_val")
                assembly_all_meta = gr.State([])
                assembly_my = gr.State([None] * NUM_ASSEMBLY_SLOTS)
                assembly_sel_idx = gr.State(-1)

            with gr.Group(visible=False) as classify_view:
                gr.Markdown("### AI Scene Classification")

                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("#### 1. Analyze")
                        gr.Markdown("Run AI to detect shot types (Close-up, Wide) and camera movement.")
                        cls_run_btn = gr.Button("Run AI Analysis", variant="primary")
                        cls_status = gr.Textbox(label="Status", interactive=False)
                        gr.Markdown("#### 3. Edit Tag")
                        gr.Markdown("Select a scene below, then edit its tag (e.g. WS → MS).")
                        cls_edit_tag = gr.Textbox(label="Tag", placeholder="e.g. WS, MS, CU, ECU | Static, Pan Right", lines=1)
                        cls_edit_btn = gr.Button("Save Tag", variant="primary", size="sm")
                        cls_edit_status = gr.Markdown("", visible=True)

                    with gr.Column(scale=3):
                        gr.Markdown("#### 2. Filter Results")
                        cls_filter = gr.Radio(["All"], label="Show Scenes By Tag", value="All", interactive=True)

                cls_gallery = gr.Gallery(label="Filtered Scenes", columns=5, height=600, allow_preview=True, elem_id="classify-gallery")
                cls_sel_idx = gr.State(-1)
                cls_back_btn = gr.Button("Back to Project", variant="secondary")

            with gr.Group(visible=False) as detect_view:
                gr.Markdown("## New Detection Task")
                with gr.Row():
                    with gr.Column(scale=1):
                        upload_vid = gr.File(label="Source Video", file_types=["video"])
                    with gr.Column(scale=1):
                        gr.Markdown("### Settings")
                        detect_method = gr.Radio(["TransNet V2 (AI SOTA)", "Content (Standard)", "Adaptive (Pro)", "Threshold (Flash)"], value="TransNet V2 (AI SOTA)", label="Algorithm")
                        with gr.Row():
                            detect_threshold = gr.Slider(0.5, 50, value=3.0, label="Sensitivity")
                            min_scene_len = gr.Slider(5, 120, value=15, label="Min Frames")
                        gr.Markdown("---")
                        start_detect_btn = gr.Button("Start Analysis", variant="primary", size="lg")
                        detect_status = gr.Code(
                            label="System Log",
                            language="shell",
                            interactive=False,
                            lines=8,
                            value="[Ready] Upload a video and click Start Analysis.",
                            elem_classes=["terminal-log"]
                        )
                        goto_workstation_btn = gr.Button("Open in Workstation", visible=False, variant="primary")

        views = [library_view, workstation_view, detect_view, classify_view, assembly_view]
        def show_view(idx): return [gr.update(visible=(i==idx)) for i in range(len(views))]

        def _show_library_view(state=None):
            user_email = (state.get("user", {}).get("email") if state else None) or "guest"
            w_vis, l_vis = _library_welcome_visibility(user_email)
            return show_view(0) + [w_vis, l_vis]

        def _assembly_enter(state=None):
            user_email = (state.get("user", {}).get("email") if state else None) or "guest"
            items, meta = get_all_scenes_flat(user_email)
            return show_view(4) + [items, meta]

        # --- Auth: Register / Login / Upgrade / Logout ---

        def handle_register(email, pwd):
            if not email or "@" not in str(email):
                return "⚠️ Invalid email address."
            if not pwd:
                return "⚠️ Password cannot be empty."
            success, msg = user_manager.register(str(email).strip(), pwd)
            return msg

        def handle_login(email, pwd):
            success, user_data = user_manager.login(str(email).strip() if email else "", pwd or "")
            if success:
                new_header = _header_logo_html(user_data)
                w_vis, l_vis = _library_welcome_visibility(user_data.get("email", "guest"))
                new_state = {"user": user_data}
                return (
                    gr.update(visible=False),
                    gr.update(visible=True),
                    "",
                    new_state,
                    new_header,
                    w_vis,
                    l_vis,
                )
            return (
                gr.update(visible=True),
                gr.update(visible=False),
                f"⚠️ {user_data}",
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
            )

        def handle_upgrade(current_state, key):
            if not current_state or "user" not in current_state:
                return "⚠️ Please login first.", gr.update(), gr.update(visible=True), gr.update()
            email = current_state["user"].get("email")
            if not email or email == "guest":
                return "⚠️ Please login first.", gr.update(), gr.update(visible=True), gr.update()
            success, msg = user_manager.upgrade_to_pro(email, (key or "").strip())
            if success:
                new_user = {**current_state["user"], "plan": "Pro", "credits": 9999}
                new_state = {**current_state, "user": new_user}
                new_header = _header_logo_html(new_user)
                return f"✅ {msg}", new_header, gr.update(visible=False), new_state
            return f"❌ {msg}", gr.update(), gr.update(visible=True), gr.update()

        def logout_action():
            return gr.update(visible=True), gr.update(visible=False), ""

        reg_btn.click(handle_register, inputs=[reg_email, reg_pass], outputs=[auth_msg])
        login_btn.click(
            handle_login,
            inputs=[login_email, login_pass],
            outputs=[login_view, main_app, auth_msg, state, header_html, welcome_screen, library_content],
        )
        nav_upgrade.click(lambda: gr.update(visible=True), outputs=[upgrade_modal])
        close_upgrade_btn.click(lambda: gr.update(visible=False), outputs=[upgrade_modal])
        upgrade_confirm_btn.click(
            handle_upgrade,
            inputs=[state, upgrade_key],
            outputs=[upgrade_msg, header_html, upgrade_modal, state],
        )
        nav_logout.click(None, js="() => { window.location.href = '/logout_route'; }")

        nav_lib.click(_show_library_view, inputs=[state], outputs=views + [welcome_screen, library_content])
        big_start_btn.click(lambda: show_view(2), outputs=views)
        nav_assembly.click(_assembly_enter, inputs=[state], outputs=views + [assembly_gallery, assembly_all_meta])
        nav_detect.click(lambda: show_view(2), outputs=views)
        nav_classify.click(lambda: show_view(3), outputs=views)

        # -----------------------------
        # Wiring / Events
        # -----------------------------

        # --- Library view ---
        refresh_lib_btn.click(
            get_library_gallery_data_and_visibility,
            inputs=[state],
            outputs=[lib_gallery, welcome_screen, library_content],
        )

        LIB_SELECT_OUTPUTS = [
            state,
            library_view, workstation_view, detect_view, classify_view, assembly_view,
            scene_gallery, main_player, video_info_md, scene_header_md,
            inp_start, inp_end, inp_tag, inp_annotation,
            pdf_download,
            cls_gallery, cls_filter,
            status_msg, btn_prev, btn_next,
            welcome_screen, library_content,
        ]
        assert len(LIB_SELECT_OUTPUTS) == WORKSTATION_VIEW_OUTPUTS, "LIB_SELECT_OUTPUTS length must match WORKSTATION_VIEW_OUTPUTS"

        lib_gallery.select(
            handle_library_gallery_select,
            inputs=[delete_mode, state],
            outputs=LIB_SELECT_OUTPUTS,
        )

        delete_mode.change(fn=None, js="(checked) => { const el = document.getElementById('project-gallery'); if(el) el.classList.toggle('delete-active', checked); }")

        # --- Workstation view (scene list / nav / edit) ---
        SCENE_OUTPUTS = [state, main_player, scene_header_md, inp_start, inp_end, inp_tag, inp_annotation, btn_prev, btn_next]
        scene_gallery.select(select_scene_from_sidebar, inputs=[state], outputs=SCENE_OUTPUTS)
        btn_prev.click(nav_prev_scene, inputs=[state], outputs=SCENE_OUTPUTS)
        btn_next.click(nav_next_scene, inputs=[state], outputs=SCENE_OUTPUTS)
        save_attr_btn.click(
            update_scene_attributes,
            inputs=[state, inp_start, inp_end, inp_tag, inp_annotation],
            outputs=[state, scene_gallery, status_msg],
        )
        merge_btn.click(merge_current_scene, inputs=[state], outputs=[state, scene_gallery, status_msg])
        split_btn.click(split_current_scene, inputs=[state], outputs=[state, scene_gallery, status_msg])
        snapshot_btn.click(resnap_thumbnail_handler, inputs=[state], outputs=[state, scene_gallery, status_msg])
        ai_gen_btn.click(generate_ai_note_handler, inputs=[state], outputs=[inp_annotation])
        export_pdf_btn.click(
            export_pdf_handler,
            inputs=[state, pdf_project_name, pdf_director_notes],
            outputs=[pdf_download],
        )
        back_btn.click(_show_library_view, inputs=[state], outputs=views + [welcome_screen, library_content])

        # --- Detect view ---
        def _threshold_for_method(m):
            if "TransNet" in m: return 30.0
            if "Adaptive" in m: return 3.0
            return 30.0
        detect_method.change(lambda m: gr.update(value=_threshold_for_method(m)), inputs=[detect_method], outputs=[detect_threshold])
        start_detect_btn.click(
            run_detection_pipeline,
            inputs=[upload_vid, detect_method, detect_threshold, min_scene_len, state],
            outputs=[detect_status, goto_workstation_btn],
        )
        goto_workstation_btn.click(
            _open_latest_project_in_workstation,
            inputs=[state],
            outputs=LIB_SELECT_OUTPUTS,
        )

        # --- Classify view ---
        cls_run_btn.click(
            run_classification_analysis,
            inputs=[state],
            outputs=[state, cls_status, cls_filter, cls_gallery],
        )
        cls_filter.change(
            filter_classification_gallery,
            inputs=[state, cls_filter],
            outputs=[cls_gallery],
        )
        cls_gallery.select(
            select_classify_scene_for_edit,
            inputs=[state, cls_filter],
            outputs=[state, cls_edit_tag, cls_sel_idx, cls_edit_status],
        )
        cls_edit_btn.click(
            save_classify_tag_edit,
            inputs=[state, cls_sel_idx, cls_edit_tag, cls_filter],
            outputs=[state, cls_gallery, cls_filter, cls_edit_status],
        )
        cls_back_btn.click(lambda: show_view(1), outputs=views)

        # -----------------------------
        # Assembly view (drag/drop + buttons)
        # -----------------------------
        assembly_refresh_btn.click(assembly_refresh_fn, inputs=[state], outputs=[assembly_gallery, assembly_all_meta])
        assembly_gallery.select(
            assembly_add_handler,
            inputs=[assembly_all_meta, assembly_my],
            outputs=[assembly_my, assembly_my_gallery, assembly_empty_state, assembly_timeline_group],
        )
        drop_trigger_idx.change(
            assembly_add_handler_from_drop,
            inputs=[drop_trigger_idx, assembly_all_meta, assembly_my],
            outputs=[assembly_my, assembly_my_gallery, drop_trigger_idx, assembly_empty_state, assembly_timeline_group],
        )
        asm_upload_btn.click(
            assembly_upload_handler,
            inputs=[assembly_my, assembly_upload],
            outputs=[assembly_my, assembly_my_gallery, assembly_upload, assembly_empty_state, assembly_timeline_group],
        )
        assembly_my_gallery.select(
            assembly_select_handler,
            inputs=[assembly_my],
            outputs=[assembly_sel_idx, assembly_status],
        )
        asm_remove_btn.click(
            assembly_remove_handler,
            inputs=[assembly_my, assembly_sel_idx],
            outputs=[assembly_my, assembly_my_gallery, assembly_sel_idx, assembly_status, assembly_empty_state, assembly_timeline_group],
        )
        asm_move_up_btn.click(
            assembly_move_up_handler,
            inputs=[assembly_my, assembly_sel_idx],
            outputs=[assembly_my, assembly_my_gallery, assembly_sel_idx, assembly_status, assembly_empty_state, assembly_timeline_group],
        )
        asm_move_down_btn.click(
            assembly_move_down_handler,
            inputs=[assembly_my, assembly_sel_idx],
            outputs=[assembly_my, assembly_my_gallery, assembly_sel_idx, assembly_status, assembly_empty_state, assembly_timeline_group],
        )
        asm_clear_btn.click(
            assembly_clear_handler,
            outputs=[assembly_my, assembly_my_gallery, assembly_sel_idx, assembly_status, assembly_empty_state, assembly_timeline_group],
        )
        asm_export_btn.click(
            assembly_export_handler,
            inputs=[assembly_my, asm_pdf_project_name, asm_pdf_director_notes],
            outputs=[assembly_pdf_download, assembly_status],
        )
        asm_export_edl_btn.click(
            edl_export_handler,
            inputs=[assembly_my, asm_pdf_project_name],
            outputs=[assembly_edl_download, assembly_status],
        )
        def asm_export_video_handler(my_asm, progress=gr.Progress()):
            path, msg = export_assembly_video(my_asm, progress=progress)
            if path:
                return gr.update(value=path, visible=True), msg
            return gr.update(visible=False), f"⚠️ {msg}"
        asm_export_video_btn.click(
            asm_export_video_handler,
            inputs=[assembly_my],
            outputs=[assembly_video_download, assembly_status],
        )
        assembly_back_btn.click(lambda: show_view(1), outputs=views)

        nav_about.click(lambda: gr.update(visible=True), outputs=[about_panel])

        # === 檢查登入狀態的函數 ===
        def check_auth_status(request: gr.Request):
            user_session = get_session_user(request)
            if user_session:
                email = user_session.get("email") or ""
                name = user_session.get("name") or ""
                picture = user_session.get("picture") or ""

                # Google 登入後同步 SQLite：insert-or-ignore，再從 DB 讀回 plan/credits（DB 為權威）
                user_data = user_manager.ensure_google_user(email, name=name, picture=picture)

                new_header = _header_logo_html(user_data)
                w_vis, l_vis = _library_welcome_visibility(email)
                # 依登入者 email 載入專屬 Project Library（不再混用 guest）
                gallery_data = get_library_gallery_data(email)

                return (
                    gr.update(visible=False),
                    gr.update(visible=True),
                    "",
                    {"user": user_data},
                    new_header,
                    w_vis,
                    l_vis,
                    gallery_data,  # 顯示該使用者的專案列表
                )
            else:
                return (
                    gr.update(visible=True),
                    gr.update(visible=False),
                    "",
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    gr.update(),
                )

        # 網頁一載入時，立刻觸發檢查
        app.load(
            check_auth_status,
            inputs=None,
            outputs=[login_view, main_app, auth_msg, state, header_html, welcome_screen, library_content, lib_gallery],
        )

    return app

if __name__ == "__main__":
    ui = create_ui()
    ui.launch(inbrowser=True, server_name="0.0.0.0", server_port=7860, theme=sci_fi_theme, css=css_pro, js=js_drag_drop)
