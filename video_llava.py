"""
LLaVA-NeXT (Video-LLaVA) video understanding module.
Uses HuggingFace LlavaNextVideoForConditionalGeneration to generate AI notes for scenes.
"""

import os
import numpy as np
from typing import Optional, List, Any
from pathlib import Path

# Classification prompt (short label format for gallery filter)
CLASSIFY_PROMPT = "Describe this video shot in a short label: shot type (wide/medium/close-up/full etc.) | movement (static/pan/tilt/push etc.). Output only the label, under 50 words."

# Lazy load to save resources at startup
_model = None
_processor = None
_AVAILABLE = None


def _check_available() -> bool:
    """Check if LLaVA-NeXT Video is available."""
    global _AVAILABLE
    if _AVAILABLE is not None:
        return _AVAILABLE
    try:
        import torch
        from transformers import LlavaNextVideoForConditionalGeneration, LlavaNextVideoProcessor
        _AVAILABLE = True
    except ImportError:
        _AVAILABLE = False
    return _AVAILABLE


def _get_model():
    """Lazy-load LLaVA-NeXT Video model (singleton)."""
    global _model, _processor
    if _model is not None and _processor is not None:
        return _model, _processor
    if not _check_available():
        return None, None
    try:
        import torch
        import av
        from transformers import LlavaNextVideoForConditionalGeneration, LlavaNextVideoProcessor

        model_id = "llava-hf/LLaVA-NeXT-Video-7B-hf"
        print("[Video-LLaVA] Loading LLaVA-NeXT Video model...")
        _model = LlavaNextVideoForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch.float16, device_map="auto"
        )
        _processor = LlavaNextVideoProcessor.from_pretrained(model_id)
        _model.eval()
        print("[Video-LLaVA] Model loaded.")
        return _model, _processor
    except Exception as e:
        print(f"[Video-LLaVA] Load failed: {e}")
        return None, None


def _read_video_frames(video_path: str, num_frames: int = 8, start_sec: float = 0, end_sec: Optional[float] = None) -> Optional[np.ndarray]:
    """Sample frames from video with PyAV; returns (num_frames, H, W, 3)."""
    try:
        import av
    except ImportError:
        print("[Video-LLaVA] Please install: pip install av")
        return None

    video_path = str(video_path)
    if not os.path.exists(video_path):
        return None

    try:
        container = av.open(video_path)
        stream = container.streams.video[0]
        total_frames = stream.frames
        fps = stream.average_rate
        if fps is None or fps <= 0:
            fps = 24.0
        duration_sec = total_frames / float(fps) if total_frames and fps else 60.0

        start_frame = int(start_sec * fps)
        end_frame = int((end_sec or duration_sec) * fps) if end_sec else total_frames or int(duration_sec * fps)
        if end_frame <= start_frame:
            end_frame = start_frame + int(fps * 2)

        frame_indices = np.linspace(start_frame, min(end_frame - 1, total_frames - 1) if total_frames else end_frame - 1, num_frames, dtype=int)
        frame_indices = np.clip(frame_indices, 0, (total_frames or 1) - 1)

        frames = []
        container.seek(0)
        for i, frame in enumerate(container.decode(video=0)):
            if i > int(frame_indices[-1]) + 1:
                break
            if i in frame_indices:
                arr = frame.to_ndarray(format="rgb24")
                frames.append(arr)
        container.close()

        if not frames:
            return None
        return np.stack(frames)
    except Exception as e:
        print(f"[Video-LLaVA] Failed to read video: {e}")
        return None


def _extract_assistant_response(raw: str, prompt: str) -> str:
    """Extract plain description from LLaVA chat output; strip USER/ASSISTANT tags and duplicate prompt."""
    text = raw.strip()
    if not text:
        return ""
    # Take content after ASSISTANT: or ASSISTANT：
    for marker in ("ASSISTANT:", "ASSISTANT：", "assistant:", "assistant："):
        if marker in text:
            text = text.split(marker, 1)[-1].strip()
            break
    # Remove any remaining USER: / USER： lines
    lines = text.split("\n")
    cleaned = [
        ln for ln in lines
        if not (ln.strip().upper().startswith("USER:") or ln.strip().startswith("USER："))
    ]
    text = "\n".join(cleaned).strip()
    # Remove duplicate prompt
    if prompt and prompt in text:
        text = text.replace(prompt, "").strip()
    return text.strip()


def _read_image_as_video(image_path: str) -> Optional[np.ndarray]:
    """Convert single image to 1-frame video array (1, H, W, 3)."""
    try:
        from PIL import Image
        img = Image.open(image_path).convert("RGB")
        arr = np.array(img)
        return arr[np.newaxis, ...]
    except Exception as e:
        print(f"[Video-LLaVA] Failed to read image: {e}")
        return None


def generate_scene_note(
    video_path: Optional[str] = None,
    image_path: Optional[str] = None,
    start_sec: float = 0,
    end_sec: Optional[float] = None,
    prompt: Optional[str] = None,
    max_new_tokens: int = 150,
) -> str:
    """
    Generate description for video/image using LLaVA-NeXT Video.

    Args:
        video_path: Path to video (clip or main file)
        image_path: Fallback image if no video
        start_sec: Start time in main video
        end_sec: End time
        prompt: Custom prompt
        max_new_tokens: Max generated tokens

    Returns:
        Description text, or error message on failure.
    """
    model, processor = _get_model()
    if model is None or processor is None:
        return "[Please install transformers>=5.0 and av: pip install transformers av]"

    default_prompt = "Describe this video frame concisely: setting, people, action. Under 100 words."
    text_prompt = prompt or default_prompt

    video = None
    if video_path and os.path.exists(video_path):
        video = _read_video_frames(video_path, num_frames=8, start_sec=start_sec, end_sec=end_sec)
    if video is None and image_path and os.path.exists(image_path):
        video = _read_image_as_video(image_path)
    if video is None:
        return "[Cannot read video or thumbnail]"

    try:
        import torch

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": text_prompt},
                    {"type": "video"},
                ],
            },
        ]
        prompt_formatted = processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = processor(text=prompt_formatted, videos=video, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)

        generated = processor.batch_decode(out, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        text = generated[0].strip() if generated else ""
        if text and text.endswith("</s>"):
            text = text[:-4].strip()
        text = _extract_assistant_response(text, text_prompt)
        return text or "[No output]"
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"[Generation failed: {str(e)}]"


def classify_scenes_with_llava(
    scenes: List[Any],
    video_path: str,
    progress_callback=None,
) -> List[Any]:
    """
    Classify scenes with LLaVA-NeXT Video; set scene.tag for filtering.
    """
    if not scenes:
        return scenes
    total = len(scenes)
    for i, s in enumerate(scenes):
        if progress_callback:
            progress_callback(i / total, f"LLaVA-NeXT analyzing {i+1}/{total}...")
        clip_path = getattr(s, "video_clip_path", None)
        thumb_path = getattr(s, "thumbnail_path", None)
        start_sec = getattr(s, "start_time", 0)
        end_sec = getattr(s, "end_time", None)
        if clip_path and os.path.exists(clip_path):
            tag = generate_scene_note(video_path=clip_path, prompt=CLASSIFY_PROMPT, max_new_tokens=80)
        elif video_path and os.path.exists(video_path):
            tag = generate_scene_note(
                video_path=video_path,
                start_sec=start_sec,
                end_sec=end_sec,
                prompt=CLASSIFY_PROMPT,
                max_new_tokens=80,
            )
        elif thumb_path and os.path.exists(thumb_path):
            tag = generate_scene_note(image_path=thumb_path, prompt=CLASSIFY_PROMPT, max_new_tokens=80)
        else:
            tag = "[No input]"
        s.tag = tag.strip() if tag and not tag.startswith("[") else ""
    if progress_callback:
        progress_callback(1.0, "LLaVA-NeXT classification done")
    return scenes
