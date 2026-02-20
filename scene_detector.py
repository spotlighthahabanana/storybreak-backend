"""
Scene detection module (scene_detector.py).
Requires: pip install opencv-python "scenedetect[opencv]"
for Content / Adaptive / Threshold algorithms.
Optional: TransNet V2, CLIP - see requirements.txt.
"""

import os
import cv2
import numpy as np
from pathlib import Path
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector, AdaptiveDetector, ThresholdDetector
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
import time

# --- AI dependency checks ---
TRANSNET_AVAILABLE = False
_transnet_model = None

CLIP_AVAILABLE = False
_clip_model = None
_clip_processor = None

try:
    import torch
    try:
        from transformers import CLIPProcessor, CLIPModel
        from PIL import Image
        CLIP_AVAILABLE = True
    except ImportError:
        pass
except ImportError:
    pass


def _get_transnet_model():
    """Lazy-load TransNet V2 model (singleton)."""
    global _transnet_model, TRANSNET_AVAILABLE
    if _transnet_model is not None:
        return _transnet_model
    try:
        from transnetv2_pytorch import TransNetV2
        import torch
        model = TransNetV2(device='auto')
        model.eval()
        weights_path = None
        for p in [
            Path(__file__).parent / "transnetv2-pytorch-weights.pth",
            Path("transnetv2-pytorch-weights.pth"),
        ]:
            if p.exists():
                weights_path = str(p)
                break
        if not weights_path:
            try:
                from huggingface_hub import hf_hub_download
                weights_path = hf_hub_download(
                    repo_id="Sn4kehead/TransNetV2",
                    filename="transnetv2-pytorch-weights.pth"
                )
            except Exception:
                pass
        if weights_path:
            state_dict = torch.load(weights_path, map_location="cpu")
            # Compatible with different weight sources (Sn4kehead / allenday etc.)
            try:
                model.load_state_dict(state_dict, strict=True)
            except Exception:
                model.load_state_dict(state_dict, strict=False)
            _transnet_model = model
            TRANSNET_AVAILABLE = True
            return model
    except ImportError as e:
        print(f"[TransNetV2] Please install: pip install transnetv2-pytorch torch - {e}")
    except Exception as e:
        print(f"[TransNetV2] Load failed: {e}")
    return None


def _get_clip_model():
    """Lazy-load CLIP model (singleton)."""
    global _clip_model, _clip_processor, CLIP_AVAILABLE
    if _clip_model is not None:
        return _clip_model, _clip_processor

    if not CLIP_AVAILABLE:
        return None, None

    try:
        import torch
        from transformers import CLIPProcessor, CLIPModel
        print("Loading CLIP model (for shot classification)...")
        model_name = "openai/clip-vit-base-patch32"
        _clip_model = CLIPModel.from_pretrained(model_name)
        _clip_processor = CLIPProcessor.from_pretrained(model_name)

        if torch.cuda.is_available():
            _clip_model = _clip_model.to("cuda")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            _clip_model = _clip_model.to("mps")

        return _clip_model, _clip_processor
    except Exception as e:
        print(f"[CLIP] Model load failed: {e}")
        CLIP_AVAILABLE = False
        return None, None


@dataclass
class SceneInfo:
    scene_number: int
    start_time: float
    end_time: float
    start_frame: int
    end_frame: int
    duration: float
    thumbnail_path: Optional[str] = None
    annotation: Optional[str] = None
    video_clip_path: Optional[str] = None
    tag: Optional[str] = None
    group_id: Optional[str] = None
    movement: Optional[str] = None

    @property
    def scene_id(self) -> str:
        """Stable identity for gallery/UI: same frames => same id."""
        return f"{self.start_frame}-{self.end_frame}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scene_number": self.scene_number,
            "scene_id": self.scene_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "thumbnail_path": str(self.thumbnail_path) if self.thumbnail_path else None,
            "annotation": self.annotation,
            "video_clip_path": str(self.video_clip_path) if self.video_clip_path else None,
            "tag": self.tag,
            "group_id": self.group_id,
            "movement": self.movement,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SceneInfo":
        return cls(
            scene_number=int(d.get("scene_number", 0)),
            start_time=float(d.get("start_time", 0)),
            end_time=float(d.get("end_time", 0)),
            start_frame=int(d.get("start_frame", 0)),
            end_frame=int(d.get("end_frame", 0)),
            duration=float(d.get("duration", 0)),
            thumbnail_path=d.get("thumbnail_path"),
            annotation=d.get("annotation"),
            video_clip_path=d.get("video_clip_path"),
            tag=d.get("tag"),
            group_id=d.get("group_id"),
            movement=d.get("movement"),
        )


class CameraMovementAnalyzer:
    """Analyze camera movement (Pan, Tilt, Zoom, Static) using OpenCV optical flow."""
    def analyze(self, video_path, scene: SceneInfo) -> str:
        try:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                return ""

            sample_points = [
                scene.start_frame + 5,
                int((scene.start_frame + scene.end_frame) / 2),
                scene.end_frame - 5
            ]

            total_dx, total_dy = 0.0, 0.0
            total_mag = 0.0
            frame_count = 0

            for p in sample_points:
                if p >= scene.end_frame:
                    continue
                cap.set(cv2.CAP_PROP_POS_FRAMES, p)
                ret, frame1 = cap.read()
                ret2, frame2 = cap.read()

                if not ret or not ret2:
                    continue

                h, w = frame1.shape[:2]
                scale = 320 / w
                small_w, small_h = int(w * scale), int(h * scale)

                prev = cv2.resize(frame1, (small_w, small_h))
                curr = cv2.resize(frame2, (small_w, small_h))

                prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
                curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)

                flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

                dx = np.mean(flow[..., 0])
                dy = np.mean(flow[..., 1])
                mag = np.mean(np.sqrt(flow[..., 0]**2 + flow[..., 1]**2))

                total_dx += dx
                total_dy += dy
                total_mag += mag
                frame_count += 1

            cap.release()

            if frame_count == 0:
                return "Static"

            avg_dx = total_dx / frame_count
            avg_dy = total_dy / frame_count
            avg_mag = total_mag / frame_count

            threshold_static = 0.3
            threshold_move = 0.8

            if avg_mag < threshold_static:
                return "Static"

            result = []
            if abs(avg_dx) > threshold_move:
                result.append("Pan Right" if avg_dx < 0 else "Pan Left")
            if abs(avg_dy) > threshold_move:
                result.append("Tilt Down" if avg_dy < 0 else "Tilt Up")

            if not result:
                return "Handheld"

            return " & ".join(result)

        except Exception as e:
            print(f"Movement analysis error: {e}")
            return ""


class ShotTypeClassifier:
    """Shot type classification with CLIP + OpenCV movement analysis."""
    def __init__(self):
        self.labels = [
            "Extreme Close Up Shot",
            "Close Up Shot",
            "Medium Shot",
            "Full Shot",
            "Wide Shot",
        ]
        self.label_map = {
            "Extreme Close Up Shot": "ECU",
            "Close Up Shot": "CU",
            "Medium Shot": "MS",
            "Full Shot": "FS",
            "Wide Shot": "WS"
        }
        self.movement_analyzer = CameraMovementAnalyzer()

    def classify_scenes(self, scenes: List[SceneInfo], video_path: str, progress_callback=None) -> List[SceneInfo]:
        """Classify scenes and set SceneInfo.tag (shot type + movement)."""
        from PIL import Image
        import torch

        total = len(scenes)
        if progress_callback:
            progress_callback(0.1, "Analyzing movement (Optical Flow)...")

        for i, s in enumerate(scenes):
            mov = self.movement_analyzer.analyze(video_path, s)
            s.movement = mov
            if progress_callback and i % 5 == 0:
                progress_callback(0.1 + 0.2 * (i / total), f"Movement {i}/{total}")

        model, processor = _get_clip_model()
        if model is None:
            for s in scenes:
                if s.movement:
                    s.tag = s.movement
            return scenes

        valid_scenes = [s for s in scenes if s.thumbnail_path and os.path.exists(s.thumbnail_path)]
        if not valid_scenes:
            for s in scenes:
                if s.movement:
                    s.tag = s.movement
            return scenes

        device = model.device
        batch_size = 8
        total_valid = len(valid_scenes)

        if progress_callback:
            progress_callback(0.3, "Starting AI shot classification (CLIP)...")

        for i in range(0, total_valid, batch_size):
            batch_scenes = valid_scenes[i:i + batch_size]
            images = []
            for s in batch_scenes:
                try:
                    img = Image.open(s.thumbnail_path)
                    images.append(img)
                except Exception:
                    images.append(None)

            valid_indices = [idx for idx, img in enumerate(images) if img is not None]
            if not valid_indices:
                continue
            filtered_images = [images[idx] for idx in valid_indices]

            try:
                inputs = processor(text=self.labels, images=filtered_images, return_tensors="pt", padding=True).to(device)
                with torch.no_grad():
                    outputs = model(**inputs)
                    probs = outputs.logits_per_image.softmax(dim=1)

                for j, idx in enumerate(valid_indices):
                    scene = batch_scenes[idx]
                    prob_values = probs[j]
                    top_idx = prob_values.argmax().item()
                    label_en = self.labels[top_idx]
                    label_cn = self.label_map.get(label_en, label_en)

                    tag_parts = [label_cn]
                    if scene.movement and scene.movement != "Static":
                        tag_parts.append(scene.movement)
                    scene.tag = " | ".join(tag_parts)

            except Exception as e:
                print(f"[ShotTypeClassifier] Batch error: {e}")

            if progress_callback:
                progress_callback(0.3 + 0.7 * min((i + batch_size) / total_valid, 1.0), f"AI classifying {min(i + batch_size, total_valid)}/{total_valid}")

        for s in scenes:
            if not s.tag and s.movement:
                s.tag = s.movement

        return scenes


def get_video_info(video_path):
    """Get basic video info."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {}
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps > 0 else 0
    cap.release()
    return {
        'fps': fps,
        'width': width,
        'height': height,
        'frame_count': frame_count,
        'duration': duration
    }


class SceneDetector:
    def __init__(self, output_dir="output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.thumbs_dir = self.output_dir / "thumbnails"
        self.thumbs_dir.mkdir(exist_ok=True)
        self.clips_dir = self.output_dir / "clips"
        self.clips_dir.mkdir(exist_ok=True)
        self.covers_dir = self.output_dir / "covers"
        self.covers_dir.mkdir(exist_ok=True)
        self.classifier = ShotTypeClassifier()

    def process_video(self, video_path, method="adaptive", threshold=30.0, min_scene_len=15, progress_callback=None):
        """Run scene detection and auto classification."""
        video_path = str(video_path)
        if method == "transnet":
            scenes = self._process_video_transnet(video_path, threshold, min_scene_len, progress_callback)
        else:
            video_manager = VideoManager([video_path])
            scene_manager = SceneManager()

            actual_threshold = float(threshold) / 10 if method == "adaptive" else float(threshold)

            if method == "adaptive":
                dt = AdaptiveDetector(adaptive_threshold=actual_threshold, min_scene_len=int(min_scene_len))
            elif method == "threshold":
                dt = ThresholdDetector(threshold=actual_threshold, min_scene_len=int(min_scene_len))
            else:
                dt = ContentDetector(threshold=actual_threshold, min_scene_len=int(min_scene_len))

            scene_manager.add_detector(dt)
            video_manager.set_downscale_factor()
            video_manager.start()
            scene_manager.detect_scenes(frame_source=video_manager)

            scene_list = scene_manager.get_scene_list()
            video_manager.release()
            optimized_scene_list = self._optimize_scene_list(video_path, scene_list, min_duration_sec=0.8)
            scenes = self._process_scene_list(video_path, optimized_scene_list)

        if progress_callback:
            progress_callback(0.8, "Visual grouping...")
        scenes = self.group_scenes(scenes, threshold=0.75)

        scenes = self.classifier.classify_scenes(scenes, video_path, progress_callback)

        return scenes

    def _process_video_transnet(self, video_path, threshold=30.0, min_scene_len=15, progress_callback=None):
        """Run scene detection with TransNet V2."""
        if progress_callback:
            progress_callback(0.1, "Initializing TransNet V2...")

        model = _get_transnet_model()
        if model is None:
            if progress_callback:
                progress_callback(0.2, "TransNet V2 not loaded, falling back to Adaptive...")
            return self.process_video(video_path, method="adaptive", threshold=threshold, min_scene_len=min_scene_len, progress_callback=progress_callback)

        transnet_threshold = max(0.1, min(0.8, float(threshold) / 100.0))

        if progress_callback:
            progress_callback(0.2, "TransNet V2 analyzing...")

        import torch
        with torch.no_grad():
            scenes_raw = model.detect_scenes(video_path, threshold=transnet_threshold)

        if not scenes_raw:
            if progress_callback:
                progress_callback(0.5, "No scenes detected, falling back to Adaptive...")
            return self.process_video(video_path, method="adaptive", threshold=threshold, min_scene_len=min_scene_len, progress_callback=progress_callback)

        info = get_video_info(video_path)
        fps = info.get('fps', 24.0) or 24.0

        processed = []
        cap = cv2.VideoCapture(video_path)
        for i, s in enumerate(scenes_raw):
            start_time = float(s.get('start_time', s.get('start', 0)))
            end_time = float(s.get('end_time', s.get('end', start_time + 1)))
            start_frame = int(start_time * fps)
            end_frame = int(end_time * fps)
            duration = end_time - start_time

            mid_frame = int(start_frame + (end_frame - start_frame) / 2)
            thumb_path = None
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_POS_FRAMES, mid_frame)
                ret, frame = cap.read()
                if ret and frame is not None:
                    thumb_name = f"{Path(video_path).stem}_scene_{i+1:03d}.jpg"
                    thumb_full = self.thumbs_dir / thumb_name
                    cv2.imwrite(str(thumb_full), frame)
                    thumb_path = str(thumb_full)

            processed.append(SceneInfo(
                scene_number=i + 1,
                start_time=start_time,
                end_time=end_time,
                start_frame=start_frame,
                end_frame=end_frame,
                duration=duration,
                thumbnail_path=thumb_path
            ))

        if cap.isOpened():
            cap.release()

        if progress_callback:
            progress_callback(0.9, f"TransNet V2 detected {len(processed)} scenes")

        return processed

    def group_scenes(self, scenes: List[SceneInfo], threshold: float = 0.75, progress_callback=None):
        """Group scenes by visual similarity."""
        n = len(scenes)
        if n == 0:
            return scenes

        hists = []
        valid_indices = []

        for i, s in enumerate(scenes):
            hist = None
            if s.thumbnail_path and os.path.exists(s.thumbnail_path):
                try:
                    img = cv2.imread(s.thumbnail_path)
                    if img is not None:
                        img_small = cv2.resize(img, (64, 64))
                        hsv = cv2.cvtColor(img_small, cv2.COLOR_BGR2HSV)
                        hist = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
                        cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
                except Exception:
                    pass
            hists.append(hist)
            if hist is not None:
                valid_indices.append(i)

        adj = {i: [] for i in range(n)}
        num_valid = len(valid_indices)

        for idx_i in range(num_valid):
            i = valid_indices[idx_i]
            for idx_j in range(idx_i + 1, num_valid):
                j = valid_indices[idx_j]
                sim = cv2.compareHist(hists[i], hists[j], cv2.HISTCMP_CORREL)
                if sim >= threshold:
                    adj[i].append(j)
                    adj[j].append(i)

        visited = set()
        group_id_counter = 0

        for i in range(n):
            if i not in visited:
                if i not in adj or not adj[i]:
                    scenes[i].group_id = group_id_counter
                    group_id_counter += 1
                    visited.add(i)
                    continue

                stack = [i]
                visited.add(i)
                members = []
                while stack:
                    curr = stack.pop()
                    members.append(curr)
                    for neighbor in adj[curr]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            stack.append(neighbor)

                for m in members:
                    scenes[m].group_id = group_id_counter
                group_id_counter += 1

        return scenes

    def _calculate_histogram_similarity(self, frame1, frame2):
        if frame1 is None or frame2 is None:
            return 0.0
        hsv1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2HSV)
        hsv2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2HSV)
        hist1 = cv2.calcHist([hsv1], [0, 1], None, [180, 256], [0, 180, 0, 256])
        hist2 = cv2.calcHist([hsv2], [0, 1], None, [180, 256], [0, 180, 0, 256])
        cv2.normalize(hist1, hist1, 0, 1, cv2.NORM_MINMAX)
        cv2.normalize(hist2, hist2, 0, 1, cv2.NORM_MINMAX)
        return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

    def _optimize_scene_list(self, video_path, scene_list, min_duration_sec=0.8):
        if not scene_list:
            return []
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return scene_list

        optimized = []
        temp_list = [list(scene) for scene in scene_list]
        i = 0

        while i < len(temp_list):
            current_scene = temp_list[i]
            if i == len(temp_list) - 1:
                optimized.append(current_scene)
                break

            next_scene = temp_list[i + 1]
            duration = current_scene[1].get_seconds() - current_scene[0].get_seconds()

            if duration < min_duration_sec:
                frame_prev_idx = max(0, current_scene[0].get_frames() - 5)
                frame_next_idx = next_scene[0].get_frames() + 5
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_prev_idx)
                ret1, frame_prev = cap.read()
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_next_idx)
                ret2, frame_next = cap.read()
                similarity = 0.0
                if ret1 and ret2:
                    similarity = self._calculate_histogram_similarity(frame_prev, frame_next)
                if similarity > 0.7:
                    merged_scene = [current_scene[0], next_scene[1]]
                    temp_list[i] = merged_scene
                    temp_list.pop(i + 1)
                    continue
                else:
                    optimized.append(current_scene)
                    i += 1
            else:
                optimized.append(current_scene)
                i += 1

        cap.release()
        return optimized

    def _process_scene_list(self, video_path, scene_list):
        processed_scenes = []
        cap = cv2.VideoCapture(str(video_path))

        for i, scene in enumerate(scene_list):
            start_frame, end_frame = scene[0].get_frames(), scene[1].get_frames()
            start_time, end_time = scene[0].get_seconds(), scene[1].get_seconds()
            duration = end_time - start_time

            mid_frame = int(start_frame + (end_frame - start_frame) / 2)
            cap.set(cv2.CAP_PROP_POS_FRAMES, mid_frame)
            ret, frame = cap.read()

            thumb_path = None
            if ret:
                thumb_name = f"{Path(video_path).stem}_scene_{i+1:03d}.jpg"
                thumb_full_path = self.thumbs_dir / thumb_name
                cv2.imwrite(str(thumb_full_path), frame)
                thumb_path = str(thumb_full_path)

            s_info = SceneInfo(
                scene_number=i + 1,
                start_time=start_time,
                end_time=end_time,
                start_frame=start_frame,
                end_frame=end_frame,
                duration=duration,
                thumbnail_path=thumb_path
            )
            processed_scenes.append(s_info)

        cap.release()
        return processed_scenes

    def extract_video_clips(self, video_path, scenes, progress_callback=None):
        """Extract clips with FFmpeg (H.264)."""
        import subprocess
        video_path = str(video_path)
        try:
            import imageio_ffmpeg
            ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            ffmpeg_path = "ffmpeg"
        print(f"[FFmpeg] using: {ffmpeg_path}")

        total = len(scenes)
        use_ffmpeg = True
        creation_flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0

        for i, scene in enumerate(scenes):
            if progress_callback:
                progress_callback(i / total, f"Exporting clip {i+1}/{total}")

            out_name = f"{Path(video_path).stem}_scene_{scene.scene_number:03d}.mp4"
            out_path = self.clips_dir / out_name
            start_time = scene.start_time
            duration = scene.duration

            if use_ffmpeg:
                cmd = [
                    ffmpeg_path, "-y",
                    "-ss", str(start_time),
                    "-i", video_path,
                    "-t", str(duration),
                    "-c:v", "libx264", "-preset", "fast", "-crf", "23",
                    "-c:a", "aac",
                    "-movflags", "+faststart",
                    str(out_path)
                ]
                try:
                    subprocess.run(cmd, capture_output=True, check=True, creationflags=creation_flags)
                    scene.video_clip_path = str(out_path)
                except (subprocess.CalledProcessError, FileNotFoundError, OSError):
                    use_ffmpeg = False
                    if i == 0:
                        print("[Note] FFmpeg not available, using OpenCV")

            if not use_ffmpeg:
                cap = cv2.VideoCapture(video_path)
                fps = cap.get(cv2.CAP_PROP_FPS)
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fourcc = cv2.VideoWriter_fourcc(*'avc1')
                out = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
                if not out.isOpened():
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    out = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
                cap.set(cv2.CAP_PROP_POS_FRAMES, scene.start_frame)
                current_frame = scene.start_frame
                while current_frame < scene.end_frame:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    out.write(frame)
                    current_frame += 1
                out.release()
                cap.release()
                scene.video_clip_path = str(out_path)

        if progress_callback:
            progress_callback(1.0, "Done")
        return scenes

    def extract_cover(self, video_path, position=0.1):
        try:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                return None
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames <= 0:
                cap.release()
                return None
            target_frame = int(total_frames * position)
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
            ret, frame = cap.read()
            cap.release()
            if not ret or frame is None:
                return None
            import hashlib
            video_hash = hashlib.md5(Path(video_path).name.encode()).hexdigest()[:8]
            cover_path = self.covers_dir / f"cover_{video_hash}.jpg"
            cv2.imwrite(str(cover_path), frame)
            return str(cover_path)
        except Exception as e:
            print(f"Cover extraction failed: {e}")
            return None

    def extract_thumbnails(self, video_path, scenes, thumbnail_position=0.3, progress_callback=None):
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")
        total = len(scenes)
        for i, scene in enumerate(scenes):
            if progress_callback:
                progress_callback(i / total, f"Extracting thumbnail {i+1}/{total}")
            target_frame = int(scene.start_frame + (scene.end_frame - scene.start_frame) * thumbnail_position)
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
            ret, frame = cap.read()
            if ret and frame is not None:
                thumb_name = f"{Path(video_path).stem}_scene_{scene.scene_number:03d}.jpg"
                thumb_path = self.thumbs_dir / thumb_name
                cv2.imwrite(str(thumb_path), frame)
                scene.thumbnail_path = str(thumb_path)
        cap.release()
        if progress_callback:
            progress_callback(1.0, "Done")
        return scenes

    def extract_thumbnail_at_position(self, video_path, scene, position=0.3):
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None
        target_frame = int(scene.start_frame + (scene.end_frame - scene.start_frame) * position)
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        ret, frame = cap.read()
        cap.release()
        if not ret or frame is None:
            return None
        thumb_name = f"{Path(video_path).stem}_scene_{scene.scene_number:03d}.jpg"
        thumb_path = self.thumbs_dir / thumb_name
        cv2.imwrite(str(thumb_path), frame)
        return str(thumb_path)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        detector = SceneDetector()
        scenes = detector.process_video(sys.argv[1])
        print(f"\nDetected {len(scenes)} scenes")
        for s in scenes:
            print(f"  Scene {s.scene_number}: {s.start_time:.2f}s - {s.end_time:.2f}s, tag: {s.tag}")
