"""
AI annotation module.
Uses OpenAI Vision API to generate scene descriptions.
"""

import os
import base64
from typing import List, Optional
from openai import OpenAI
from scene_detector import SceneInfo


class AIAnnotator:
    """AI scene annotator."""
    
    ANNOTATION_STYLES = {
        "detailed": "Detailed - full description (setting, people, action, mood)",
        "cinematic": "Cinematic - professional film terms (shot type, camera movement)",
        "simple": "Simple - short scene summary",
        "screenplay": "Screenplay - script-style description"
    }
    
    PROMPTS = {
        "detailed": """Describe this film/video frame in detail:
1. Setting (indoor/outdoor, place, time)
2. People (number, appearance, position)
3. Action/behavior
4. Expression/mood
5. Lighting/color
6. Key props

Keep under 100 words.""",
        
        "cinematic": """As a film analyst, describe this shot in film terms:
1. Shot type (close-up/medium/wide/full etc.)
2. Angle (high/low/eye-level/dutch etc.)
3. Camera movement (if any)
4. Composition
5. Subject and setting

Use professional terms, under 80 words.""",
        
        "simple": """Describe the main content of this frame in one short sentence. Under 30 words.""",
        
        "screenplay": """Describe this scene in screenplay format:
SETTING: [place and time]
VISUAL: [what is visible]
ACTION: [what is happening]

Under 80 words."""
    }
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize AI annotator.
        api_key: OpenAI API key; if not provided, read from env OPENAI_API_KEY.
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if self.api_key:
            self.client = OpenAI(api_key=self.api_key)
        else:
            self.client = None
    
    def set_api_key(self, api_key: str):
        """Set API key."""
        self.api_key = api_key
        self.client = OpenAI(api_key=api_key)
    
    def _encode_image(self, image_path: str) -> str:
        """Encode image to base64."""
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    
    def annotate_image(
        self,
        image_path: str,
        style: str = "detailed",
        custom_prompt: Optional[str] = None
    ) -> str:
        """
        Generate annotation for a single image.
        image_path: path to image
        style: annotation style
        custom_prompt: optional custom prompt
        Returns: annotation text
        """
        if not self.client:
            return "[Set OpenAI API key to enable AI annotation]"
        
        if not os.path.exists(image_path):
            return f"[Image not found: {image_path}]"
        
        prompt = custom_prompt or self.PROMPTS.get(style, self.PROMPTS["detailed"])
        base64_image = self._encode_image(image_path)
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}",
                                    "detail": "low"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=300
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            return f"[AI annotation failed: {str(e)}]"
    
    def annotate_scenes(
        self,
        scenes: List[SceneInfo],
        style: str = "detailed",
        custom_prompt: Optional[str] = None,
        progress_callback=None
    ) -> List[SceneInfo]:
        """
        Generate annotations for multiple scenes.
        scenes: list of scenes
        style: annotation style
        custom_prompt: optional custom prompt
        progress_callback: progress callback
        Returns: updated scenes with annotations
        """
        total = len(scenes)
        
        for i, scene in enumerate(scenes):
            if progress_callback:
                progress_callback(
                    i / total,
                    f"Generating annotation {i+1}/{total}..."
                )
            
            if scene.thumbnail_path and os.path.exists(scene.thumbnail_path):
                scene.annotation = self.annotate_image(
                    scene.thumbnail_path,
                    style,
                    custom_prompt
                )
            else:
                scene.annotation = "[No thumbnail]"
        
        if progress_callback:
            progress_callback(1.0, "Annotations done.")
        
        return scenes


class LocalAnnotator:
    """
    Local annotator - no AI, basic scene info only.
    For offline use or when not using the API.
    """
    
    def annotate_scenes(
        self,
        scenes: List[SceneInfo],
        progress_callback=None
    ) -> List[SceneInfo]:
        """Generate basic annotations for scenes."""
        total = len(scenes)
        
        for i, scene in enumerate(scenes):
            if progress_callback:
                progress_callback(i / total, f"Processing {i+1}/{total}...")
            
            scene.annotation = (
                f"Scene #{scene.scene_number}\n"
                f"Time: {scene.start_time:.2f}s - {scene.end_time:.2f}s\n"
                f"Duration: {scene.duration:.2f}s\n"
                f"Frames: {scene.start_frame} - {scene.end_frame}"
            )
        
        if progress_callback:
            progress_callback(1.0, "Done.")
        
        return scenes


if __name__ == "__main__":
    print("AI Annotation module")
    print("Available styles:")
    for key, desc in AIAnnotator.ANNOTATION_STYLES.items():
        print(f"  - {key}: {desc}")
