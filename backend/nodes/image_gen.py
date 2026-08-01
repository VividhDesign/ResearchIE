"""Image generation node — uses Gemini Imagen to generate real images for the report."""
from __future__ import annotations
import os
import re
import base64
from pathlib import Path
from datetime import datetime
from langchain_core.messages import HumanMessage
from backend.state import ResearchState
from backend.utils.llm import get_llm, extract_text


# ---------------------------------------------------------------------------
# Step 1 – Image Planner: ask the LLM to place [[IMAGE_N]] placeholders
# in the markdown and return the prompt for each image.
# ---------------------------------------------------------------------------

IMAGE_PLANNER_PROMPT = """You are an expert research report editor and visual content specialist.

You have a completed research report in markdown format. Your job is to:
1. Identify 2-4 ideal positions in the report where a relevant image would significantly enhance understanding.
2. Insert a placeholder token like [[IMAGE_1]], [[IMAGE_2]], etc. at each position (right after the section heading or paragraph where it makes most sense).
3. For EACH placeholder, write a detailed, specific image generation prompt (for an AI image model like Imagen/DALL-E) that describes exactly what the image should show. The prompt must be highly descriptive and specific to the report topic — NOT generic.

Rules:
- Place images after ## headings or after key explanatory paragraphs, NOT in the middle of sentences.
- Image prompts must be informative, technical-diagram-style or data-visualization-style, NOT decorative/abstract.
- Each image prompt should describe a concept, process, comparison, or data visualization related to the exact topic.
- Format: Insert [[IMAGE_N]] on its own line, then continue the markdown normally.

Return ONLY a JSON object with this exact structure:
{{
  "markdown_with_placeholders": "<the full report markdown with [[IMAGE_N]] tokens inserted>",
  "images": [
    {{
      "placeholder": "[[IMAGE_1]]",
      "prompt": "<detailed image generation prompt>",
      "caption": "<short caption for the image, max 15 words>"
    }},
    ...
  ]
}}

Report:
{report}
"""


def _generate_image_with_gemini(prompt: str, output_path: str) -> bool:
    """Generate an image using Gemini Imagen API. Returns True on success."""
    try:
        import google.generativeai as genai
        
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            return False
            
        genai.configure(api_key=api_key)
        
        # Use Imagen 3 via the google-generativeai SDK
        imagen = genai.ImageGenerationModel("imagen-3.0-generate-002")
        result = imagen.generate_images(
            prompt=prompt,
            number_of_images=1,
            safety_filter_level="block_only_high",
            person_generation="allow_adult",
            aspect_ratio="16:9",
        )
        
        if result.images:
            image = result.images[0]
            # Save as PNG
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "wb") as f:
                f.write(image._image_bytes)
            return True
    except Exception as e:
        print(f"[image_gen] Imagen generation error: {e}")
    return False


def _generate_image_with_pollinations(prompt: str, output_path: str) -> bool:
    """Free fallback: generate image using Pollinations.ai (no API key needed)."""
    try:
        import requests
        import urllib.parse
        
        encoded = urllib.parse.quote(prompt)
        # Pollinations.ai — completely free, no rate limits, no API key
        url = f"https://image.pollinations.ai/prompt/{encoded}?width=1200&height=675&nologo=true&enhance=true"
        
        resp = requests.get(url, timeout=60)
        if resp.status_code == 200 and len(resp.content) > 10000:  # at least 10KB
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "wb") as f:
                f.write(resp.content)
            return True
    except Exception as e:
        print(f"[image_gen] Pollinations fallback error: {e}")
    return False


def _image_to_base64(image_path: str) -> str | None:
    """Convert a saved image to a base64 data URI for inline embedding."""
    try:
        with open(image_path, "rb") as f:
            data = base64.b64encode(f.read()).decode("utf-8")
        ext = Path(image_path).suffix.lstrip(".").lower()
        mime = "image/jpeg" if ext in ("jpg", "jpeg") else "image/png"
        return f"data:{mime};base64,{data}"
    except Exception:
        return None


def image_gen_node(state: ResearchState) -> dict:
    """
    1. Use LLM to plan image placements (insert [[IMAGE_N]] placeholders).
    2. For each placeholder, generate a real image using Imagen or Pollinations.
    3. Replace placeholders in the markdown with actual embedded images.
    4. Store generated images separately for the UI Images tab.
    """
    import json
    
    report = state.stitched_report
    if not report or len(report) < 200:
        return {
            "stitched_report": report,
            "progress_log": ["[image_gen] Report too short, skipping image generation."],
            "status": "📸 Image generation skipped (short report)",
        }

    llm = get_llm(temperature=0.4)
    
    # --- Step 1: Plan image placements ---
    try:
        prompt = IMAGE_PLANNER_PROMPT.format(report=report[:8000])  # truncate for token safety
        response = llm.invoke([HumanMessage(content=prompt)])
        raw = extract_text(response.content).strip()
        
        # Strip code fences if the LLM wrapped in ```json ... ```
        raw = re.sub(r'^```(?:json)?\s*', '', raw)
        raw = re.sub(r'\s*```$', '', raw)
        
        plan = json.loads(raw)
        md_with_placeholders = plan.get("markdown_with_placeholders", report)
        image_specs = plan.get("images", [])
    except Exception as e:
        return {
            "stitched_report": report,
            "progress_log": [f"[image_gen] Planning failed: {e}. Skipping images."],
            "status": "📸 Image planning failed",
        }
    
    if not image_specs:
        return {
            "stitched_report": report,
            "progress_log": ["[image_gen] No images planned."],
            "status": "📸 No images to generate",
        }

    # --- Step 2: Generate images and replace placeholders ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    images_dir = Path(__file__).parent.parent.parent / "data" / "generated_images"
    images_dir.mkdir(parents=True, exist_ok=True)
    
    generated_images = []  # list of dicts for UI Images tab
    progress = []
    final_md = md_with_placeholders
    
    for i, spec in enumerate(image_specs):
        placeholder = spec.get("placeholder", f"[[IMAGE_{i+1}]]")
        img_prompt = spec.get("prompt", "")
        caption = spec.get("caption", f"Figure {i+1}")
        
        if not img_prompt:
            continue
        
        # File path for this image
        img_filename = f"report_{timestamp}_img{i+1}.png"
        img_path = str(images_dir / img_filename)
        
        success = False
        method = ""
        
        # Try Gemini Imagen first (requires google-generativeai package)
        try:
            import google.generativeai
            success = _generate_image_with_gemini(img_prompt, img_path)
            if success:
                method = "Imagen 3"
        except ImportError:
            pass
        
        # Fall back to Pollinations.ai (free, no API key needed)
        if not success:
            success = _generate_image_with_pollinations(img_prompt, img_path)
            if success:
                method = "Pollinations.ai"
        
        if success:
            # Convert to base64 for inline embedding (works on ephemeral Streamlit Cloud)
            b64_uri = _image_to_base64(img_path)
            
            if b64_uri:
                # Replace placeholder with an actual markdown image tag
                img_md = f"\n\n![{caption}]({b64_uri})\n*{caption}*\n\n"
                final_md = final_md.replace(placeholder, img_md)
                generated_images.append({
                    "placeholder": placeholder,
                    "caption": caption,
                    "prompt": img_prompt,
                    "base64_uri": b64_uri,
                    "method": method,
                })
                progress.append(f"📸 Image {i+1} generated via {method}: '{caption}'")
            else:
                final_md = final_md.replace(placeholder, "")
                progress.append(f"⚠️ Image {i+1} generated but could not be embedded.")
        else:
            # Remove the placeholder so the report doesn't have broken tokens
            final_md = final_md.replace(placeholder, "")
            progress.append(f"⚠️ Image {i+1} generation failed (all providers exhausted).")
    
    return {
        "stitched_report": final_md,
        "generated_images": generated_images,
        "status": f"📸 {len(generated_images)} image(s) generated",
        "progress_log": progress or ["[image_gen] No images were successfully generated."],
    }
