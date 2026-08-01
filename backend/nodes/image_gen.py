"""Image generation node — generates real images for each report section.

Strategy: 
- SKIP LLM JSON parsing (too fragile). Instead, derive image prompts directly from 
  section titles and the topic. This always works even if the LLM is flaky.
- Use Pollinations.ai as the reliable free generator (no API key, no rate limit).
- Embed images as base64 data URIs so they survive Streamlit Cloud's ephemeral FS.
"""
from __future__ import annotations
import os
import re
import base64
import urllib.parse
import requests
from pathlib import Path
from datetime import datetime
from backend.state import ResearchState


# ---------------------------------------------------------------------------
# Image generation using Pollinations.ai (free, no API key needed)
# ---------------------------------------------------------------------------

def _make_image_prompt(topic: str, section_title: str, section_text: str) -> str:
    """Create a good, specific image prompt from section context."""
    # Extract key concepts from section text (first 500 chars)
    snippet = section_text[:500].strip()
    
    # Build a prompt that combines topic + section title + key terms
    base = (
        f"Professional technical research illustration for topic: '{topic}'. "
        f"Section: '{section_title}'. "
        f"Style: clean, modern data visualization or technical diagram, "
        f"white background, scientific/academic aesthetic, infographic style. "
        f"NO people or faces. Include relevant charts, graphs, or concept diagrams."
    )
    return base


def _generate_image_pollinations(prompt: str, output_path: str) -> bool:
    """Generate an image using Pollinations.ai. Returns True on success."""
    try:
        encoded = urllib.parse.quote(prompt)
        url = (
            f"https://image.pollinations.ai/prompt/{encoded}"
            f"?width=1200&height=675&nologo=true&enhance=true&model=flux"
        )
        
        resp = requests.get(url, timeout=90)
        if resp.status_code == 200 and len(resp.content) > 5000:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "wb") as f:
                f.write(resp.content)
            return True
    except Exception as e:
        print(f"[image_gen] Pollinations error: {e}")
    return False


def _to_base64_uri(image_path: str) -> str | None:
    """Convert a saved image file to a base64 data URI."""
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
    Generate real images using Pollinations.ai for 2-3 key sections.
    Embeds them as base64 data URIs directly into the stitched_report markdown.
    """
    report = state.stitched_report
    topic = state.topic or "research topic"
    
    if not report or len(report) < 100:
        return {
            "stitched_report": report,
            "generated_images": [],
            "progress_log": ["[image_gen] Report too short, skipping."],
            "status": "📸 Image generation skipped",
        }

    # --- Pick which sections to illustrate ---
    # Find ## headings in the report
    section_matches = list(re.finditer(r'^## (.+)$', report, re.MULTILINE))
    
    if not section_matches:
        return {
            "stitched_report": report,
            "generated_images": [],
            "progress_log": ["[image_gen] No sections found to illustrate."],
            "status": "📸 No sections to illustrate",
        }
    
    # Pick up to 3 evenly-distributed sections to avoid rate limiting
    num_to_generate = min(3, len(section_matches))
    step = max(1, len(section_matches) // num_to_generate)
    chosen_matches = section_matches[::step][:num_to_generate]
    
    # Prepare output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    images_dir = Path(__file__).parent.parent.parent / "data" / "generated_images"
    images_dir.mkdir(parents=True, exist_ok=True)
    
    generated_images = []
    progress = []
    final_md = report

    for i, match in enumerate(chosen_matches):
        section_title = match.group(1).strip()
        # Get the section body text (up to 800 chars after the heading)
        start = match.end()
        next_match = section_matches[section_matches.index(match) + 1] if (section_matches.index(match) + 1 < len(section_matches)) else None
        end = next_match.start() if next_match else min(start + 800, len(report))
        section_body = report[start:end].strip()
        
        # Build the image prompt
        img_prompt = _make_image_prompt(topic, section_title, section_body)
        caption = f"Figure {i+1}: {section_title[:60]}"
        
        img_filename = f"report_{timestamp}_img{i+1}.png"
        img_path = str(images_dir / img_filename)
        
        progress.append(f"[image_gen] Generating image {i+1} for '{section_title}'...")
        
        success = _generate_image_pollinations(img_prompt, img_path)
        
        if success:
            b64_uri = _to_base64_uri(img_path)
            if b64_uri:
                img_tag = f"\n\n![{caption}]({b64_uri})\n\n*{caption}*\n\n"
                # Inject image right after the ## heading line
                heading_str = match.group(0)  # "## Section Title"
                # Only replace the first occurrence
                final_md = final_md.replace(heading_str, heading_str + img_tag, 1)
                
                generated_images.append({
                    "placeholder": f"[[IMAGE_{i+1}]]",
                    "caption": caption,
                    "prompt": img_prompt,
                    "base64_uri": b64_uri,
                    "method": "Pollinations.ai (flux)",
                    "section": section_title,
                })
                progress.append(f"📸 Image {i+1} created for '{section_title}'")
            else:
                progress.append(f"⚠️ Image {i+1}: generated but base64 encoding failed.")
        else:
            progress.append(f"⚠️ Image {i+1}: Pollinations.ai request failed for '{section_title}'.")

    status_msg = f"📸 {len(generated_images)} image(s) generated" if generated_images else "📸 Image generation failed (network error)"
    
    return {
        "stitched_report": final_md,
        "generated_images": generated_images,
        "status": status_msg,
        "progress_log": progress,
    }
