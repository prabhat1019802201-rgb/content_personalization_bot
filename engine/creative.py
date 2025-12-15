# engine/creative.py
"""
Creative generation utilities for content_personalization_bot.

Responsibilities:
- Build 1..N image prompts per text variant
- Generate images (pluggable backends: stub, Automatic1111 http, Diffusers)
- Compose headline / subtitle / CTA onto images using Pillow
- Run a Vision-Language Model review (via Ollama HTTP) for compliance/legibility
- Save creatives and metadata for audit

Notes:
- For local experiments use the stub (fast) or Automatic1111 if running locally.
- Diffusers helper is included but may require model weights and setup (SDXL).
- The main entrypoint used by pipeline is `generate_creatives_for_variant(...)`
  which returns {"folder": <path>, "meta": <metadata dict>}
"""

from __future__ import annotations
import os
import io
import json
import time
import base64
import hashlib
import logging
from typing import List, Dict, Tuple, Optional, Callable

from PIL import Image, ImageDraw, ImageFont, ImageOps
import requests

# Optional heavy imports guarded at runtime
try:
    import torch
    from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler
    from diffusers.utils import is_xformers_available
    _HAS_DIFFUSERS = True
except Exception:
    _HAS_DIFFUSERS = False

# ----------------------
# Configuration
# ----------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
CREATIVES_DIR = os.path.join(PROJECT_ROOT, "assets", "creatives")
os.makedirs(CREATIVES_DIR, exist_ok=True)

OLLAMA_URL = "http://localhost:11434/api/generate"
VLM_DEFAULT_MODEL = "qwen2.5vl:7b"

AUTOMATIC1111_API = "http://localhost:7860"
AUTO1111_TXT2IMG = f"{AUTOMATIC1111_API}/sdapi/v1/txt2img"

# ----------------------
# Prompt templates (image)
# ----------------------
IMAGE_PROMPT_TEMPLATES = {
    "ev_loan": (
        "Photorealistic electric vehicle on a clean modern street at golden hour, "
        "clean composition with space on left for text overlay, Union Bank green accent, "
        "minimal clutter, high resolution, photorealistic"
    ),
    "premium_card": (
        "Sophisticated premium lifestyle scene, warm lighting, elegant composition, "
        "abstract credit card motif, room for overlay text, editorial quality"
    ),
    "diwali": (
        "Diwali festival lights, warm glow, tasteful family celebration scene, "
        "cultural motif and lanterns, plenty of negative space for headline"
    ),
    "generic": "Clean editorial marketing photo with clear negative space for overlay text."
}

# ----------------------
# Utilities
# ----------------------
def _hash_for_filename(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:10]


def _unique_creative_folder(campaign_id: str, variant_tag: str) -> str:
    ts = int(time.time())
    folder = os.path.join(CREATIVES_DIR, f"{campaign_id}_{variant_tag}_{ts}")
    os.makedirs(folder, exist_ok=True)
    return folder


# ----------------------
# Image prompt generator
# ----------------------
def generate_image_prompts(
    product_key: str,
    variant_text_brief: str,
    n_prompts: int = 3,
) -> List[str]:
    base = IMAGE_PROMPT_TEMPLATES.get(product_key, IMAGE_PROMPT_TEMPLATES["generic"])
    prompts = []
    for i in range(n_prompts):
        prompt = f"{base}. Focus: {variant_text_brief}. Style: clean, high contrast, photography."
        if i % 2 == 0:
            prompt += " shallow depth of field, crisp subject"
        else:
            prompt += " clean background, high-detail texture"
        prompts.append(prompt)
    return prompts


# ----------------------
# Image generation helpers (pluggable)
# ----------------------
def run_image_model_stub(prompt: str, width: int = 1200, height: int = 400, seed: Optional[int] = None) -> Image.Image:
    """Creates a placeholder image with faint prompt text for development."""
    img = Image.new("RGBA", (width, height), (245, 245, 245, 255))
    draw = ImageDraw.Draw(img)
    # faint background rectangle
    draw.rectangle([(20, 20), (width - 20, height - 20)], fill=(230, 245, 235, 255))
    # write prompt text (small)
    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except Exception:
        font = ImageFont.load_default()
    text = f"PROMPT: {prompt[:800]}{'...' if len(prompt) > 800 else ''}"
    draw.text((40, 40), text, fill=(60, 60, 60), font=font)
    return img


def run_image_model_auto1111(prompt: str, width: int = 1200, height: int = 400, steps: int = 20, seed: Optional[int] = None) -> Image.Image:
    """
    Call AUTOMATIC1111 txt2img API to get an image. Returns a PIL.Image.

    Requires Automatic1111 running locally at AUTOMATIC1111_API.
    """
    payload = {
        "prompt": prompt,
        "sampler_name": "Euler a",
        "steps": steps,
        "width": width,
        "height": height,
        "cfg_scale": 7.0,
    }
    if seed is not None:
        payload["seed"] = seed

    resp = requests.post(AUTO1111_TXT2IMG, json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    if not data.get("images"):
        raise RuntimeError("automatic1111 returned no images")
    img_b64 = data["images"][0]
    img_bytes = base64.b64decode(img_b64.split(",", 1)[-1] if "," in img_b64 else img_b64)
    img = Image.open(io.BytesIO(img_bytes)).convert("RGBA")
    return img


def run_image_model_diffusers(
    prompt: str,
    width: int = 1200,
    height: int = 400,
    steps: int = 20,
    seed: Optional[int] = None,
    model_id: Optional[str] = "stabilityai/stable-diffusion-xl-base-1.0",
    device: Optional[str] = None,
) -> Image.Image:
    """
    Generate an image with Diffusers (SDXL). Returns a PIL.Image.

    Requires model weights and appropriate environment (HF token, disk space, GPU).
    """
    if not _HAS_DIFFUSERS:
        raise RuntimeError("diffusers pipeline is not available in this environment")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    generator = None
    if seed is not None:
        gen = torch.Generator(device)
        gen.manual_seed(int(seed))
        generator = gen

    torch_dtype = torch.float16 if device == "cuda" else torch.float32

    pipe = StableDiffusionXLPipeline.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        requires_safety_checker=False,
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    if device == "cuda":
        try:
            if is_xformers_available():
                pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pass
        pipe.enable_attention_slicing()
        pipe.to("cuda")
    else:
        pipe.to("cpu")

    out = pipe(
        prompt,
        negative_prompt=None,
        width=width,
        height=height,
        num_inference_steps=steps,
        generator=generator,
    )
    img = out.images[0].convert("RGBA")
    return img


# ----------------------
# Composition (Pillow)
# ----------------------
def _choose_font_size(draw: ImageDraw.ImageDraw, text: str, max_width: int, base_font: str = None, max_font_size: int = 56, min_font_size: int = 12):
    """Return a PIL ImageFont with the largest size that fits max_width."""
    if base_font:
        for size in range(max_font_size, min_font_size - 1, -2):
            try:
                f = ImageFont.truetype(base_font, size)
            except Exception:
                f = ImageFont.load_default()
            w, _ = draw.textsize(text, font=f)
            if w <= max_width:
                return f
        return ImageFont.truetype(base_font, min_font_size)
    else:
        # fallback using default font
        for size in range(max_font_size, min_font_size - 1, -2):
            try:
                f = ImageFont.truetype("arial.ttf", size)
            except Exception:
                f = ImageFont.load_default()
            w, _ = draw.textsize(text, font=f)
            if w <= max_width:
                return f
        return ImageFont.load_default()


def compose_banner(
    image: Image.Image,
    headline: str,
    subtitle: Optional[str],
    cta_text: Optional[str],
    output_size: Tuple[int, int] = (1200, 400),
    align: str = "left",
    bg_padding: int = 40,
) -> Image.Image:
    """Overlay headline/subtitle/cta on the image. Returns a new PIL.Image."""
    target_w, target_h = output_size
    img = ImageOps.fit(image.convert("RGBA"), (target_w, target_h), method=Image.LANCZOS)
    draw = ImageDraw.Draw(img)

    text_area_w = int(target_w * 0.58)
    text_x = bg_padding if align == "left" else int((target_w - text_area_w) / 2)
    text_y = int(target_h * 0.2)

    font_head = _choose_font_size(draw, headline or "", max_width=text_area_w, max_font_size=48, min_font_size=20)
    font_sub = _choose_font_size(draw, subtitle or "", max_width=text_area_w, max_font_size=28, min_font_size=12)

    # Contrast overlay
    rect_w = text_area_w + 2 * 16
    rect_h = int(target_h * 0.35)
    overlay = Image.new("RGBA", (rect_w, rect_h), (255, 255, 255, 180))
    img_crop = img.crop((text_x, text_y, min(text_x + rect_w, target_w), min(text_y + rect_h, target_h))).convert("L")
    avg = sum(img_crop.getdata()) / (img_crop.size[0] * img_crop.size[1]) if img_crop.size[0] * img_crop.size[1] > 0 else 200
    if avg < 120:
        overlay = Image.new("RGBA", (rect_w, rect_h), (0, 0, 0, 140))

    img.paste(overlay, (text_x - 16, text_y - 12), overlay)
    draw = ImageDraw.Draw(img)

    headline_color = (20, 35, 20) if avg > 120 else (255, 255, 255)
    draw.text((text_x, text_y), headline or "", font=font_head, fill=headline_color)

    if subtitle:
        sub_y = text_y + int(getattr(font_head, "size", 24) * 1.4)
        subtitle_color = (40, 40, 40) if avg > 120 else (230, 230, 230)
        draw.text((text_x, sub_y), subtitle, font=font_sub, fill=subtitle_color)

    if cta_text:
        try:
            cta_w, cta_h = draw.textsize(cta_text, font=font_sub)
        except Exception:
            cta_w, cta_h = (len(cta_text) * 6, 20)
        pill_w = cta_w + 28
        pill_h = cta_h + 12
        pill_x = text_x
        pill_y = text_y + rect_h - pill_h - 12
        pill = Image.new("RGBA", (pill_w, pill_h), (20, 120, 60, 230))
        pill_draw = ImageDraw.Draw(pill)
        pill_draw.text((14, 6), cta_text, font=font_sub, fill=(255, 255, 255))
        img.paste(pill, (pill_x, pill_y), pill)

    return img


# ----------------------
# VLM review (Ollama)
# ----------------------
def vlm_review_image_text(image: Image.Image, headline: str, subtitle: Optional[str], cta: Optional[str], model: str = VLM_DEFAULT_MODEL) -> Dict:
    """
    Send the composed image and text to a Vision-Language model via Ollama for review.
    Returns: {"score":float, "blocked":bool, "comments":[...]}
    """
    try:
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        system_prompt = (
            "You are a creative review assistant for Union Bank. "
            "Given an image and the overlay text, check for: (1) relevance to product, "
            "(2) compliance issues (no promises/guarantees), (3) avoid calling out customer behavior counts, "
            "(4) legibility recommendations. Reply JSON with keys: score, blocked, comments."
        )
        user_prompt = (
            f"Image (base64) attached. Headline: {headline}. Subtitle: {subtitle or ''}. CTA: {cta or ''}.\n"
            "Return ONLY valid JSON: {\"score\": 0-1, \"blocked\": true/false, \"comments\": [..]}"
        )
        payload = {
            "model": model,
            "system": system_prompt,
            "prompt": user_prompt,
            "format": "json",
            "options": {"temperature": 0.0},
        }
        resp = requests.post(OLLAMA_URL, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        raw = data.get("response") or data.get("output") or json.dumps(data)
        try:
            review = json.loads(raw) if isinstance(raw, str) else raw
            # normalize
            score = float(review.get("score", 0.8))
            blocked = bool(review.get("blocked", False))
            comments = review.get("comments", []) or []
            return {"score": score, "blocked": blocked, "comments": comments}
        except Exception:
            # best-effort parse
            return {"score": 0.75, "blocked": False, "comments": ["vlm_parse_failed; assume OK"]}
    except Exception as e:
        # fallback if Ollama not available
        logging.debug(f"VLM review failed: {e}")
        return {"score": 0.7, "blocked": False, "comments": [f"vlm_error:{str(e)[:200]}"]}


# ----------------------
# Main orchestration: generate_creatives_for_variant
# ----------------------
def generate_creatives_for_variant(
    campaign_id: str,
    variant_tag: str,
    product_key: str,
    variant_brief: str,
    headline: str,
    subtitle: Optional[str],
    cta_text: Optional[str],
    n_images: int = 2,
    output_sizes: Optional[List[Tuple[int, int]]] = None,
    image_gen_fn: Optional[Callable] = None,
    image_model_choice: str = "stub",
    steps: int = 20,
    seed: Optional[int] = None,
    device: Optional[str] = None,
) -> Dict:
    """
    Generate creatives for a variant and return metadata.

    Returns:
      {"folder": <folder_path>, "meta": {...}}
    """
    if output_sizes is None:
        output_sizes = [(1200, 400), (800, 800), (600, 200)]

    folder = _unique_creative_folder(campaign_id, variant_tag)
    meta = {"campaign_id": campaign_id, "variant_tag": variant_tag, "created_at": int(time.time()), "items": []}

    # choose generator function
    def _select_image_fn():
        if image_gen_fn:
            return image_gen_fn
        if image_model_choice and "automatic1111" in image_model_choice.lower():
            return run_image_model_auto1111
        if image_model_choice and "diffuser" in image_model_choice.lower() or "sdxl" in image_model_choice.lower():
            return run_image_model_diffusers
        return run_image_model_stub

    gen_fn = _select_image_fn()

    prompts = generate_image_prompts(product_key, variant_brief, n_prompts=n_images)

    for idx, p in enumerate(prompts):
        # generate full-res image (target first output size)
        try:
            base_w, base_h = output_sizes[0]
            img = gen_fn(p, width=base_w, height=base_h, steps=steps, seed=seed, device=device)
        except TypeError:
            # fallback in case gen_fn signature differs
            try:
                img = gen_fn(p, width=output_sizes[0][0], height=output_sizes[0][1])
            except Exception:
                img = run_image_model_stub(p, width=output_sizes[0][0], height=output_sizes[0][1])
        except Exception:
            img = run_image_model_stub(p, width=output_sizes[0][0], height=output_sizes[0][1])

        item = {"prompt": p, "variants": []}

        for size in output_sizes:
            try:
                composed = compose_banner(img, headline=headline, subtitle=subtitle, cta_text=cta_text, output_size=size)
            except Exception:
                # if compose fails, just resize
                composed = ImageOps.fit(img.convert("RGBA"), size, method=Image.LANCZOS)

            review = vlm_review_image_text(composed, headline=headline, subtitle=subtitle, cta=cta_text)

            size_slug = f"{size[0]}x{size[1]}"
            filename = os.path.join(folder, f"{variant_tag}_{idx}_{size_slug}.png")
            try:
                composed.convert("RGB").save(filename, quality=90)
            except Exception:
                # fallback: save as PNG bytes
                composed.save(filename, format="PNG")

            item["variants"].append({"size": size, "file": filename, "review": review})

        meta["items"].append(item)

    # write meta.json
    try:
        with open(os.path.join(folder, "meta.json"), "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2)
    except Exception as e:
        logging.debug(f"Failed to write meta.json: {e}")

    return {"folder": folder, "meta": meta}


# ----------------------
# If used as script: quick smoke test
# ----------------------
if __name__ == "__main__":
    out = generate_creatives_for_variant(
        campaign_id="camp_demo",
        variant_tag="A",
        product_key="ev_loan",
        variant_brief="sustainability and low-cost ownership",
        headline="Drive Green. Finance Electric.",
        subtitle="Union Green Vehicle Loans — attractive rates and fast approvals for EV buyers.",
        cta_text="Know More",
        n_images=1,
        image_gen_fn=run_image_model_stub,
    )
    print("Generated folder:", out["folder"])
    print("Meta items:", len(out["meta"]["items"]))
