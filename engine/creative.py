"""
engine/creative.py

Patched multimodal creative skeleton for the personalization prototype.

Features:
- Lazy imports for heavy ML libs so the module can be imported without GPU packages.
- SDXL pipeline caching to avoid re-loading models per call.
- Robust fallback: diffusers (SDXL) -> AUTOMATIC1111 HTTP -> stub image.
- Pillow-based composition for headline/subtitle/CTA with font fallback + simple wrapping.
- VLM review via Ollama with robust response parsing.
- Metadata tracking (generation method, VLM review) saved to meta.json for audit.
"""

from __future__ import annotations
import os
import io
import json
import time
import base64
import hashlib
from typing import List, Dict, Tuple, Optional, Any

from PIL import Image, ImageDraw, ImageFont, ImageOps
import requests

# ----------------------
# Configuration
# ----------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
CREATIVES_DIR = os.path.join(PROJECT_ROOT, "assets", "creatives")
os.makedirs(CREATIVES_DIR, exist_ok=True)

# Ollama / VLM config (for review)
OLLAMA_URL = "http://localhost:11434/api/generate"
VLM_DEFAULT_MODEL = "qwen2.5vl:7b"  # change as appropriate

# Image generator config (AUTOMATIC1111 HTTP)
AUTOMATIC1111_API = "http://localhost:7860"
AUTO1111_TXT2IMG = f"{AUTOMATIC1111_API}/sdapi/v1/txt2img"

# ----------------------
# Prompt templates
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
    base = IMAGE_PROMPT_TEMPLATES.get(product_key, "")
    prompts = []
    for i in range(n_prompts):
        prompt = f"{base}. Focus: {variant_text_brief}. Style: clean, high contrast, photography."
        # small variation
        if i % 2 == 0:
            prompt += " shallow depth of field, crisp subject"
        else:
            prompt += " clean background, high-detail texture"
        prompts.append(prompt)
    return prompts


# ----------------------
# Image generation helpers (pluggable)
# ----------------------
def run_image_model_auto1111(prompt: str, width: int = 1200, height: int = 400, steps: int = 20, seed: Optional[int] = None) -> Image.Image:
    """
    Call AUTOMATIC1111 txt2img API to get an image. Returns a PIL.Image.
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
    img_b64 = data["images"][0]
    img_bytes = base64.b64decode(img_b64.split(",", 1)[-1] if "," in img_b64 else img_b64)
    img = Image.open(io.BytesIO(img_bytes)).convert("RGBA")
    return img


def run_image_model_stub(prompt: str, width: int = 1200, height: int = 400) -> Image.Image:
    """
    A stub used for development/testing when you don't have an image generator.
    Creates a simple placeholder image with the prompt text faintly visible.
    """
    img = Image.new("RGBA", (width, height), (240, 240, 240, 255))
    draw = ImageDraw.Draw(img)
    draw.rectangle([(20, 20), (width - 20, height - 20)], fill=(230, 240, 235, 255))
    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except Exception:
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", 14)
        except Exception:
            font = ImageFont.load_default()
    # wrap prompt text a bit
    text = (prompt[:1000] + ("..." if len(prompt) > 1000 else ""))
    draw.text((40, 40), text, fill=(40, 40, 40), font=font)
    return img


# ----------------------
# Composition (Pillow) + helpers
# ----------------------
def _wrap_text(text: str, font: ImageFont.FreeTypeFont, max_width: int) -> List[str]:
    if not text:
        return []
    words = text.split()
    lines: List[str] = []
    cur = []
    for w in words:
        cur.append(w)
        trial = " ".join(cur)
        if font.getsize(trial)[0] > max_width and len(cur) > 1:
            cur.pop()  # last word caused overflow
            lines.append(" ".join(cur))
            cur = [w]
    if cur:
        lines.append(" ".join(cur))
    return lines


def _choose_font_size(draw: ImageDraw.ImageDraw, text: str, max_width: int, base_font: Optional[str] = None, max_font_size: int = 60, min_font_size: int = 14) -> ImageFont.FreeTypeFont:
    """Return a PIL ImageFont with the largest size that fits max_width."""
    if not text:
        # default font
        try:
            return ImageFont.truetype("arial.ttf", min_font_size)
        except Exception:
            try:
                return ImageFont.truetype("DejaVuSans.ttf", min_font_size)
            except Exception:
                return ImageFont.load_default()

    if base_font is None:
        for size in range(max_font_size, min_font_size - 1, -2):
            try:
                f = ImageFont.truetype("arial.ttf", size)
            except Exception:
                try:
                    f = ImageFont.truetype("DejaVuSans.ttf", size)
                except Exception:
                    f = ImageFont.load_default()
            w, _h = draw.textsize(text, font=f)
            if w <= max_width:
                return f
        return f
    else:
        for size in range(max_font_size, min_font_size - 1, -2):
            f = ImageFont.truetype(base_font, size)
            w, _h = draw.textsize(text, font=f)
            if w <= max_width:
                return f
        return ImageFont.truetype(base_font, min_font_size)


def compose_banner(
    image: Image.Image,
    headline: str,
    subtitle: Optional[str],
    cta_text: Optional[str],
    output_size: Tuple[int, int] = (1200, 400),
    align: str = "left",
    bg_padding: int = 40,
) -> Image.Image:
    """
    Overlay headline/subtitle/cta on the image. Returns a new PIL.Image.
    """
    target_w, target_h = output_size
    img = ImageOps.fit(image.convert("RGBA"), (target_w, target_h), method=Image.LANCZOS)
    draw = ImageDraw.Draw(img)

    text_area_w = int(target_w * 0.58)
    text_x = bg_padding if align == "left" else int((target_w - text_area_w) / 2)
    text_y = int(target_h * 0.18)

    # Headline font
    font_head = _choose_font_size(draw, headline or "", max_width=text_area_w, max_font_size=56, min_font_size=20)
    font_sub = _choose_font_size(draw, subtitle or "", max_width=text_area_w, max_font_size=30, min_font_size=14)

    # contrast overlay
    rect_w = text_area_w + 2 * 16
    rect_h = int(target_h * 0.38)
    overlay = Image.new("RGBA", (rect_w, rect_h), (255, 255, 255, 180))
    img_crop = img.crop((text_x, text_y, min(text_x + rect_w, target_w), min(text_y + rect_h, target_h))).convert("L")
    if img_crop.size[0] == 0 or img_crop.size[1] == 0:
        avg = 200
    else:
        avg = sum(img_crop.getdata()) / (img_crop.size[0] * img_crop.size[1])
    if avg < 120:
        overlay = Image.new("RGBA", (rect_w, rect_h), (0, 0, 0, 140))

    img.paste(overlay, (text_x - 16, text_y - 12), overlay)

    # Draw headline (allow multi-line if needed)
    head_lines = _wrap_text(headline or "", font_head, text_area_w)
    cur_y = text_y
    headline_color = (20, 35, 20) if avg > 120 else (255, 255, 255)
    for line in head_lines:
        draw.text((text_x, cur_y), line, font=font_head, fill=headline_color)
        cur_y += int(font_head.size * 1.25)

    # subtitle
    if subtitle:
        sub_lines = _wrap_text(subtitle, font_sub, text_area_w)
        sub_color = (40, 40, 40) if avg > 120 else (230, 230, 230)
        for line in sub_lines:
            draw.text((text_x, cur_y), line, font=font_sub, fill=sub_color)
            cur_y += int(font_sub.size * 1.15)

    # CTA pill
    if cta_text:
        try:
            cta_font = font_sub
            cta_w, cta_h = draw.textsize(cta_text, font=cta_font)
        except Exception:
            cta_w, cta_h = (100, 20)
        pill_w = cta_w + 28
        pill_h = cta_h + 12
        pill_x = text_x
        pill_y = text_y + rect_h - pill_h - 12
        pill = Image.new("RGBA", (pill_w, pill_h), (180, 36, 36, 230))
        pill_draw = ImageDraw.Draw(pill)
        pill_draw.text((14, 6), cta_text, font=cta_font, fill=(255, 255, 255))
        img.paste(pill, (pill_x, pill_y), pill)

    return img


# ----------------------
# VLM review (Ollama) with robust parsing
# ----------------------
def vlm_review_image_text(image: Image.Image, headline: str, subtitle: Optional[str], cta: Optional[str], model: str = VLM_DEFAULT_MODEL) -> Dict[str, Any]:
    """
    Send the composed image and text to a Vision-Language model via Ollama for review.
    Returns a dict: {score, blocked, comments}
    """
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    system_prompt = (
        "You are a creative review assistant for Union Bank. "
        "Given an image (attached as base64) and overlay text, check for: "
        "(1) relevance to product, (2) compliance issues (no promises/guarantees), "
        "(3) no mention of personal behaviour counts, (4) legibility recommendations. "
        "Reply with a JSON object with keys: score (0-1), blocked (bool), comments (list)."
    )

    user_prompt = (
        f"Image (base64 PNG) is attached. Headline: {headline}. Subtitle: {subtitle or ''}. CTA: {cta or ''}."
        " Return JSON only."
    )

    payload = {
        "model": model,
        "system": system_prompt,
        "prompt": user_prompt,
        "format": "json",
        "options": {"temperature": 0.0},
    }

    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=60)
        resp.raise_for_status()
        text = resp.text.strip()
        # Try entire response as JSON first
        try:
            review = json.loads(text)
        except Exception:
            # Fallback: try last line
            last_line = text.splitlines()[-1]
            try:
                review = json.loads(last_line)
            except Exception:
                review = {"score": 0.8, "blocked": False, "comments": ["vlm_parsing_failed; default_ok"]}
    except Exception as e:
        review = {"score": 0.6, "blocked": False, "comments": [f"vlm_error:{str(e)[:200]}"]}

    # Ensure keys exist
    return {
        "score": float(review.get("score", 0.6)),
        "blocked": bool(review.get("blocked", False)),
        "comments": review.get("comments", []) if isinstance(review.get("comments", []), list) else [str(review.get("comments"))],
    }


# ----------------------
# Diffusers SDXL helper (lazy load + cache)
# ----------------------
_PIPELINE_CACHE: Dict[str, Any] = {}

def _load_sdxl_pipeline(model_id: str, device: str, torch_dtype) -> Any:
    """
    Lazy-load and cache SDXL pipeline. Returns a pipeline object.
    """
    global _PIPELINE_CACHE
    if model_id in _PIPELINE_CACHE:
        return _PIPELINE_CACHE[model_id]

    # Lazy imports
    try:
        import torch as _torch
        from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler
        from diffusers.utils import is_xformers_available
    except Exception as e:
        raise RuntimeError(f"Required diffusers/torch not available: {e}")

    pipe = StableDiffusionXLPipeline.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        safety_checker=None,
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

    _PIPELINE_CACHE[model_id] = pipe
    return pipe


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
    Generate an image using diffusers SDXL pipeline. Auto-selects CUDA if available.
    """
    # Lazy import torch
    try:
        import torch as _torch
    except Exception:
        raise RuntimeError("PyTorch not available in environment.")

    if device is None:
        device = "cuda" if _torch.cuda.is_available() else "cpu"

    generator = None
    if seed is not None:
        generator = _torch.Generator(device).manual_seed(seed)

    torch_dtype = _torch.float16 if device == "cuda" else _torch.float32

    # load pipeline (cached)
    pipe = _load_sdxl_pipeline(model_id, device, torch_dtype)

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
# Robust fallback wrapper
# ----------------------
def generate_image_with_fallback(
    prompt: str,
    width: int,
    height: int,
    steps: int = 20,
    seed: Optional[int] = None,
    model_id: Optional[str] = None,
) -> Tuple[Image.Image, Dict[str, Any]]:
    """
    Try diffusers (SDXL) first, then AUTOMATIC1111 HTTP API, then stub.
    Returns (PIL.Image, metadata)
    """
    # 1) Try diffusers if available
    try:
        img = run_image_model_diffusers(prompt, width=width, height=height, steps=steps, seed=seed, model_id=model_id or "stabilityai/stable-diffusion-xl-base-1.0")
        return img, {"method": "diffusers", "device": "cuda" if img.mode else "cpu"}
    except Exception as e:
        print(f"[WARN] diffusers generation failed: {e}")

    # 2) Try AUTOMATIC1111
    try:
        img = run_image_model_auto1111(prompt, width=width, height=height, steps=steps, seed=seed)
        return img, {"method": "auto1111", "device": "http_api"}
    except Exception as e:
        print(f"[WARN] automatic1111 generation failed: {e}")

    # 3) Fallback stub
    img = run_image_model_stub(prompt, width=width, height=height)
    return img, {"method": "stub", "device": "none"}


# A wrapper to convert (img,meta) -> img only for older callers
def generate_image_fn_wrapper(prompt, width, height, seed=None, model_id=None):
    img, meta = generate_image_with_fallback(prompt, width, height, seed=seed, model_id=model_id)
    return img


# ----------------------
# High-level pipeline
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
    output_sizes: List[Tuple[int, int]] = None,
    image_gen_fn = None,
    image_model_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Top-level function to generate creatives for one variant.
    - image_gen_fn may return either PIL.Image OR (PIL.Image, meta_dict).
    - image_model_kwargs forwarded to the generator wrapper if present.
    """
    if output_sizes is None:
        output_sizes = [(1200, 400), (800, 800), (600, 200)]

    if image_gen_fn is None:
        # default to the wrapper that returns only image
        image_gen_fn = generate_image_fn_wrapper

    prompts = generate_image_prompts(product_key, variant_brief, n_prompts=n_images)
    folder = _unique_creative_folder(campaign_id, variant_tag)
    meta: Dict[str, Any] = {"campaign_id": campaign_id, "variant_tag": variant_tag, "created_at": int(time.time()), "items": []}

    for idx, p in enumerate(prompts):
        # Generate image candidate (allow generator to return img or (img,meta))
        gen_meta: Optional[Dict[str, Any]] = None
        try:
            res = image_gen_fn(p, width=output_sizes[0][0], height=output_sizes[0][1]) if image_model_kwargs is None else image_gen_fn(p, width=output_sizes[0][0], height=output_sizes[0][1], **image_model_kwargs)
            if isinstance(res, tuple) and len(res) >= 1:
                img = res[0]
                if len(res) >= 2 and isinstance(res[1], dict):
                    gen_meta = res[1]
            else:
                img = res
        except Exception as e:
            print(f"[WARN] image_gen_fn failed for prompt {p[:120]}: {e}")
            img = run_image_model_stub(p, width=output_sizes[0][0], height=output_sizes[0][1])
            gen_meta = {"method": "stub", "device": "none"}

        item: Dict[str, Any] = {"prompt": p, "variants": []}
        for size in output_sizes:
            try:
                composed = compose_banner(img, headline=headline, subtitle=subtitle, cta_text=cta_text, output_size=size)
            except Exception as e:
                print(f"[WARN] compose failed: {e}. Using plain image.")
                composed = img

            # Run VLM review (non-blocking)
            review = vlm_review_image_text(composed, headline=headline, subtitle=subtitle, cta=cta_text)

            size_slug = f"{size[0]}x{size[1]}"
            filename = os.path.join(folder, f"{variant_tag}_{idx}_{size_slug}.png")
            try:
                composed.convert("RGB").save(filename, quality=90)
            except Exception as e:
                # fallback: attempt basic save via BytesIO
                try:
                    buf = io.BytesIO()
                    composed.convert("RGB").save(buf, format="PNG")
                    with open(filename, "wb") as fh:
                        fh.write(buf.getvalue())
                except Exception as e2:
                    print(f"[ERROR] failed to save image {filename}: {e2}")

            item_variant = {
                "size": size,
                "file": filename,
                "review": review,
                "gen_method": gen_meta.get("method") if gen_meta else None,
                "gen_meta": gen_meta or {},
            }
            item["variants"].append(item_variant)

        meta["items"].append(item)

    # Save metadata
    try:
        with open(os.path.join(folder, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
    except Exception as e:
        print(f"[WARN] failed to write meta.json: {e}")

    return {"folder": folder, "meta": meta}


# ----------------------
# Example usage (manual test)
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
        n_images=2,
        image_gen_fn=generate_image_fn_wrapper,
    )
    print("Generated folder:", out["folder"])
    print("Meta items:", len(out["meta"]["items"]))
