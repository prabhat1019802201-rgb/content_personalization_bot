# engine/settings.py
"""
Simple mapping for UI model selection strings -> internal identifiers (backend, device).
Extend as you add new backends.
"""

def map_image_choice(ui_choice: str):
    """
    Map UI choice (e.g., "Automatic1111 (http://localhost:7860)") to a backend name and device.
    Returns dict: {"backend": <str>, "device": <"cuda"|"cpu"|None>}
    """
    if not ui_choice:
        return {"backend": "stub", "device": None}

    s = ui_choice.lower()

    if "automatic1111" in s or "auto1111" in s or "automatic1111" in ui_choice:
        return {"backend": "auto1111", "device": None}
    if "diffusers" in s or "sdxl" in s:
        # prefer cuda if available; pipeline will pass device 'cuda' or 'cpu'
        return {"backend": "sdxl", "device": "cuda"}
    if "no image" in s or "text only" in s:
        return {"backend": "none", "device": None}
    # default stub
    return {"backend": "stub", "device": None}
