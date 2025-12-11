# test/create_sample_image.py
from PIL import Image, ImageDraw, ImageFont
img = Image.new("RGB", (800, 300), color=(240,240,240))
d = ImageDraw.Draw(img)
try:
    f = ImageFont.truetype("DejaVuSans.ttf", 28)
except Exception:
    f = ImageFont.load_default()
d.text((30,120), "Union Green Vehicle Loans", font=f, fill=(30,80,30))
img.save("test_banner.png")
print("Saved test_banner.png")

