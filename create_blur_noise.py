from pathlib import Path
from PIL import Image, ImageFilter
import random

source_dirs = [
    Path("dataset/train/bird"),
    Path("dataset/val/bird"),
    Path("dataset/train/airplane"),
    Path("dataset/val/airplane"),
    Path("dataset/train/helicopter"),
    Path("dataset/val/helicopter"),
    Path("dataset/train/cloud_empty"),
    Path("dataset/val/cloud_empty"),
]

train_out = Path("dataset/train/blur_noise")
val_out = Path("dataset/val/blur_noise")
train_out.mkdir(parents=True, exist_ok=True)
val_out.mkdir(parents=True, exist_ok=True)

allowed_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".jfif"}

images = []
for d in source_dirs:
    if d.exists():
        for p in d.rglob("*"):
            if p.is_file() and p.suffix.lower() in allowed_exts:
                images.append(p)

random.shuffle(images)

def make_blur(src, dst):
    img = Image.open(src).convert("RGB")
    img = img.resize((224, 224))
    img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(4, 10)))
    img.save(dst)

train_count = min(200, int(len(images) * 0.75))
val_count = min(60, len(images) - train_count)

for i, src in enumerate(images[:train_count]):
    make_blur(src, train_out / f"blur_train_{i:04d}.jpg")

for i, src in enumerate(images[train_count:train_count + val_count]):
    make_blur(src, val_out / f"blur_val_{i:04d}.jpg")

print(f"Created {train_count} train blur/noise images")
print(f"Created {val_count} val blur/noise images")