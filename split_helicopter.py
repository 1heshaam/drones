from pathlib import Path
import shutil
import random

raw_dir = Path.home() / "Desktop" / "raw_helicopter"

train_dir = Path("dataset/train/helicopter")
val_dir = Path("dataset/val/helicopter")

train_dir.mkdir(parents=True, exist_ok=True)
val_dir.mkdir(parents=True, exist_ok=True)

allowed_exts = {
    ".jpg", ".jpeg", ".png", ".webp", ".bmp", ".jfif", ".avif",
    ".JPG", ".JPEG", ".PNG", ".WEBP", ".BMP", ".JFIF", ".AVIF"
}

images = [
    p for p in raw_dir.rglob("*")
    if p.is_file() and p.suffix in allowed_exts
]

print(f"Looking in: {raw_dir}")
print(f"Found {len(images)} images")

if len(images) == 0:
    print("Files found in folder:")
    for p in raw_dir.rglob("*"):
        print(p)
    raise FileNotFoundError("No supported image files found inside Desktop/raw_helicopter.")

random.shuffle(images)

split_index = int(len(images) * 0.8)

train_images = images[:split_index]
val_images = images[split_index:]

for i, src in enumerate(train_images):
    shutil.copy(src, train_dir / f"helicopter_train_{i:04d}{src.suffix.lower()}")

for i, src in enumerate(val_images):
    shutil.copy(src, val_dir / f"helicopter_val_{i:04d}{src.suffix.lower()}")

print(f"Copied {len(train_images)} helicopter images to train")
print(f"Copied {len(val_images)} helicopter images to val")