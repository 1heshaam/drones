from pathlib import Path
import shutil
import random

# Change this if your extracted CUB folder is somewhere else
cub_images_dir = Path.home() / "Desktop" / "images"

train_dir = Path("dataset/train/bird")
val_dir = Path("dataset/val/bird")

train_dir.mkdir(parents=True, exist_ok=True)
val_dir.mkdir(parents=True, exist_ok=True)

if not cub_images_dir.exists():
    raise FileNotFoundError(f"CUB images folder not found: {cub_images_dir}")

image_paths = list(cub_images_dir.rglob("*.jpg"))

if not image_paths:
    raise FileNotFoundError(f"No .jpg images found inside: {cub_images_dir}")

random.shuffle(image_paths)

train_count = 300
val_count = 80

train_images = image_paths[:train_count]
val_images = image_paths[train_count:train_count + val_count]

for i, src in enumerate(train_images):
    dst = train_dir / f"bird_train_{i:04d}.jpg"
    shutil.copy(src, dst)

for i, src in enumerate(val_images):
    dst = val_dir / f"bird_val_{i:04d}.jpg"
    shutil.copy(src, dst)

print(f"Copied {len(train_images)} train bird images")
print(f"Copied {len(val_images)} val bird images")