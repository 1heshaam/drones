"""Fetch the trained-classifier artifacts at build time on Render.

Reads MODEL_URL and FEATURE_INDEX_URL from the environment and downloads them
to the repo root if they are set. Missing or unset URLs are skipped silently
so the CLIP zero-shot mode can still build and run.

Downloads stream into a temporary file in the destination directory and are
atomically renamed into place on success, so a failed/aborted download never
leaves a partial file on disk.
"""
import os
import shutil
import sys
import tempfile
import urllib.request
from pathlib import Path


REPO = Path(__file__).resolve().parent

ARTIFACTS = [
    ("MODEL_URL", REPO / "normal_sky_model.pt"),
    ("FEATURE_INDEX_URL", REPO / "feature_index.pt"),
]

DOWNLOAD_TIMEOUT_SECONDS = 120


def download(url: str, dest: Path) -> None:
    print(f"[artifacts] downloading {dest.name} from {url}")

    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{dest.name}.",
        suffix=".part",
        dir=str(dest.parent),
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as out, urllib.request.urlopen(
            url, timeout=DOWNLOAD_TIMEOUT_SECONDS
        ) as resp:
            shutil.copyfileobj(resp, out, length=1 << 16)
        os.replace(tmp_path, dest)
    except BaseException:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass
        raise

    size = dest.stat().st_size
    print(f"[artifacts] saved {dest} ({size} bytes)")


def main() -> int:
    for env_var, dest in ARTIFACTS:
        url = os.environ.get(env_var, "").strip()
        if not url:
            print(f"[artifacts] {env_var} not set; skipping {dest.name}")
            continue
        if dest.exists():
            print(f"[artifacts] {dest.name} already present; skipping")
            continue
        try:
            download(url, dest)
        except Exception as exc:  # noqa: BLE001
            print(f"[artifacts] failed to download {dest.name}: {exc}")
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
