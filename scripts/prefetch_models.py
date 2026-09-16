"""
prefetch_models.py - Download and cache the Whisper weights before the demo.

WHISPER_MODEL_SIZE now defaults to large-v3-turbo (~1.6GB). faster-whisper fetches
it lazily on first transcribe, so without this script the download happens the first
time someone clicks Record -- on stage, on venue wifi. Run it the night before.

Usage:
    python scripts/prefetch_models.py
    python scripts/prefetch_models.py --model small     # override
"""
import argparse
import os
import sys
import time
from pathlib import Path

# Must be set BEFORE huggingface_hub is imported. Its Xet transfer backend failed
# here mid-download with "CAS Client Error: Format error: I/O error: error decoding
# response body", leaving a partial cache and no usable model. The plain HTTP path
# completed the same download without trouble. Override by exporting the var yourself.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import WHISPER_MODEL_SIZE, ASR_DEVICE, ASR_COMPUTE_TYPE  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=WHISPER_MODEL_SIZE)
    args = ap.parse_args()

    from faster_whisper import WhisperModel

    print(f"Fetching '{args.model}' for {ASR_DEVICE} ({ASR_COMPUTE_TYPE})...")
    print("First run downloads the weights; later runs just verify the cache.")
    start = time.time()
    model = WhisperModel(args.model, device=ASR_DEVICE, compute_type=ASR_COMPUTE_TYPE)
    print(f"Loaded in {time.time() - start:.1f}s.")

    # A real decode, so a corrupt or partial download fails HERE and not on stage.
    ref = Path(__file__).resolve().parent.parent / "assets" / "demo_samples" / "demo_5_work.mp3"
    if ref.exists():
        start = time.time()
        segments, info = model.transcribe(str(ref), language="sw", beam_size=5)
        text = " ".join(s.text.strip() for s in segments).strip()
        print(f"Smoke test ({ref.name}, {info.duration:.1f}s audio): "
              f"{time.time() - start:.1f}s -> '{text[:70]}'")
    else:
        print(f"No sample at {ref} -- skipping smoke test. "
              f"Run scripts/generate_demo_audio.py first.")

    print("\nModel is cached. The demo will not download anything.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
