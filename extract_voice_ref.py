"""
Extracts clean reference audio from the user's dataset to clone the authentic Kenyan Gen Z voice.
"""
import soundfile as sf
import numpy as np
from pathlib import Path
from pydub import AudioSegment

BASE_DIR = Path(__file__).resolve().parent
ASSETS_DIR = BASE_DIR / "assets"
ASSETS_DIR.mkdir(parents=True, exist_ok=True)

SOURCE_AUDIO = Path("/home/ray/Music/Sheng_dataset/sheng0001.wav")
FALLBACK_SOURCE = Path("/home/ray/Music/Sheng_dataset/Recording 3.flac")
OUT_REF = ASSETS_DIR / "kenyan_genz_voice_ref.wav"


def extract_reference():
    src = SOURCE_AUDIO if SOURCE_AUDIO.exists() else FALLBACK_SOURCE
    if not src.exists():
        print(f"Source audio {src} not found.")
        return

    print(f"Extracting reference speaker clip from {src}...")
    audio = AudioSegment.from_file(str(src))
    
    # Extract clean 10 seconds of speech (from second 2 to second 12)
    sample = audio[2000:12000]
    sample = sample.set_channels(1).set_frame_rate(24000).set_sample_width(2)
    
    # Normalize volume
    change_in_db = -18.0 - sample.dBFS
    sample = sample.apply_gain(change_in_db)
    
    sample.export(str(OUT_REF), format="wav")
    print(f"✅ Extracted authentic Kenyan Gen Z reference voice to: {OUT_REF}")


if __name__ == "__main__":
    extract_reference()
