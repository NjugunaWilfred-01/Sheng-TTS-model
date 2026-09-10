"""
Audio Denoising & Speech Enhancement Pipeline:
Extracts high-quality speech samples from /home/ray/Music/Sheng_dataset,
applies spectral noise reduction, high-pass filtering, and loudness normalization
to create crystal-clean reference voices for authentic Kenyan Voice Cloning.
"""
import sys
import logging
from pathlib import Path
import numpy as np
import soundfile as sf
import librosa
import noisereduce as nr
from pydub import AudioSegment, effects

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("VoiceDenoise")

ASSETS_DIR = BASE_DIR / "assets"
ASSETS_DIR.mkdir(parents=True, exist_ok=True)

RAW_DATASET_DIR = Path("/home/ray/Music/Sheng_dataset")
CLEAN_REF_OUT = ASSETS_DIR / "kenyan_genz_clean_ref.wav"


def denoise_and_enhance(audio_path: str, output_path: str, start_sec: float = 2.0, duration_sec: float = 10.0):
    logger.info(f"Loading audio from: {audio_path}...")
    
    # 1. Load audio with librosa at 24000Hz (optimal for neural voice synthesis)
    sr = 24000
    y, _ = librosa.load(audio_path, sr=sr, mono=True, offset=start_sec, duration=duration_sec)

    # 2. Spectral Noise Reduction
    logger.info("Applying spectral noise reduction to strip background hiss/urban noise...")
    reduced_noise = nr.reduce_noise(
        y=y,
        sr=sr,
        prop_decrease=0.85,
        stationary=True,
        time_mask_smooth_ms=64
    )

    # 3. High-Pass Filter (removes mic rumble < 80Hz)
    from scipy.signal import butter, lfilter
    def butter_highpass_filter(data, cutoff=80, fs=sr, order=5):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='high', analog=False)
        return lfilter(b, a, data)

    filtered_audio = butter_highpass_filter(reduced_noise)

    # 4. Loudness Normalization to -16 dBFS
    max_amp = np.max(np.abs(filtered_audio))
    if max_amp > 0:
        normalized_audio = filtered_audio / max_amp * 0.90
    else:
        normalized_audio = filtered_audio

    # 5. Save enhanced WAV
    sf.write(output_path, normalized_audio, sr)
    logger.info(f"✅ Crystal-clean Kenyan Gen Z reference voice saved to: {output_path}")
    return output_path


def build_clean_voice_profile():
    # Candidates in user's dataset
    candidates = [
        RAW_DATASET_DIR / "sheng0001.wav",
        RAW_DATASET_DIR / "Recording 3.flac",
        RAW_DATASET_DIR / "Sheng0002.flac"
    ]

    for candidate in candidates:
        if candidate.exists():
            try:
                denoise_and_enhance(str(candidate), str(CLEAN_REF_OUT), start_sec=2.0, duration_sec=10.0)
                return str(CLEAN_REF_OUT)
            except Exception as e:
                logger.error(f"Failed to process {candidate}: {e}")

    logger.error("No valid audio candidate found in dataset.")
    return None


if __name__ == "__main__":
    build_clean_voice_profile()
