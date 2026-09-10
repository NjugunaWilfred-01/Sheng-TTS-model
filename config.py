"""
Configuration settings for the Swahili & Sheng S2S pipeline.
"""
import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent
AUDIO_TEMP_DIR = BASE_DIR / "temp_audio"
AUDIO_TEMP_DIR.mkdir(parents=True, exist_ok=True)

# ASR Settings
WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "small")
ASR_LANGUAGE = "sw"  # Swahili language code for Whisper
ASR_DEVICE = "cuda" if os.getenv("USE_CUDA", "false").lower() == "true" else "cpu"
ASR_COMPUTE_TYPE = "float16" if ASR_DEVICE == "cuda" else "int8"

# TTS Settings (Kenyan Neural Voices with Authentic Nairobi Prosody)
DEFAULT_VOICE = "sw-KE-RafikiNeural"  # Native Kenyan Swahili & Sheng Voice (Male)
DEFAULT_RATE = "+0%"   # Natural human conversational speed
DEFAULT_PITCH = "+0Hz" # Pure human vocal resonance without pitch distortion

VOICE_OPTIONS = {
    "Kenyan Male (Rafiki - Authentic Sheng & Swahili)": "sw-KE-RafikiNeural",
    "Kenyan Female (Zuri - Warm & Natural Swahili)": "sw-KE-ZuriNeural",
    "Nairobi Gen Z Guy (Chilemba - Urban English/Sheng)": "en-KE-ChilembaNeural",
    "Nairobi Gen Z Lady (Asilia - Urban English/Sheng)": "en-KE-AsiliaNeural"
}

# LLM Settings
LLM_BACKEND = os.getenv("LLM_BACKEND", "heuristic")  # "heuristic", "lora", "openai", "gemini", "ollama"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")

# Web Server Settings
GRADIO_SERVER_PORT = int(os.getenv("PORT", "7860"))
GRADIO_SERVER_NAME = os.getenv("HOST", "0.0.0.0")
