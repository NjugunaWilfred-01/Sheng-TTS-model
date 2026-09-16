"""
Configuration settings for the Swahili & Sheng S2S pipeline.
"""
import os
import sys
from pathlib import Path

# Force UTF-8 on stdout/stderr.
#
# Every entry point imports config, so this is the one choke point that covers them
# all. Windows consoles default to cp1252, and the emoji in our log/status lines
# ("🇰🇪", "OK", arrows) raise UnicodeEncodeError on the very first
# print -- app.py, cli.py and test_pipeline.py all died at startup before this.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass  # already UTF-8, or a stream that does not support reconfigure

# Paths
BASE_DIR = Path(__file__).resolve().parent
AUDIO_TEMP_DIR = BASE_DIR / "temp_audio"
AUDIO_TEMP_DIR.mkdir(parents=True, exist_ok=True)

# ASR Settings
# large-v3-turbo: pruned 4-layer decoder, ~8x faster than large-v3 at near-identical
# accuracy, and far stronger than "small" on code-switched Sheng. This is the best
# accuracy-per-millisecond option for CPU-only inference. The ~1.6GB weights are
# downloaded and cached on first use -- run scripts/prefetch_models.py BEFORE the
# demo so this never happens on stage.
WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "large-v3-turbo")
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
# Default to the API path. The local 0.5B LoRA was trained on 150 samples and
# memorises rather than generalises; the heuristic engine is deterministic. Only a
# real instruct model holds a multi-turn Sheng conversation. ShengLLMEngine falls
# back to heuristic automatically if the key is missing or the call fails, so this
# default is safe even with no network.
LLM_BACKEND = os.getenv("LLM_BACKEND", "openai")  # "openai", "heuristic", "lora", "gemini", "ollama"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
# Groq by default: OpenAI-compatible, free tier, ~10x lower latency than gpt-4o-mini,
# which matters when the whole glass-to-glass budget is 3 seconds.
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.groq.com/openai/v1")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "llama-3.3-70b-versatile")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")

# Web Server Settings
GRADIO_SERVER_PORT = int(os.getenv("PORT", "7860"))
GRADIO_SERVER_NAME = os.getenv("HOST", "0.0.0.0")
