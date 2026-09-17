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
#
# "small" is the measured default for CPU. large-v3-turbo was tried and is NOT
# better here; on the six demo clips (same prompt, same decode settings):
#
#                   load    raw WER   after normalize()   per clip   RTF
#   small           2.5s      0.699         0.139           3.58s   1.09x
#   large-v3-turbo  5.1s      0.593         0.376          14.88s   4.52x
#
# Turbo wins on RAW transcription and still loses end-to-end, for two reasons:
#
#  1. It is 4.2x slower on CPU. At RTF 4.52 it cannot keep up with real time, and
#     the glass-to-glass budget is already blown.
#  2. ASR_CORRECTION_RULES were hand-written against the errors "small" makes on the
#     six demo clips, so turbo's different mistakes do not match them and it misses
#     the 0.139 figure. NOTE: that normalizer gain is itself demo-only -- on real
#     human-referenced audio zero rules fire for either model (AUDIT_RESULTS.md §11).
#     So reason 1, the 4.2x latency cost, is the one that actually decides this.
#
# On a GPU the latency argument disappears and turbo's better RAW WER (0.593 vs
# 0.699) should win. Re-run scripts/ab_prompt_bias.py there before switching.
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
# LoRA adapters, newest first. The engine takes the first that exists on disk.
# 1.5B trained 2026-09-17 on 2,500 slot-composed turns (train 0.152 / eval 0.163);
# the 0.5B was trained on 150 and memorised them. See AUDIT_RESULTS.md section 10
# for why neither is the recommended backend.
LORA_ADAPTER_CANDIDATES = [
    "llm_sheng_lora_output_1_5B/final_adapter",
    "llm_sheng_lora_output/final_adapter",
]
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")

# Web Server Settings
# GRADIO_SHARE=true publishes a public *.gradio.live tunnel so someone off this
# network can open the app. The link is unguessable but genuinely public and lives
# ~72h, so keep it off by default and only turn it on to hand the demo to someone.
GRADIO_SHARE = os.getenv("GRADIO_SHARE", "false").lower() == "true"
GRADIO_SERVER_PORT = int(os.getenv("PORT", "7860"))
GRADIO_SERVER_NAME = os.getenv("HOST", "0.0.0.0")
