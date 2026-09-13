# Nairobi Bot: Swahili & Sheng S2S Agent Integration

## What It Is
This project is an end-to-end Speech-to-Speech (S2S) AI agent designed to understand and speak Kenyan Sheng naturally. Dubbed "Nairobi Bot", the pipeline chains together three core microservices: an Automatic Speech Recognition (ASR) engine tuned for local dialects (Faster-Whisper), a Sheng-prompted Large Language Model (LLM) for natural conversation, and a Text-to-Speech (TTS) engine (Edge-TTS) that dynamically adjusts prosody. It processes audio input and returns a conversational voice response, bridging the gap between standard AI models and authentic Nairobi street culture.

## Quick-Start
1. Ensure your environment is active: `conda activate tts_env`
2. Start the Gradio interface: `python app.py`
3. Open your browser to `http://localhost:7860`
4. Click the microphone icon, speak a 3 to 5-second prompt, and await the audio reply.

## 6 Demo Scenarios
1. **Greeting:** Casual check-in on the neighborhood vibe.
2. **Hustle:** Talking about work, money, and the daily grind.
3. **Kibanda:** Asking for cheap, local lunch recommendations.
4. **Nganya:** Navigating matatu routes and stages.
5. **Weekend:** Planning for weekend parties and turn-ups.
6. **Drip:** Hyping up fashion and new shoes.
*(See `DEMO_SCRIPT.md` for exact prompts).*

## Known Limitations (Honest Assessment)
* **Hardware Latency:** The Whisper ASR model is currently running on CPU (INT8). Transcribing a 60-second audio clip takes ~30 seconds. For live demos, user audio inputs **must** be kept under 5 seconds to achieve the < 5000ms glass-to-glass latency target.
* **Audio Post-Processing Bypassed:** Due to CPU instruction set conflicts (core dumps), the `pedalboard` DSP library (compressor/reverb) was bypassed. The TTS output is functional and clear, but slightly drier than a true phone-call simulation.
* **LLM API Dependency:** If the `LLM_BACKEND` fails to connect to the external API, the system automatically falls back to a 1ms heuristic (hardcoded) text response to keep the pipeline moving.
