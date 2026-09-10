"""
TTS Engine: Fast Kenyan Neural Voice synthesis via Edge-TTS.
"""
import time
import asyncio
import logging
from pathlib import Path
from typing import Dict, Any, Tuple

from config import DEFAULT_VOICE, DEFAULT_RATE, DEFAULT_PITCH, AUDIO_TEMP_DIR, VOICE_OPTIONS
from sheng_lexicon import ShengAcousticPhonetics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ShengTTSEngine")


class ShengTTSEngine:
    def __init__(self, default_voice: str = DEFAULT_VOICE, default_rate: str = DEFAULT_RATE, default_pitch: str = DEFAULT_PITCH):
        self.default_voice = default_voice
        self.default_rate = default_rate
        self.default_pitch = default_pitch

    async def synthesize_async(
        self,
        text: str,
        output_path: Optional_Path = None,
        voice: str = None,
        rate: str = None,
        pitch: str = None
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Synthesizes text to speech using Kenyan Neural voices asynchronously with Gen Z prosody.
        """
        import edge_tts

        selected_voice = voice or self.default_voice
        selected_rate = rate or self.default_rate
        selected_pitch = pitch or self.default_pitch
        if not output_path:
            filename = f"tts_output_{int(time.time() * 1000)}.mp3"
            output_path = str(AUDIO_TEMP_DIR / filename)

        start_time = time.time()
        try:
            # Format text into natural Sheng conversational phonetics and breath pauses
            cleaned_text = ShengAcousticPhonetics.format_for_speech(text.strip())
            if not cleaned_text:
                return "", {"error": "Empty text"}

            communicate = edge_tts.Communicate(
                text=cleaned_text,
                voice=selected_voice,
                rate=selected_rate,
                pitch=selected_pitch
            )
            await communicate.save(str(output_path))
            duration = time.time() - start_time

            metadata = {
                "voice": selected_voice,
                "output_path": str(output_path),
                "tts_time_ms": round(duration * 1000, 2),
                "text_length": len(cleaned_text)
            }
            logger.info(f"TTS synthesized in {metadata['tts_time_ms']}ms to {output_path}")
            return str(output_path), metadata

        except Exception as e:
            logger.error(f"TTS synthesis error: {e}")
            return "", {"error": str(e), "tts_time_ms": round((time.time() - start_time) * 1000, 2)}

    def synthesize(
        self,
        text: str,
        output_path: str = None,
        voice: str = None,
        rate: str = "+0%",
        pitch: str = "+0Hz"
    ) -> Tuple[str, Dict[str, Any]]:
        """Synchronous wrapper for synthesize_async."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If running inside an existing loop (e.g. Jupyter or Gradio async worker)
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    return pool.submit(
                        asyncio.run,
                        self.synthesize_async(text, output_path, voice, rate, pitch)
                    ).result()
            else:
                return loop.run_until_complete(
                    self.synthesize_async(text, output_path, voice, rate, pitch)
                )
        except RuntimeError:
            return asyncio.run(
                self.synthesize_async(text, output_path, voice, rate, pitch)
            )


Optional_Path = Any
