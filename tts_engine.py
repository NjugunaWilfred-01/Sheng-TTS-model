"""
TTS Engine: Fast Kenyan Neural Voice synthesis via Edge-TTS.
"""
import time
import asyncio
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

from config import DEFAULT_VOICE, DEFAULT_RATE, DEFAULT_PITCH, AUDIO_TEMP_DIR, VOICE_OPTIONS
from sheng_lexicon import ShengAcousticPhonetics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ShengTTSEngine")


class ShengTTSEngine:
    def __init__(
        self,
        default_voice: str = DEFAULT_VOICE,
        default_rate: str = DEFAULT_RATE,
        default_pitch: str = DEFAULT_PITCH,
        request_timeout_sec: float = 12.0,
        max_attempts: int = 3,
    ):
        self.default_voice = default_voice
        self.default_rate = default_rate
        self.default_pitch = default_pitch
        # 12s: comfortably above a healthy synthesis (~1.5-2.5s) and well under the
        # point where an audience notices the pipeline has died.
        self.request_timeout_sec = request_timeout_sec
        self.max_attempts = max_attempts

    async def synthesize_async(
        self,
        text: str,
        output_path: Optional[Path] = None,
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

            # Edge-TTS is a NETWORK service, not a local model, and it throttles.
            # Measured on three identical calls: 47s, 28s, 2.2s. An unbounded await
            # means one stalled request hangs the whole demo, so cap each attempt and
            # retry rather than waiting out a stall.
            for attempt in range(1, self.max_attempts + 1):
                communicate = edge_tts.Communicate(
                    text=cleaned_text,
                    voice=selected_voice,
                    rate=selected_rate,
                    pitch=selected_pitch
                )
                try:
                    await asyncio.wait_for(
                        communicate.save(str(output_path)),
                        timeout=self.request_timeout_sec,
                    )
                    break
                except Exception as e:  # includes asyncio.TimeoutError
                    if attempt == self.max_attempts:
                        raise
                    logger.warning(
                        f"Edge-TTS attempt {attempt}/{self.max_attempts} failed "
                        f"({type(e).__name__}); retrying."
                    )

            # Edge-TTS writes mp3. Round-trip through wav so pedalboard can polish it.
            if str(output_path).endswith(".mp3"):
                from pydub import AudioSegment
                wav_path = str(output_path).replace(".mp3", ".wav")
                AudioSegment.from_mp3(str(output_path)).export(wav_path, format="wav")
                self._polish(wav_path)
                AudioSegment.from_wav(wav_path).export(str(output_path), format="mp3")
                import os
                os.remove(wav_path)
            else:
                self._polish(str(output_path))

            # Must sit OUTSIDE the branch above: it used to be assigned only on the
            # mp3 path, so any other extension raised NameError, which the except
            # below swallowed into a silent empty-audio failure.
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

    def _polish(self, path: str):
        """Compress + tiny reverb so the voice doesn't sound like a vacuum recording."""
        try:
            from pedalboard import Pedalboard, Reverb, Compressor, HighpassFilter, LowShelfFilter
            from pedalboard.io import AudioFile
        except ImportError:
            return   # silently skip if pedalboard missing
        board = Pedalboard([
            HighpassFilter(cutoff_frequency_hz=80),
            LowShelfFilter(cutoff_frequency_hz=250, gain_db=2),
            Compressor(threshold_db=-18, ratio=3.0),
            Reverb(room_size=0.15, wet_level=0.05),
        ])
        with AudioFile(path) as f:
            audio = f.read(f.frames)
            sr = f.samplerate
        polished = board(audio, sr)
        with AudioFile(path, "w", sr, polished.shape[0]) as f:
            f.write(polished)   

