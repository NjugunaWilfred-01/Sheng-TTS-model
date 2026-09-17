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
        request_timeout_sec: float = 20.0,
        max_attempts: int = 2,
        cold_start_timeout_sec: float = 35.0,
    ):
        self.default_voice = default_voice
        self.default_rate = default_rate
        self.default_pitch = default_pitch
        # 20s with 2 attempts. Edge-TTS is a network service and its latency is a
        # property of YOUR link, not of this code: the same six clips measured
        # 1.3-3s on a good connection and 15-27s on a degraded one. An aggressive
        # 12s cap was worse than useless -- it killed slow-but-working requests and
        # retried, turning a ~14s call into 26.8s. Tune this on the machine that will
        # actually run the demo, not on a laptop; scripts/bench_tts.py measures it.
        self.request_timeout_sec = request_timeout_sec
        # The FIRST call is different: TLS handshake and service wake-up measured 19.1s
        # here, against 1.4s and 1.3s for the two calls after it. A flat 12s timeout
        # killed the cold call and then both retries -- 36s spent to produce no audio.
        self.cold_start_timeout_sec = cold_start_timeout_sec
        self.max_attempts = max_attempts
        self._warm = False

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
                timeout = self.request_timeout_sec if self._warm else self.cold_start_timeout_sec
                try:
                    await asyncio.wait_for(communicate.save(str(output_path)), timeout=timeout)
                    self._warm = True
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

    def warmup(self) -> bool:
        """
        Pay the cold-start cost at startup instead of on the first thing anyone says.

        Edge-TTS is a network service; the first request here took 19.1s while the
        next two took 1.4s and 1.3s. Called from the pipeline constructor so the
        audience never waits for it.
        """
        try:
            path, meta = self.synthesize("Niaje", output_path=str(AUDIO_TEMP_DIR / "_warmup.mp3"))
            if path:
                logger.info(f"TTS warmed up in {meta.get('tts_time_ms', 0):.0f}ms")
                Path(path).unlink(missing_ok=True)
                return True
            logger.warning(f"TTS warmup failed: {meta.get('error')}. Live synthesis may be slow.")
        except Exception as e:
            logger.warning(f"TTS warmup failed: {e}. Live synthesis may be slow.")
        return False

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

