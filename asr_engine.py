"""
ASR Engine: Faster-Whisper with Sheng Prompt Biasing and Normalization.
"""
import time
import logging
from pathlib import Path
from typing import Tuple, Dict, Any, Optional

from config import WHISPER_MODEL_SIZE, ASR_LANGUAGE, ASR_DEVICE, ASR_COMPUTE_TYPE
from sheng_lexicon import WHISPER_SHENG_PROMPT, ShengNormalizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ShengASREngine")


class ShengASREngine:
    def __init__(
        self,
        model_size: str = WHISPER_MODEL_SIZE,
        device: str = ASR_DEVICE,
        compute_type: str = ASR_COMPUTE_TYPE,
        initial_prompt: str = WHISPER_SHENG_PROMPT
    ):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.initial_prompt = initial_prompt
        self.model = None
        self._load_model()

    def _load_model(self):
        """Lazy loads the Faster-Whisper model."""
        try:
            from faster_whisper import WhisperModel
            logger.info(f"Loading Whisper model '{self.model_size}' on device '{self.device}' ({self.compute_type})...")
            self.model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type
            )
            logger.info("Whisper model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load Faster-Whisper: {e}")
            self.model = None

    def transcribe(self, audio_path: str) -> Tuple[str, str, Dict[str, Any]]:
        """
        Transcribes audio to Swahili/Sheng text.
        
        Returns:
            Tuple[raw_text, normalized_text, metadata]
        """
        if not self.model:
            self._load_model()
            if not self.model:
                return "", "", {"error": "ASR model not initialized"}

        start_time = time.time()
        try:
            segments, info = self.model.transcribe(
                audio_path,
                language=ASR_LANGUAGE,
                initial_prompt=self.initial_prompt,
                beam_size=5,
                temperature=0.0,
                condition_on_previous_text=False,
                no_speech_threshold=0.6,
                compression_ratio_threshold=2.4,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=400)
            )

            raw_text = " ".join([seg.text.strip() for seg in segments]).strip()
            normalized_text = ShengNormalizer.normalize(raw_text)
            duration = time.time() - start_time

            metadata = {
                "detected_language": info.language,
                "language_probability": round(info.language_probability, 3),
                "duration_seconds": info.duration,
                "inference_time_ms": round(duration * 1000, 2)
            }
            logger.info(f"ASR complete in {metadata['inference_time_ms']}ms: '{normalized_text}'")
            return raw_text, normalized_text, metadata

        except Exception as e:
            logger.error(f"Error during transcription: {e}")
            return "", "", {"error": str(e), "inference_time_ms": round((time.time() - start_time) * 1000, 2)}
