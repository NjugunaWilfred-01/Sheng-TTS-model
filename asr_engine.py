"""
ASR Engine: Faster-Whisper with Sheng Prompt Biasing and Normalization.
"""
import re
import time
import logging
from difflib import SequenceMatcher
from pathlib import Path
from typing import Tuple, Dict, Any, Optional

from config import WHISPER_MODEL_SIZE, ASR_LANGUAGE, ASR_DEVICE, ASR_COMPUTE_TYPE
from sheng_lexicon import WHISPER_SHENG_PROMPT, WHISPER_PROMPT_ECHO_MARKERS, ShengNormalizer

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

    # Tried in order when the configured model will not load. Smaller means worse
    # Sheng accuracy, but a degraded transcriber beats a dead pipeline on stage.
    FALLBACK_SIZES = ("small", "base", "tiny")

    def _load_model(self):
        """
        Load the Faster-Whisper model, degrading to a smaller one if necessary.

        The configured default (large-v3-turbo, ~1.6GB) is fetched on first use, and
        that download proved flaky here -- it failed once with a HuggingFace Xet
        transfer error and once by stalling. Previously any such failure left
        self.model as None and every transcribe() returned "ASR model not
        initialized", i.e. total pipeline failure. Now we fall back through smaller
        models, which are likely already cached, and report which one is live.
        """
        from faster_whisper import WhisperModel

        candidates = [self.model_size] + [
            s for s in self.FALLBACK_SIZES if s != self.model_size
        ]
        for size in candidates:
            try:
                logger.info(f"Loading Whisper '{size}' on {self.device} ({self.compute_type})...")
                self.model = WhisperModel(size, device=self.device, compute_type=self.compute_type)
                if size != self.model_size:
                    logger.warning(
                        f"Could not load '{self.model_size}'; running on '{size}' instead. "
                        f"Transcription quality will be lower. Run "
                        f"scripts/prefetch_models.py to cache the intended model."
                    )
                self.model_size = size
                logger.info(f"Whisper model '{size}' loaded successfully.")
                return
            except Exception as e:
                logger.error(f"Failed to load Whisper '{size}': {e}")

        logger.error("No Whisper model could be loaded. ASR is unavailable.")
        self.model = None

    def _strip_prompt_echo(self, text: str) -> str:
        """
        Discard transcripts that are just the initial_prompt read back.

        Whisper treats initial_prompt as text it has already written and will
        continue it, so on quiet or non-speech audio it emits the prompt verbatim as
        the transcript -- 16 of 123 clips (13%) on results_baseline.csv.

        We cannot test similarity against the WHOLE prompt. Its first clause is also
        demo scenario 1 ("Niaje chief, form ni gani leo mtaani?"), so that exact
        string appears in the baseline both as a leak and as a correct transcript --
        the same bytes, genuinely inseparable. Instead we match only the clauses no
        user would say (WHISPER_PROMPT_ECHO_MARKERS), and bias every ambiguous case
        towards keeping the text: silently eating real speech is the worse failure.

        Validated by scripts/validate_echo_guard.py: 13/16 known leaks caught, zero
        false positives on known-good transcripts.
        """
        if not text:
            return text

        hyp = re.sub(r"\s+", " ", re.sub(r"[^\w\s']", " ", text.lower())).strip()
        if not hyp:
            return text

        for marker in WHISPER_PROMPT_ECHO_MARKERS:
            # Ordered similarity, because echoes come back garbled and reordered
            # rather than byte-exact. 0.60 was picked by sweeping against the 16
            # known leaks and the known-good transcripts; see the validate script.
            ratio = SequenceMatcher(None, hyp, marker).ratio()
            if ratio >= 0.60:
                logger.warning(f"Discarding prompt echo (sim {ratio:.2f}): '{text[:70]}'")
                return ""
        return text

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
                # A TUPLE, not a scalar. compression_ratio_threshold and
                # log_prob_threshold can only reject a bad decode if there is a
                # hotter temperature to retry at. With the old scalar 0.0 there was
                # no fallback, so detected garbage was kept anyway -- that is what
                # produced the "Ha ha ha ..." x67 loops in results_baseline.csv.
                temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
                condition_on_previous_text=False,
                no_speech_threshold=0.6,
                compression_ratio_threshold=2.4,
                log_prob_threshold=-1.0,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=400)
            )

            raw_text = " ".join([seg.text.strip() for seg in segments]).strip()
            raw_text = self._strip_prompt_echo(raw_text)
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
