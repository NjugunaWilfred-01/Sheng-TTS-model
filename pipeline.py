"""
End-to-End Speech-to-Speech (S2S) Orchestration Pipeline for Swahili & Sheng.
"""
import time
import logging
from pathlib import Path
from typing import Dict, Any, Optional

from asr_engine import ShengASREngine
from llm_engine import ShengLLMEngine
from tts_engine import ShengTTSEngine
from config import DEFAULT_VOICE, AUDIO_TEMP_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("S2SPipeline")


class SpeechToSpeechPipeline:
    def __init__(
        self,
        asr_engine: Optional[ShengASREngine] = None,
        llm_engine: Optional[ShengLLMEngine] = None,
        tts_engine: Optional[ShengTTSEngine] = None
    ):
        logger.info("Initializing S2S Pipeline components...")
        self.asr = asr_engine or ShengASREngine()
        self.llm = llm_engine or ShengLLMEngine()
        self.tts = tts_engine or ShengTTSEngine()
        logger.info("S2S Pipeline successfully initialized.")

    def reset_conversation(self):
        """Clears conversational multi-turn history."""
        self.llm.reset_history()

    def process_audio(
        self,
        input_audio_path: str,
        voice: str = DEFAULT_VOICE,
        rate: str = "+0%",
        pitch: str = "+0Hz",
        backend: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Executes full Audio In -> ASR -> LLM -> TTS -> Audio Out loop.
        """
        if not input_audio_path or not Path(input_audio_path).exists():
            return {
                "success": False,
                "error": "Input audio file does not exist."
            }

        if backend:
            self.llm.backend = backend

        total_start = time.time()

        # Step 1: Automatic Speech Recognition (ASR)
        raw_text, normalized_text, asr_meta = self.asr.transcribe(input_audio_path)
        if not normalized_text:
            return {
                "success": False,
                "error": "Hakuna sauti iliyosikika vizuri (No clear speech detected).",
                "asr_metadata": asr_meta
            }

        # Step 2: Sheng Dialogue Brain (LLM)
        bot_reply, llm_meta = self.llm.generate_response(normalized_text)

        # Step 3: Kenyan Neural Voice Synthesis (TTS)
        out_filename = f"s2s_response_{int(time.time() * 1000)}.mp3"
        out_path = str(AUDIO_TEMP_DIR / out_filename)
        audio_output, tts_meta = self.tts.synthesize(
            text=bot_reply,
            output_path=out_path,
            voice=voice,
            rate=rate,
            pitch=pitch
        )

        total_latency_ms = round((time.time() - total_start) * 1000, 2)

        return {
            "success": True,
            "user_raw_transcription": raw_text,
            "user_normalized_sheng": normalized_text,
            "bot_response_text": bot_reply,
            "output_audio_path": audio_output,
            "latencies": {
                "asr_ms": asr_meta.get("inference_time_ms", 0),
                "llm_ms": llm_meta.get("llm_time_ms", 0),
                "tts_ms": tts_meta.get("tts_time_ms", 0),
                "total_glass_to_glass_ms": total_latency_ms
            }
        }
