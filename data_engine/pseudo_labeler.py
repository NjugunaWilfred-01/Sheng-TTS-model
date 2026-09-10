"""
Sheng Pseudo-Labeling Pipeline:
Transcribes audio chunks, applies Sheng normalizations, and exports HuggingFace ASR datasets.
"""
import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Any

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from asr_engine import ShengASREngine
from sheng_lexicon import ShengNormalizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("PseudoLabeler")


class ShengPseudoLabeler:
    def __init__(self, model_size: str = "small"):
        self.asr = ShengASREngine(model_size=model_size)

    def label_directory(
        self,
        audio_dir: str,
        output_jsonl: str = "train_sheng_asr.jsonl"
    ) -> List[Dict[str, Any]]:
        """
        Iterates over all audio files in a directory, transcribes them, and outputs a JSONL dataset.
        """
        audio_paths = sorted(list(Path(audio_dir).glob("*.wav")) + list(Path(audio_dir).glob("*.mp3")))
        logger.info(f"Found {len(audio_paths)} audio files to pseudo-label in {audio_dir}...")

        records = []
        with open(output_jsonl, "w", encoding="utf-8") as f_out:
            for idx, audio_file in enumerate(audio_paths, 1):
                raw_text, norm_text, meta = self.asr.transcribe(str(audio_file))
                if not norm_text or len(norm_text.split()) < 2:
                    continue  # Skip empty or single-character noise

                record = {
                    "id": f"sheng_audio_{idx:06d}",
                    "audio_filepath": str(audio_file.resolve()),
                    "duration": meta.get("duration_seconds", 0),
                    "raw_transcription": raw_text,
                    "normalized_text": norm_text,
                    "language": "sw-KE"
                }
                records.append(record)
                f_out.write(json.dumps(record, ensure_ascii=False) + "\n")

                if idx % 10 == 0 or idx == len(audio_paths):
                    logger.info(f"Progress: [{idx}/{len(audio_paths)}] labeled -> '{norm_text[:40]}...'")

        logger.info(f"✅ Pseudo-labeling complete! Generated {len(records)} samples in {output_jsonl}")
        return records
