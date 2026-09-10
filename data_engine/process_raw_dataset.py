"""
End-to-End Pipeline to process raw Kenyan/Sheng audio files from /home/ray/Music/Sheng_dataset.
Slices raw audio via VAD and generates pseudo-labeled ASR datasets.
"""
import os
import sys
import json
import csv
import time
import logging
from pathlib import Path
from typing import List, Dict, Any

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from data_engine.audio_preprocessor import ShengAudioPreprocessor
from data_engine.pseudo_labeler import ShengPseudoLabeler

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ProcessRawDataset")

RAW_INPUT_DIR = Path("/home/ray/Music/Sheng_dataset")
DATASET_OUTPUT_DIR = BASE_DIR / "dataset"
SEGMENTED_DIR = DATASET_OUTPUT_DIR / "segmented"
OUTPUT_JSONL = DATASET_OUTPUT_DIR / "train_sheng_asr.jsonl"
OUTPUT_CSV = DATASET_OUTPUT_DIR / "train_sheng_asr.csv"


def process_dataset(max_files: int = None, model_size: str = "small"):
    print("==================================================================")
    print("🇰🇪 RAW SHENG AUDIO INGESTION & DATASET CREATION PIPELINE")
    print("==================================================================")
    print(f"📁 Raw Audio Source: {RAW_INPUT_DIR}")
    print(f"📦 Output Directory:  {DATASET_OUTPUT_DIR}\n")

    DATASET_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SEGMENTED_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Discover raw audio files
    audio_files = sorted(list(RAW_INPUT_DIR.glob("*.flac")) + list(RAW_INPUT_DIR.glob("*.wav")) + list(RAW_INPUT_DIR.glob("*.mp3")))
    if not audio_files:
        logger.error(f"No audio files found in {RAW_INPUT_DIR}")
        return

    if max_files:
        audio_files = audio_files[:max_files]

    logger.info(f"Discovered {len(audio_files)} raw audio files to process.")

    # 2. Step 1: Preprocess & VAD Slice
    preprocessor = ShengAudioPreprocessor(
        target_sr=16000,
        min_chunk_duration_sec=2.0,
        max_chunk_duration_sec=12.0,
        silence_thresh_db=-36,
        min_silence_len_ms=350
    )

    all_chunks = []
    print("\n[Step 1/2] Normalizing & VAD Segmenting Raw Audio...")
    for idx, raw_audio in enumerate(audio_files, 1):
        file_stem = raw_audio.stem.replace(" ", "_")
        target_sub_dir = SEGMENTED_DIR / file_stem
        target_sub_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"[{idx}/{len(audio_files)}] Segmenting: {raw_audio.name}...")
        try:
            chunks = preprocessor.chunk_audio_by_vad(str(raw_audio), output_dir=str(target_sub_dir))
            all_chunks.extend(chunks)
        except Exception as e:
            logger.error(f"Failed to process {raw_audio.name}: {e}")

    logger.info(f"✅ Step 1 Complete: Sliced into {len(all_chunks)} speech chunks in {SEGMENTED_DIR}")

    # 3. Step 2: Pseudo-Label with Sheng Prompt Biasing
    print(f"\n[Step 2/2] Pseudo-Labeling {len(all_chunks)} chunks with Whisper ('{model_size}')...")
    labeler = ShengPseudoLabeler(model_size=model_size)

    dataset_records = []
    total_speech_seconds = 0.0

    with open(OUTPUT_JSONL, "w", encoding="utf-8") as f_jsonl, open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f_csv:
        csv_writer = csv.DictWriter(f_csv, fieldnames=["id", "audio_filepath", "duration", "raw_transcription", "normalized_text", "language"])
        csv_writer.writeheader()

        for idx, chunk_info in enumerate(all_chunks, 1):
            audio_path = chunk_info["file_path"]
            raw_text, norm_text, meta = labeler.asr.transcribe(audio_path)

            if not norm_text or len(norm_text.split()) < 2:
                continue  # Skip uninformative/empty noise

            duration = meta.get("duration_seconds", chunk_info.get("duration_sec", 0))
            total_speech_seconds += duration

            record = {
                "id": f"sheng_rec_{idx:06d}",
                "audio_filepath": str(Path(audio_path).resolve()),
                "duration": round(duration, 2),
                "raw_transcription": raw_text,
                "normalized_text": norm_text,
                "language": "sw-KE"
            }
            dataset_records.append(record)
            f_jsonl.write(json.dumps(record, ensure_ascii=False) + "\n")
            csv_writer.writerow(record)

            if idx % 5 == 0 or idx == len(all_chunks):
                logger.info(f"[{idx}/{len(all_chunks)}] Transcribed: \"{norm_text[:50]}...\" ({round(duration, 1)}s)")

    # 4. Summary Statistics
    total_minutes = round(total_speech_seconds / 60.0, 2)
    print("\n==================================================================")
    print("🎉 SWAHILI & SHENG DATASET GENERATION SUMMARY")
    print("==================================================================")
    print(f"📊 Total Labeled Samples:   {len(dataset_records)}")
    print(f"⏱️  Total Speech Duration:   {total_minutes} minutes ({round(total_speech_seconds, 1)} seconds)")
    print(f"📄 JSONL Dataset Path:      {OUTPUT_JSONL}")
    print(f"📊 CSV Dataset Path:        {OUTPUT_CSV}")
    print("==================================================================\n")


if __name__ == "__main__":
    process_dataset()
