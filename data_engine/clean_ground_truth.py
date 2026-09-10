"""
Ground-Truth Sheng Dataset Cleaner:
Cleans noisy Whisper pseudo-labels, fixes phonetic misrecognitions,
and standardizes orthography into authentic Nairobi Sheng.
"""
import re
import json
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("GroundTruthCleaner")

BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_JSONL = BASE_DIR / "dataset" / "train_sheng_asr.jsonl"
OUTPUT_JSONL = BASE_DIR / "dataset" / "train_sheng_asr_cleaned.jsonl"
OUTPUT_CSV = BASE_DIR / "dataset" / "train_sheng_asr_cleaned.csv"

# Comprehensive Nairobi Sheng Orthographic Normalization Rules
SHENG_CLEANING_RULES: List[Tuple[re.Pattern, str]] = [
    # 1. Greetings & Everyday Slang
    (re.compile(r"\bniage\b", re.IGNORECASE), "niaje"),
    (re.compile(r"\bni\s+aje\b", re.IGNORECASE), "niaje"),
    (re.compile(r"\bfom\b", re.IGNORECASE), "form"),
    (re.compile(r"\bthe\s+form\b", re.IGNORECASE), "form"),
    (re.compile(r"\bradah\b", re.IGNORECASE), "rada"),
    (re.compile(r"\bleom\s+tani\b", re.IGNORECASE), "leo mtaani"),
    (re.compile(r"\bleo\s+mtani\b", re.IGNORECASE), "leo mtaani"),
    (re.compile(r"\bmtani\b", re.IGNORECASE), "mtaani"),
    (re.compile(r"\bm\s+tani\b", re.IGNORECASE), "mtaani"),
    (re.compile(r"\bm\s+taa\b", re.IGNORECASE), "mtaa"),

    # 2. People & Identities
    (re.compile(r"\bba\s+zenga\b", re.IGNORECASE), "bazenga"),
    (re.compile(r"\bbazeng\b", re.IGNORECASE), "bazenga"),
    (re.compile(r"\bm\s+bogi\b", re.IGNORECASE), "mbogi"),
    (re.compile(r"\bm\s+resh\b", re.IGNORECASE), "mresh"),
    (re.compile(r"\bmo\s+rio\b", re.IGNORECASE), "morio"),
    (re.compile(r"\bamorio\b", re.IGNORECASE), "morio"),
    (re.compile(r"\bmsee\s+wa\b", re.IGNORECASE), "msee wa"),
    (re.compile(r"\bmadya\b", re.IGNORECASE), "matha"),
    (re.compile(r"\bonyango\b", re.IGNORECASE), "Onyango"),

    # 3. Money & Value
    (re.compile(r"\bcha\s+paa\b", re.IGNORECASE), "chapaa"),
    (re.compile(r"\bchapa\b", re.IGNORECASE), "chapaa"),
    (re.compile(r"\bchuani\b", re.IGNORECASE), "chwani"),
    (re.compile(r"\bchwa\s+ni\b", re.IGNORECASE), "chwani"),
    (re.compile(r"\bga\s+nji\b", re.IGNORECASE), "ganji"),
    (re.compile(r"\bfinje\b", re.IGNORECASE), "finje"),

    # 4. Transport & Places
    (re.compile(r"\bndu\s+thi\b", re.IGNORECASE), "nduthi"),
    (re.compile(r"\bma\s+three\b", re.IGNORECASE), "mathree"),
    (re.compile(r"\bki\s+banda\b", re.IGNORECASE), "kibanda"),
    (re.compile(r"\bkeje\b", re.IGNORECASE), "keja"),
    (re.compile(r"\bke\s+ja\b", re.IGNORECASE), "keja"),
    (re.compile(r"\bganiya\b", re.IGNORECASE), "nganya"),
    (re.compile(r"\bpandan\s+ganiya\b", re.IGNORECASE), "panda nganya"),
    (re.compile(r"\bnga\s+nya\b", re.IGNORECASE), "nganya"),

    # 5. Colloquial Verbs & Phrases
    (re.compile(r"\blunci\b", re.IGNORECASE), "lunch"),
    (re.compile(r"\bku\s+chill\b", re.IGNORECASE), "kuchill"),
    (re.compile(r"\bku\s+mnoki\b", re.IGNORECASE), "kumnoki"),
    (re.compile(r"\bwa\s+zi\b", re.IGNORECASE), "wazi"),
    (re.compile(r"\blu\s+ku\b", re.IGNORECASE), "luku"),
    (re.compile(r"\bmanze\b", re.IGNORECASE), "maze"),
    (re.compile(r"\bniaje,\s+keze\b", re.IGNORECASE), "nielekeze"),

    # 6. Repetitive Artifacts Removal (Whisper looping glitch)
    (re.compile(r"(kwa\s+){3,}", re.IGNORECASE), "kwa "),
    (re.compile(r"(nuku,\s+){3,}", re.IGNORECASE), "nuku, "),
]


def clean_text(text: str) -> str:
    """Applies clean Sheng orthography rules."""
    cleaned = text
    for pattern, replacement in SHENG_CLEANING_RULES:
        cleaned = pattern.sub(replacement, cleaned)
    # Clean redundant punctuation and spaces
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def clean_dataset():
    logger.info(f"Loading raw dataset from {INPUT_JSONL}...")
    if not INPUT_JSONL.exists():
        logger.error(f"Input file not found: {INPUT_JSONL}")
        return

    cleaned_records = []
    with open(INPUT_JSONL, "r", encoding="utf-8") as f_in, \
         open(OUTPUT_JSONL, "w", encoding="utf-8") as f_out:
        
        for line in f_in:
            if not line.strip():
                continue
            item = json.loads(line.strip())
            original_text = item["normalized_text"]
            perfected_text = clean_text(original_text)

            item["normalized_text"] = perfected_text
            item["clean_sheng_text"] = perfected_text
            cleaned_records.append(item)

            f_out.write(json.dumps(item, ensure_ascii=False) + "\n")

    logger.info(f"✅ Cleaned {len(cleaned_records)} ground-truth records.")
    logger.info(f"📄 Saved cleaned JSONL to: {OUTPUT_JSONL}")

    # Display sample corrections
    print("\n=======================================================")
    print("✨ SAMPLE GROUND-TRUTH SHENG CORRECTIONS")
    print("=======================================================")
    for rec in cleaned_records[:5]:
        print(f"ID: {rec['id']}")
        print(f"  Audio: {Path(rec['audio_filepath']).name}")
        print(f"  Cleaned Sheng Text: \"{rec['normalized_text'][:80]}...\"\n")
    print("=======================================================\n")


if __name__ == "__main__":
    clean_dataset()
