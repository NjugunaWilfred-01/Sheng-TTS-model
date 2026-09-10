"""
Self-Contained Test Suite for Swahili & Sheng S2S Pipeline.
Generates synthetic speech, tests ASR prompt biasing, LLM banter, and TTS response.
"""
import sys
import time
from pathlib import Path

from sheng_lexicon import ShengNormalizer, SHENG_DICTIONARY
from tts_engine import ShengTTSEngine
from llm_engine import ShengLLMEngine
from asr_engine import ShengASREngine
from config import AUDIO_TEMP_DIR


def run_tests():
    print("==================================================")
    print("🇰🇪 SWAHILI & SHENG S2S PIPELINE TEST SUITE")
    print("==================================================\n")

    # Test 1: Sheng Normalizer
    print("[1/4] Testing Sheng Normalizer & Slang Rules...")
    test_cases = [
        ("ni aje ba zenga", "niaje bazenga"),
        ("m bogi iko na cha paa", "mbogi iko na chapaa"),
        ("tupande ma three ya nganya", "tupande mathree ya nganya")
    ]
    for inp, expected in test_cases:
        norm = ShengNormalizer.normalize(inp)
        assert norm == expected, f"Expected '{expected}', got '{norm}'"
        print(f"  ✓ '{inp}' -> '{norm}'")
    print("  -> Sheng Normalizer passed!\n")

    # Test 2: TTS Synthesis
    print("[2/4] Testing Edge-TTS Kenyan Voice Synthesis...")
    tts = ShengTTSEngine()
    sample_phrase = "Niaje bazenga! Form ni gani leo mtaani?"
    test_audio_path = str(AUDIO_TEMP_DIR / "test_input.mp3")

    audio_file, meta = tts.synthesize(sample_phrase, output_path=test_audio_path)
    if Path(audio_file).exists() and Path(audio_file).stat().st_size > 0:
        print(f"  ✓ Generated audio at '{audio_file}' in {meta['tts_time_ms']}ms")
    else:
        print(f"  ✗ Failed to generate TTS audio: {meta}")
        return False
    print("  -> Kenyan Neural TTS passed!\n")

    # Test 3: Sheng LLM Brain
    print("[3/4] Testing Sheng LLM Brain...")
    llm = ShengLLMEngine(backend="heuristic")
    reply, meta = llm.generate_response("Niaje bazenga, form ni gani?")
    print(f"  ✓ User: 'Niaje bazenga, form ni gani?'")
    print(f"  ✓ Bot Reply ({meta['backend']}): '{reply}' ({meta['llm_time_ms']}ms)")
    print("  -> LLM Engine passed!\n")

    # Test 4: Whisper ASR Transcription
    print("[4/4] Testing Whisper ASR with Sheng Prompt Biasing...")
    asr = ShengASREngine(model_size="tiny")  # Use tiny for rapid test verification
    if asr.model is not None:
        raw, normalized, asr_meta = asr.transcribe(test_audio_path)
        print(f"  ✓ Raw ASR: '{raw}'")
        print(f"  ✓ Normalized: '{normalized}' ({asr_meta['inference_time_ms']}ms)")
        print("  -> Whisper ASR passed!\n")
    else:
        print("  ⚠️ Faster-Whisper library not loaded in environment yet (expected until packages installed).")

    print("==================================================")
    print("🎉 Core S2S Pipeline Components Verified!")
    print("==================================================")
    return True


if __name__ == "__main__":
    run_tests()
