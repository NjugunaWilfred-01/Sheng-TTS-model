"""
eval_asr.py - Batch ASR evaluation for the Sheng-TTS-model pipeline.

Runs Faster-Whisper over an audio/transcript dataset (zoza_transcripts/mapped_data)
and computes WER/CER per file + summary statistics. Designed to be run identically
on the baseline (main) and the experiment branch so results are comparable.

Usage:
  # Baseline (off-the-shelf whisper small, as per config.py on main)
  python eval_asr.py --model small --data_dir zoza_transcripts/mapped_data --out results_baseline.csv

  # Experiment (fine-tuned checkpoint, converted to faster-whisper format)
  python eval_asr.py --model /path/to/fine-tuned-ct2-model --data_dir zoza_transcripts/mapped_data --out results_experiment.csv

Then compare the two CSV summaries.
"""
import argparse
import csv
import re
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Text normalization for fair WER/CER (whisper-style lowercasing + punctuation strip)
# ---------------------------------------------------------------------------
def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s']", " ", text)   # strip punctuation, keep letters/digits/underscore
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def pair_files(data_dir: Path):
    """Match audio_X_for_script_X.mp3 with transcript_X_for_audio_X.txt by index X."""
    audio_dir = data_dir / "audios"
    transcript_dir = data_dir / "transcripts"
    pairs = []
    for audio_path in sorted(audio_dir.glob("audio_*_for_script_*.mp3"),
                            key=lambda p: int(re.search(r"audio_(\d+)", p.name).group(1))):
        idx = int(re.search(r"audio_(\d+)", audio_path.name).group(1))
        transcript_path = transcript_dir / f"transcript_{idx}_for_audio_{idx}.txt"
        if transcript_path.exists():
            pairs.append((idx, audio_path, transcript_path))
        else:
            print(f"  ! WARNING: no transcript found for {audio_path.name}, skipping.")
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    help="Whisper model size ('small', 'medium'...) OR path to a converted faster-whisper checkpoint")
    ap.add_argument("--data_dir", default="zoza_transcripts/mapped_data")
    ap.add_argument("--out", default="results.csv")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--language", default="sw")
    ap.add_argument("--no_prompt", action="store_true",
                    help="Disable Sheng prompt biasing (match test_pipeline/asr_engine defaults: prompt ON)")
    args = ap.parse_args()

    try:
        from faster_whisper import WhisperModel
    except ImportError:
        sys.exit("faster-whisper not installed. Run: pip install faster-whisper")

    try:
        import jiwer
    except ImportError:
        sys.exit("jiwer not installed. Run: pip install jiwer")

    try:
        from sheng_lexicon import WHISPER_SHENG_PROMPT
        sheng_prompt = None if args.no_prompt else WHISPER_SHENG_PROMPT
    except ImportError:
        print("  ! sheng_lexicon.py not found - running WITHOUT Sheng prompt biasing.")
        sheng_prompt = None

    compute_type = "float16" if args.device == "cuda" else "int8"
    print(f"Loading model: {args.model} ({args.device}, {compute_type})")
    model = WhisperModel(args.model, device=args.device, compute_type=compute_type)

    pairs = pair_files(Path(args.data_dir))
    print(f"Found {len(pairs)} audio/transcript pairs in {args.data_dir}\n")

    rows = []
    for idx, audio_path, transcript_path in pairs:
        reference = transcript_path.read_text(encoding="utf-8").strip()
        start = time.time()
        segments, info = model.transcribe(
            str(audio_path),
            language=args.language,
            initial_prompt=sheng_prompt,
            beam_size=5,
            temperature=0.0,
            condition_on_previous_text=False,
            no_speech_threshold=0.6,
            compression_ratio_threshold=2.4,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=400),
        )
        hypothesis = " ".join(seg.text.strip() for seg in segments).strip()
        elapsed_ms = round((time.time() - start) * 1000, 1)

        ref_n, hyp_n = normalize_text(reference), normalize_text(hypothesis)
        wer = jiwer.wer(ref_n, hyp_n)
        cer = jiwer.cer(ref_n, hyp_n)
        rows.append({
            "id": idx,
            "audio": audio_path.name,
            "reference": reference,
            "hypothesis": hypothesis,
            "wer": round(wer, 4),
            "cer": round(cer, 4),
            "rtf": round(elapsed_ms / max(info.duration, 0.001) / 1000, 4),
            "inference_ms": elapsed_ms,
        })
        print(f"  [{idx:>3}] WER={wer:.3f} CER={cer:.3f}  {elapsed_ms:>6} ms | {hypothesis[:60]}")

    # Summary
    wers = [r["wer"] for r in rows]
    cers = [r["cer"] for r in rows]
    summary = {
        "files": len(rows),
        "mean_wer": round(sum(wers) / len(wers), 4),
        "median_wer": round(sorted(wers)[len(wers) // 2], 4),
        "mean_cer": round(sum(cers) / len(cers), 4),
    }

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("\n================ SUMMARY ================")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"  per-file results saved to: {args.out}")


if __name__ == "__main__":
    main()
