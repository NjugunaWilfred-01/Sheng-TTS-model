"""
ab_decode_config.py - A/B the OLD vs NEW Whisper decode settings on the same clips.

results_baseline.csv showed two concrete decoder failures that are independent of
model size or of how good the audio is:

  1. Prompt leakage    - the sentence-form initial_prompt emitted verbatim as the
                         transcript. 16 of 123 clips (13%) on the baseline run.
  2. Repetition loops  - "Ha ha ha ..." x67. Caused by a SCALAR temperature=0.0,
                         which leaves compression_ratio_threshold with no hotter
                         temperature to fall back to, so detected garbage is kept.

Both are measurable without ground-truth transcripts, which matters because the
human-referenced eval set (zoza_transcripts/) is not in the repo. This script counts
both failure modes under each config on the same audio, so the fix is verified rather
than assumed.

Usage:
    python scripts/ab_decode_config.py --n 40
    python scripts/ab_decode_config.py --n 40 --model small
"""
import argparse
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sheng_lexicon import WHISPER_SHENG_PROMPT  # noqa: E402
from config import WHISPER_MODEL_SIZE, ASR_DEVICE, ASR_COMPUTE_TYPE  # noqa: E402

# The sentence-form prompt exactly as it shipped, so the A side is a faithful replay.
OLD_PROMPT = (
    "Niaje chief, form ni gani leo mtaani? Niko fiti na mbogi, "
    "nisho vile tunaingia tao na nganya."
)

OLD_CFG = dict(beam_size=5, temperature=0.0, condition_on_previous_text=False,
               no_speech_threshold=0.6, compression_ratio_threshold=2.4,
               vad_filter=True, vad_parameters=dict(min_silence_duration_ms=400))

NEW_CFG = dict(beam_size=5, temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
               condition_on_previous_text=False, no_speech_threshold=0.6,
               compression_ratio_threshold=2.4, log_prob_threshold=-1.0,
               vad_filter=True, vad_parameters=dict(min_silence_duration_ms=400))


def is_prompt_echo(text, prompt):
    """True if the transcript is overwhelmingly made of prompt vocabulary."""
    prompt_words = {w.strip(" ,.?!").lower() for w in prompt.replace(",", " ").split()}
    prompt_words.discard("")
    words = [w.strip(" ,.?!").lower() for w in text.split() if w.strip(" ,.?!")]
    if len(words) < 3:
        return False
    return sum(1 for w in words if w in prompt_words) / len(words) >= 0.8


def is_loop(text):
    """True if one token dominates a long transcript -- a degenerate repeat."""
    words = [w.lower() for w in text.split()]
    if len(words) < 12:
        return False
    return Counter(words).most_common(1)[0][1] / len(words) > 0.35


def run(model, clips, prompt, cfg, label):
    leaks = loops = empties = 0
    total_ms = 0.0
    samples = []
    print(f"\n--- {label} ---")
    for i, clip in enumerate(clips, 1):
        start = time.time()
        segments, _ = model.transcribe(str(clip), language="sw",
                                       initial_prompt=prompt, **cfg)
        text = " ".join(s.text.strip() for s in segments).strip()
        total_ms += (time.time() - start) * 1000
        if is_prompt_echo(text, prompt):
            leaks += 1
            samples.append(("LEAK", clip.name, text[:70]))
        if is_loop(text):
            loops += 1
            samples.append(("LOOP", clip.name, text[:70]))
        if not text:
            empties += 1
        if i % 10 == 0:
            print(f"  {i}/{len(clips)} clips...")
    return {"label": label, "clips": len(clips), "prompt_leaks": leaks,
            "loops": loops, "empty": empties,
            "mean_ms": round(total_ms / max(len(clips), 1), 1),
            "samples": samples}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40, help="number of clips to sample")
    ap.add_argument("--model", default=WHISPER_MODEL_SIZE)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--out", default="results_decode_ab.json")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    clips = sorted((root / "dataset" / "segmented").rglob("*.wav"))
    if not clips:
        sys.exit("No clips under dataset/segmented/.")
    random.Random(args.seed).shuffle(clips)
    clips = clips[:args.n]
    print(f"Sampling {len(clips)} clips. Model: {args.model} ({ASR_DEVICE}/{ASR_COMPUTE_TYPE})")

    from faster_whisper import WhisperModel
    model = WhisperModel(args.model, device=ASR_DEVICE, compute_type=ASR_COMPUTE_TYPE)

    old = run(model, clips, OLD_PROMPT, OLD_CFG, "OLD: sentence prompt + scalar temperature")
    new = run(model, clips, WHISPER_SHENG_PROMPT, NEW_CFG, "NEW: word-list prompt + temperature fallback")

    print(f"\n{'':<14}{'OLD':>8}{'NEW':>8}   (lower is better)")
    for key in ("prompt_leaks", "loops", "empty"):
        print(f"{key:<14}{old[key]:>8}{new[key]:>8}")
    print(f"{'mean_ms':<14}{old['mean_ms']:>8}{new['mean_ms']:>8}   (speed, not quality)")

    for side in (old, new):
        if side["samples"]:
            print(f"\nExamples from {side['label']}:")
            for kind, name, text in side["samples"][:4]:
                print(f"  [{kind}] {name}: {text}")

    Path(args.out).write_text(
        json.dumps({"model": args.model, "old": old, "new": new}, indent=2, default=str),
        encoding="utf-8")
    print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()
