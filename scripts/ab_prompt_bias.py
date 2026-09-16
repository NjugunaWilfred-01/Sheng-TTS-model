"""
ab_prompt_bias.py - Does Whisper prompt biasing help or hurt? Measure, don't guess.

We have exact ground truth for the six demo clips: scripts/generate_demo_audio.py
synthesised them from known strings, so WER here is real and not an estimate. That
makes this the one honest prompt experiment available without the human-transcribed
eval set (zoza_transcripts/, which was never committed).

Three conditions on identical audio:

  none      no initial_prompt at all
  sentence  the original fluent-sentence prompt -- leaked verbatim into 16/123
            clips on results_baseline.csv
  wordlist  a bare comma-separated vocabulary list

Caveat worth stating out loud: these clips are TTS-clean, so absolute WER will be far
better here than on real conversational audio. The comparison BETWEEN conditions is
the meaningful output, not the absolute numbers.

Usage:
    python scripts/ab_prompt_bias.py
    python scripts/ab_prompt_bias.py --model small
"""
import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import WHISPER_MODEL_SIZE, ASR_DEVICE, ASR_COMPUTE_TYPE  # noqa: E402
from scripts.generate_demo_audio import DEMO_PROMPTS  # noqa: E402

SENTENCE_PROMPT = (
    "Niaje chief, form ni gani leo mtaani? Niko fiti na mbogi, "
    "nisho vile tunaingia tao na nganya."
)
WORDLIST_PROMPT = (
    "niaje, chief, form, mtaani, fiti, mbogi, nisho, tao, nganya, rada, dooh, "
    "keja, luku, hustle, msee, maze, morio, mresh, kibanda, mathree, nduthi, "
    "chwani, ganji, wazi, alafu, kuchill, tunang'ang'ana"
)

# Fluent Sheng (which the data says beats a word list) but deliberately sharing NO
# phrasing with any demo sentence -- so a verbatim echo is distinguishable from a
# correct transcript. The old SENTENCE_PROMPT literally contained demo clip 1.
DISJOINT_PROMPT = (
    "Mbogi ilikuwa imechill kejani jana, tukipiga stori za dooh na luku. "
    "Nilipanda nduthi hadi kibanda, nikanunua chapo na madondo, alafu nikarudi mtaani."
)

CONDITIONS = [("none", None), ("sentence", SENTENCE_PROMPT),
              ("wordlist", WORDLIST_PROMPT), ("disjoint", DISJOINT_PROMPT)]

DECODE = dict(beam_size=5, temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
              condition_on_previous_text=False, no_speech_threshold=0.6,
              compression_ratio_threshold=2.4, log_prob_threshold=-1.0,
              vad_filter=True, vad_parameters=dict(min_silence_duration_ms=400))


def norm(text):
    text = text.lower().strip()
    text = re.sub(r"[^\w\s']", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=WHISPER_MODEL_SIZE)
    args = ap.parse_args()

    import jiwer
    from faster_whisper import WhisperModel

    root = Path(__file__).resolve().parent.parent
    clips = [(name, root / "assets" / "demo_samples" / f"{name}.mp3", ref)
             for name, ref in DEMO_PROMPTS.items()]
    clips = [(n, p, r) for n, p, r in clips if p.exists()]
    if not clips:
        sys.exit("No demo audio. Run: python scripts/generate_demo_audio.py")

    print(f"Model: {args.model} ({ASR_DEVICE}/{ASR_COMPUTE_TYPE}) on {len(clips)} clips\n")
    model = WhisperModel(args.model, device=ASR_DEVICE, compute_type=ASR_COMPUTE_TYPE)

    results = {}
    for label, prompt in CONDITIONS:
        wers, comma_ratios, outputs = [], [], []
        for name, path, ref in clips:
            segments, _ = model.transcribe(str(path), language="sw",
                                           initial_prompt=prompt, **DECODE)
            hyp = " ".join(s.text.strip() for s in segments).strip()
            wers.append(jiwer.wer(norm(ref), norm(hyp)) if norm(hyp) else 1.0)
            # Comma density exposes list-formatting bias: the wordlist prompt can
            # push the decoder to emit a comma-separated list instead of a sentence.
            words = max(len(hyp.split()), 1)
            comma_ratios.append(hyp.count(",") / words)
            outputs.append((name, ref, hyp))
        results[label] = {
            "mean_wer": sum(wers) / len(wers),
            "comma_ratio": sum(comma_ratios) / len(comma_ratios),
            "outputs": outputs,
        }

    print(f"{'condition':<12}{'mean WER':>10}{'comma/word':>13}   (both: lower is better)")
    for label, _ in CONDITIONS:
        r = results[label]
        print(f"{label:<12}{r['mean_wer']:>10.3f}{r['comma_ratio']:>13.3f}")

    print("\nPer-clip transcripts:")
    for name, ref, _ in results["none"]["outputs"]:
        print(f"\n  {name}\n    REF       : {ref}")
        for label, _ in CONDITIONS:
            hyp = dict((n, h) for n, _, h in results[label]["outputs"])[name]
            print(f"    {label:<10}: {hyp}")

    best = min(results, key=lambda k: results[k]["mean_wer"])
    print(f"\nLowest mean WER: {best} ({results[best]['mean_wer']:.3f})")


if __name__ == "__main__":
    main()
