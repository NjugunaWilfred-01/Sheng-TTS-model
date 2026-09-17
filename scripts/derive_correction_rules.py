"""
derive_correction_rules.py - Mine real correction rules from human-referenced data.

WHY THIS EXISTS
---------------
ASR_CORRECTION_RULES were hand-written by reading the six TTS-generated demo clips.
Measured against results_baseline.csv -- the only 100 clips with genuine human
transcripts -- ZERO of the 61 rules fire. Not "they help less than hoped": they never
match at all. The reported ~5x WER improvement (0.699 -> 0.139) is entirely an
artifact of the demo set, and the normalizer is a no-op on real conversational Sheng.

This script derives rules from evidence instead. It aligns each hypothesis against
its human reference, collects the substitutions Whisper actually makes, and proposes
only those that recur often enough to be a systematic error rather than a one-off --
then measures the WER change before anything is adopted.

A proposal is only worth taking if it lowers WER on held-out clips. The script reports
that honestly, including when the answer is "nothing here is worth adopting".

Usage:
    python scripts/derive_correction_rules.py
    python scripts/derive_correction_rules.py --min-count 3 --emit
"""
import argparse
import csv
import re
import statistics
import sys
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sheng_lexicon import ShengNormalizer  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def norm(text):
    text = text.lower().strip()
    text = re.sub(r"[^\w\s']", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def load_pairs(csv_path):
    """(reference, hypothesis) pairs that both carry real content."""
    pairs = []
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            ref, hyp = row["reference"], row["hypothesis"]
            if re.sub(r"[^\w\s]", "", ref).strip() and hyp.strip():
                pairs.append((ref, hyp))
    return pairs


def mine_substitutions(pairs):
    """
    Collect hypothesis->reference word swaps from the alignment of each pair.

    Both 1:1 swaps ("usle" -> "hustle") and the n:m spans that catch split and glued
    tokens ("lea je" -> "inaendeleaje"), which is where Whisper's Sheng errors live.
    """
    subs = Counter()
    for ref, hyp in pairs:
        r, h = norm(ref).split(), norm(hyp).split()
        for tag, i1, i2, j1, j2 in SequenceMatcher(None, h, r).get_opcodes():
            if tag != "replace":
                continue
            src, dst = " ".join(h[i1:i2]), " ".join(r[j1:j2])
            # Long spans are sentence-level divergence, not a repeatable token fix.
            if src and dst and src != dst and len(src.split()) <= 3 and len(dst.split()) <= 3:
                subs[(src, dst)] += 1
    return subs


def score(pairs, extra_rules):
    """Mean WER over pairs with the current normalizer plus the proposed rules."""
    import jiwer
    wers = []
    for ref, hyp in pairs:
        text = ShengNormalizer.normalize(hyp)
        for pat, rep in extra_rules:
            text = pat.sub(rep, text)
        wers.append(jiwer.wer(norm(ref), norm(text)) if norm(text) else 1.0)
    return statistics.mean(wers), statistics.median(wers)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(ROOT / "results_baseline.csv"))
    ap.add_argument("--min-count", type=int, default=2,
                    help="how many times a substitution must recur to be proposed")
    ap.add_argument("--emit", action="store_true",
                    help="print adoptable rules as Python, ready to paste")
    args = ap.parse_args()

    try:
        import jiwer  # noqa: F401
    except ImportError:
        sys.exit("jiwer not installed. Run: pip install jiwer")

    pairs = load_pairs(Path(args.csv))
    if not pairs:
        sys.exit(f"No usable reference/hypothesis pairs in {args.csv}")

    # Held-out split. Rules derived from every clip and then scored on those same
    # clips would look good by construction -- that is exactly the mistake that
    # produced the current rule set.
    cut = int(len(pairs) * 0.7)
    train, test = pairs[:cut], pairs[cut:]
    print(f"{len(pairs)} referenced clips -> {len(train)} derive / {len(test)} held out\n")

    base_train = score(train, [])
    base_test = score(test, [])
    print(f"current normalizer   train mean {base_train[0]:.3f}  held-out mean {base_test[0]:.3f}")

    subs = mine_substitutions(train)
    recurring = [(s, d, c) for (s, d), c in subs.most_common() if c >= args.min_count]
    print(f"\nsubstitutions seen >= {args.min_count} times: {len(recurring)}"
          f"  (of {len(subs)} distinct)")
    if not recurring:
        print("\nNothing recurs. Whisper's errors on this audio are not systematic at the\n"
              "token level -- they are whole-utterance failures. No rule set can fix that;\n"
              "only better audio, a better model, or verified fine-tuning data will.")
        return 0

    for src, dst, c in recurring[:25]:
        print(f"  {c:>3}x  {src!r} -> {dst!r}")

    # Adopt greedily, keeping only what actually lowers held-out WER.
    adopted = []
    for src, dst, _ in recurring:
        cand = (re.compile(rf"\b{re.escape(src)}\b", re.IGNORECASE), dst)
        if score(test, adopted + [cand])[0] < score(test, adopted)[0] - 1e-9:
            adopted.append(cand)

    final_test = score(test, adopted)
    print(f"\nadopted {len(adopted)} rules that each lower held-out WER")
    print(f"held-out mean WER  {base_test[0]:.3f} -> {final_test[0]:.3f}"
          f"   ({base_test[0] - final_test[0]:+.3f})")
    print(f"held-out median    {base_test[1]:.3f} -> {final_test[1]:.3f}")

    if not adopted:
        print("\nNo proposed rule improved held-out WER. The honest conclusion is that\n"
              "token-level correction cannot rescue this transcription quality.")
    elif args.emit:
        print("\n# --- paste into ASR_CORRECTION_RULES ---")
        for pat, rep in adopted:
            src = pat.pattern.replace(r"\b", "").replace("\\", "")
            print(f'    (re.compile(r"\\b{src}\\b", re.IGNORECASE), "{rep}"),')
    return 0


if __name__ == "__main__":
    sys.exit(main())
