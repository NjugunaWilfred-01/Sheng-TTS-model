"""
validate_echo_guard.py - Check ShengASREngine._strip_prompt_echo against real data.

The guard has to satisfy two opposing constraints:

  RECALL     catch the 16 transcripts in results_baseline.csv that are the
             initial_prompt read back verbatim
  PRECISION  never discard a genuine utterance -- and the hard case is real, because
             the prompt is built from exactly the vocabulary users say. The demo
             phrase "Niaje chief, form ni gani leo mtaani?" is word-for-word the
             prompt's first sentence AND a correct transcript.

Any change to WHISPER_SHENG_PROMPT or to the guard thresholds must be re-run here.
Exits non-zero on a false positive, which is the failure that silently eats speech.

Usage:
    python scripts/validate_echo_guard.py
"""
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sheng_lexicon import WHISPER_SHENG_PROMPT  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

# Genuine transcripts that MUST survive. The first is the adversarial case: it is
# identical to the prompt's opening sentence and is nonetheless correct.
MUST_KEEP = [
    "Niaje chief, form ni gani leo mtaani?",
    "Niko na choani na taka kubui lungi, una pendekeza nini.",
    "Nisho pahalike jaya ko ikondio nipanda nganya tao.",
    "Hizo viatum piyazinakutoa aje.",
    "Usle inaende lea je leo chiefi.",
    "Hiwe Ken, form Ikoapi.",
    "Welcome to another episode leo tukuna dintrimen, ami.",
    "Wata, true kabisa kresi.",
]


class _Stub:
    """Exercise the real guard without loading a Whisper model."""
    initial_prompt = WHISPER_SHENG_PROMPT
    _strip_prompt_echo = None  # bound below


def load_guard():
    import asr_engine
    _Stub._strip_prompt_echo = asr_engine.ShengASREngine._strip_prompt_echo
    return _Stub()


def known_leaks():
    """Hypotheses from the baseline run that are prompt regurgitation."""
    csv_path = ROOT / "results_baseline.csv"
    if not csv_path.exists():
        return []
    markers = ("nisho vile tunaingia tao", "niko fiti na mbogi", "form ni gani leo mtaani")
    leaks = []
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            h = row["hypothesis"]
            if any(m in h.lower() for m in markers):
                leaks.append(h)
    return leaks


def main():
    guard = load_guard()
    leaks = known_leaks()

    print(f"Prompt: {WHISPER_SHENG_PROMPT}\n")

    kept_wrongly = [t for t in MUST_KEEP if not guard._strip_prompt_echo(t)]
    caught = [t for t in leaks if not guard._strip_prompt_echo(t)]

    print(f"PRECISION  genuine transcripts kept: "
          f"{len(MUST_KEEP) - len(kept_wrongly)}/{len(MUST_KEEP)}")
    for t in kept_wrongly:
        print(f"  FALSE POSITIVE (speech discarded): {t}")

    print(f"RECALL     known leaks caught: {len(caught)}/{len(leaks)}")
    for t in leaks:
        if guard._strip_prompt_echo(t):
            print(f"  missed: {t[:80]}")

    if kept_wrongly:
        print("\nFAIL: the guard is discarding real speech. Raise the thresholds.")
        return 1
    if leaks and not caught:
        print("\nFAIL: the guard catches nothing. Lower the thresholds.")
        return 1
    print("\nPASS: no false positives; leaks are being caught.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
