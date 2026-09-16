"""
generate_demo_audio.py - Produce every audio file the demo depends on.

Run this the night BEFORE the demo, with network available. Two outputs:

  assets/demo_samples/   the six preset prompts app.py's demo buttons load. One of
                         these (demo_1_greeting.mp3) was referenced by app.py but
                         had never been committed, which broke the first button.

  assets/fallback/       a canned bot REPLY for each of the six scenarios. If the
                         live pipeline dies on stage, play these instead. This is
                         the difference between "the demo failed" and "the demo had
                         a pre-recorded backup". Committed to the repo on purpose --
                         a backup that lives in /tmp is gone after a reboot.

Usage:
    python scripts/generate_demo_audio.py            # only what is missing
    python scripts/generate_demo_audio.py --force    # regenerate everything
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tts_engine import ShengTTSEngine  # noqa: E402
from config import BASE_DIR, VOICE_OPTIONS  # noqa: E402

# The user side of each scenario. Spoken by a different voice from the bot so the
# demo audio is clearly "someone asking", not the bot talking to itself.
DEMO_PROMPTS = {
    "demo_1_greeting": "Niaje chief, form ni gani leo mtaani?",
    "demo_2_food": "Niko na chwani nataka kubuy lunch, unapendekeza nini?",
    "demo_3_transport": "Nisho pahali keja yako iko ndio nipande nganya tao.",
    "demo_4_drip": "Hizo viatu mpya zinakutoa aje?",
    "demo_5_work": "Hustle inaendeleaje leo chief?",
    "demo_6_weekend": "Hii weekend form iko wapi?",
}

# The bot side. Used only if the live pipeline breaks.
FALLBACK_REPLIES = {
    "1_greeting": "Eeeh chief, form ni kuchill tu kejani na mbogi, tukipiga stori za luku na hustle. Rada yako?",
    "2_kibanda": "Hiyo chwani utapata chapo mbili safi na madondo kwa kibanda ya mtaa. Utashiba fiti sana!",
    "3_nganya": "Panda tu nganya pale stage, shuka tao alafu unisho kwa simu nikupe rada safi ya street!",
    "4_drip": "Hizo raba zimepiga luku hatari msee, unakaa chief mwenyewe!",
    "5_hustle": "Kazi inasonga fiti sana chief, tunang'ang'ana kusaka dooh bila kuogopa!",
    "6_weekend": "Weekend ni kupika luku, kutokea pale alchemist na kuparty like there is no tomorrow!",
}

USER_VOICE = VOICE_OPTIONS["Nairobi Gen Z Guy (Chilemba - Urban English/Sheng)"]


def render(tts, items, out_dir, voice, force, label):
    out_dir.mkdir(parents=True, exist_ok=True)
    made = skipped = failed = 0
    print(f"\n=== {label} -> {out_dir} ===")
    for name, text in items.items():
        path = out_dir / f"{name}.mp3"
        if path.exists() and not force:
            print(f"  - {path.name} already exists, skipping")
            skipped += 1
            continue
        result, meta = tts.synthesize(text, output_path=str(path), voice=voice)
        if result and path.exists() and path.stat().st_size > 0:
            print(f"  OK {path.name}  ({meta['tts_time_ms']:.0f} ms, {path.stat().st_size} bytes)")
            made += 1
        else:
            print(f"  FAILED {path.name}: {meta.get('error', 'unknown error')}")
            failed += 1
    return made, skipped, failed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="regenerate files that already exist")
    args = ap.parse_args()

    tts = ShengTTSEngine()
    totals = [0, 0, 0]

    for items, out_dir, voice, label in [
        (DEMO_PROMPTS, BASE_DIR / "assets" / "demo_samples", USER_VOICE, "Demo prompts (user side)"),
        (FALLBACK_REPLIES, BASE_DIR / "assets" / "fallback", None, "Fallback replies (bot side)"),
    ]:
        made, skipped, failed = render(tts, items, out_dir, voice, args.force, label)
        totals = [totals[0] + made, totals[1] + skipped, totals[2] + failed]

    print(f"\nGenerated {totals[0]}, skipped {totals[1]}, failed {totals[2]}.")
    if totals[2]:
        print("Some files failed. Edge-TTS needs network access -- check your connection and rerun.")
        return 1
    print("All demo and fallback audio is present.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
