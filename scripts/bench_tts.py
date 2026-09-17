"""
bench_tts.py - Measure Edge-TTS latency on THIS machine, and size the timeout for it.

Edge-TTS is a network service, so its latency is a property of your link rather than
of this code. The same six phrases measured 1.3-3s on a good connection and 15-27s on
a degraded one. That matters because ShengTTSEngine.request_timeout_sec has to sit
above your real p95: set it too low and it kills slow-but-working requests and retries
them, which makes latency WORSE (a ~14s call became 26.8s that way).

Run this on the machine that will actually run the demo, not on a laptop.

Usage:
    python scripts/bench_tts.py
    python scripts/bench_tts.py --runs 20
"""
import argparse
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import AUDIO_TEMP_DIR  # noqa: E402
from tts_engine import ShengTTSEngine  # noqa: E402

PHRASES = [
    "Niaje chief, form ni gani leo mtaani?",
    "Kazi inasonga fiti sana chief, tunang'ang'ana kusaka dooh bila kuogopa!",
    "Hiyo chwani utapata chapo mbili safi na madondo kwa kibanda ya mtaa.",
    "Panda tu nganya pale stage, shuka tao alafu unisho kwa simu.",
    "Weekend ni kupika luku, kutokea pale alchemist na kuparty!",
    "Hizo raba zimepiga luku hatari msee, unakaa chief mwenyewe!",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=12)
    args = ap.parse_args()

    # A generous timeout during measurement: we want the TRUE latency, not a number
    # clipped by the very setting we are trying to choose.
    tts = ShengTTSEngine(request_timeout_sec=90.0, max_attempts=1,
                         cold_start_timeout_sec=90.0)

    out = AUDIO_TEMP_DIR / "_bench.mp3"
    print("Measuring cold start (first call pays TLS + service wake-up)...")
    t0 = time.time()
    path, meta = tts.synthesize(PHRASES[0], output_path=str(out))
    cold = time.time() - t0
    if not path:
        sys.exit(f"Edge-TTS unreachable: {meta.get('error')}. Check network access.")
    print(f"  cold start: {cold:.1f}s\n")

    times, fails = [], 0
    print(f"Measuring {args.runs} warm calls...")
    for i in range(args.runs):
        t0 = time.time()
        path, meta = tts.synthesize(PHRASES[i % len(PHRASES)], output_path=str(out))
        dt = time.time() - t0
        if path:
            times.append(dt)
            print(f"  {i+1:>3}: {dt:6.2f}s")
        else:
            fails += 1
            print(f"  {i+1:>3}: FAILED ({meta.get('error')})")
    Path(out).unlink(missing_ok=True)

    if not times:
        sys.exit("Every warm call failed. Edge-TTS is not usable from this machine.")

    times.sort()
    p50 = statistics.median(times)
    p95 = times[min(int(len(times) * 0.95), len(times) - 1)]
    print(f"\n{'='*46}")
    print(f"  cold start  {cold:6.1f}s")
    print(f"  median      {p50:6.2f}s")
    print(f"  p95         {p95:6.2f}s")
    print(f"  max         {max(times):6.2f}s")
    print(f"  failures    {fails}/{args.runs}")
    print(f"{'='*46}")

    # Headroom over p95, because the cost of being wrong is asymmetric: waiting an
    # extra few seconds is far cheaper than killing a working request and retrying.
    rec = max(15.0, round(p95 * 2 + 5))
    rec_cold = max(rec, round(cold * 1.5 + 5))
    print(f"\nRecommended for this machine:")
    print(f"    ShengTTSEngine(request_timeout_sec={rec:.0f}, cold_start_timeout_sec={rec_cold:.0f})")
    if max(times) > 10:
        print("\n  This link is slow or unstable for Edge-TTS. On demo day, rehearse on")
        print("  it and keep assets/fallback/ to hand — that is exactly what it is for.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
