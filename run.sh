#!/usr/bin/env bash
#
# run.sh - Single entry point for the Sheng S2S agent.
#
#   bash run.sh              web UI on http://localhost:7860
#   bash run.sh share        same, plus a public *.gradio.live URL for remote testing
#   bash run.sh cli          terminal mode (fallback if Gradio misbehaves)
#   bash run.sh demo         scripted walkthrough of the 6 demo scenarios
#   bash run.sh test         verification suite, no UI
#   bash run.sh review       transcript review tool on http://localhost:7870
#   bash run.sh bench        measure Edge-TTS latency on THIS machine
#   bash run.sh setup        environment setup only
#
# Sets itself up on first run. Safe to re-run.
#
# Env:
#   OPENAI_API_KEY   enables the good LLM backend (Groq: console.groq.com)
#   USE_CUDA=true    use the GPU (falls back to CPU automatically if unusable)
#   PORT             default 7860
set -uo pipefail

cd "$(dirname "$0")" || exit 1
RED=$'\033[31m'; GRN=$'\033[32m'; YEL=$'\033[33m'; DIM=$'\033[2m'; RST=$'\033[0m'
MODE="${1:-web}"
PORT="${PORT:-7860}"

# Help must work before anything else -- asking how to use the script should never
# trigger a multi-minute environment build.
case "$MODE" in
  -h|--help|help) sed -n '3,20p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
esac

# Find the interpreter. VENV overrides; otherwise take the first venv that exists,
# so an environment built under another name is reused instead of rebuilt.
VPY=""
for d in "${VENV:-}" .venv .venv311 venv env; do
  [ -z "$d" ] && continue
  for c in "$d/bin/python" "$d/Scripts/python.exe"; do
    [ -x "$c" ] && { VPY="$c"; break 2; }
  done
done
[ -n "$VPY" ] || VPY=".venv/bin/python"

# ---------------------------------------------------------------- setup if needed
needs_setup() {
  [ ! -x "$VPY" ] && return 0
  "$VPY" -c "import faster_whisper, edge_tts, gradio, pydub, soundfile" 2>/dev/null || return 0
  return 1
}
if needs_setup; then
  echo "${YEL}First run — setting up.${RST}"
  bash scripts/setup_gpu.sh || { echo "${RED}Setup failed.${RST}"; exit 1; }
  for c in .venv/bin/python .venv/Scripts/python.exe; do
    [ -x "$c" ] && { VPY="$c"; break; }
  done
  echo
fi

# ---------------------------------------------------------------- free the port
# A dead-but-listening process on the demo port is worse than a crash: the old build
# keeps answering, so a broken new build looks like it started fine. This cost real
# debugging time, so clear the port before every launch rather than trusting pkill
# (which does not reach Windows processes).
free_port() {
  local port="$1" pid
  if command -v lsof >/dev/null 2>&1; then
    pid=$(lsof -ti tcp:"$port" 2>/dev/null | head -1)
  else
    pid=$(netstat -ano 2>/dev/null | grep ":$port " | grep -i listen | awk '{print $NF}' | head -1)
  fi
  [ -z "${pid:-}" ] && return 0
  echo "${YEL}Port $port already held by PID $pid — stopping it.${RST}"
  kill -9 "$pid" 2>/dev/null \
    || powershell.exe -NoProfile -Command "Stop-Process -Id $pid -Force" 2>/dev/null
  sleep 2
}

warn_no_key() {
  [ -n "${OPENAI_API_KEY:-}" ] && return 0
  echo "${YEL}No OPENAI_API_KEY — using the offline heuristic brain (varied but shallow).${RST}"
  echo "${DIM}  export LLM_BACKEND=openai OPENAI_API_KEY=gsk_...   # console.groq.com${RST}"
  echo
}

case "$MODE" in
  web|share)
    free_port "$PORT"
    warn_no_key
    [ "$MODE" = "share" ] && export GRADIO_SHARE=true
    echo "${GRN}Starting web UI${RST} on http://localhost:$PORT"
    [ "$MODE" = "share" ] && echo "${DIM}A public *.gradio.live link will be printed below (lives ~72h).${RST}"
    echo "${DIM}Remote? From your laptop:  ssh -L $PORT:localhost:$PORT <this-host>${RST}"
    echo
    exec "$VPY" app.py
    ;;
  cli)
    warn_no_key
    exec "$VPY" cli.py
    ;;
  demo)
    warn_no_key
    exec "$VPY" cli.py --demo
    ;;
  review)
    free_port 7870
    echo "${GRN}Transcript review${RST} on http://localhost:7870"
    echo "${DIM}Corrections save to dataset/verified_transcripts.jsonl as you type.${RST}"
    echo "${DIM}This is the only thing that improves ASR accuracy — see UPDATE.md.${RST}"
    echo
    exec "$VPY" scripts/review_transcripts.py
    ;;
  bench)
    exec "$VPY" scripts/bench_tts.py
    ;;
  setup)
    exec bash scripts/setup_gpu.sh
    ;;
  test)
    fail=0
    echo "${DIM}--- components ---${RST}"
    "$VPY" test_pipeline.py 2>&1 | grep -E "✓|passed!|🎉" || fail=1
    echo
    echo "${DIM}--- echo guard ---${RST}"
    "$VPY" scripts/validate_echo_guard.py 2>&1 | grep -E "PRECISION|RECALL|PASS|FAIL" || fail=1
    echo
    echo "${DIM}--- end to end, all 6 scenarios ---${RST}"
    "$VPY" - <<'PYEOF' || fail=1
import glob, os, statistics, sys
from pipeline import SpeechToSpeechPipeline
p = SpeechToSpeechPipeline()
clips = sorted(glob.glob("assets/demo_samples/*.mp3"))
ok = audio = 0
lats, replies = [], []
for c in clips:
    r = p.process_audio(c)
    if r.get("success"):
        ok += 1
        audio += bool(r.get("audio_ok"))
        lats.append(r["latencies"]["total_glass_to_glass_ms"])
        replies.append(r["bot_response_text"])
        print(f"  [{'OK' if r.get('audio_ok') else 'NO AUDIO'}] {os.path.basename(c)}"
              f"  {r['latencies']['total_glass_to_glass_ms']:.0f}ms")
        print(f"        heard: {r['user_normalized_sheng'][:70]}")
        print(f"        reply: {r['bot_response_text'][:70]}")
    else:
        print(f"  [FAIL] {os.path.basename(c)}: {r.get('error')}")
print(f"\n  pipeline {ok}/{len(clips)} | audio {audio}/{len(clips)} "
      f"| distinct replies {len(set(replies))}/{len(replies)} "
      f"| mean {statistics.mean(lats):.0f}ms" if lats else "  no clips ran")
sys.exit(0 if ok == len(clips) and audio == len(clips) else 1)
PYEOF
    echo
    [ "$fail" = "0" ] && echo "${GRN}All checks passed.${RST}" || echo "${RED}Some checks failed.${RST}"
    exit "$fail"
    ;;
  *)
    echo "${RED}Unknown mode '$MODE'.${RST}  Try: bash run.sh --help"
    exit 1
    ;;
esac
