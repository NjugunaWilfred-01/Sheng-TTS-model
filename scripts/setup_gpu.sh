#!/usr/bin/env bash
#
# setup_gpu.sh - Get the Sheng S2S pipeline running, on GPU or CPU, from scratch.
#
#   bash scripts/setup_gpu.sh
#
# Idempotent: safe to re-run. Every step reports PASS/WARN/FAIL and the script stops
# at the first thing that actually blocks the demo, so you get one clear error rather
# than a wall of output.
#
# A broken GPU is a WARN, not a FAIL: the pipeline runs correctly on CPU, just slower,
# and asr_engine falls back automatically. Only genuine blockers stop the script.
#
# Env you can override:
#   USE_CUDA=true        use the GPU (verified end-to-end before it is trusted)
#   WHISPER_MODEL_SIZE   default small   (see the note under step 6)
#   OPENAI_API_KEY       enables the good LLM backend; without it you get heuristic
#   PORT                 default 7860
set -uo pipefail

RED=$'\033[31m'; GRN=$'\033[32m'; YEL=$'\033[33m'; DIM=$'\033[2m'; RST=$'\033[0m'
ok(){   echo "${GRN}  PASS${RST}  $*"; }
warn(){ echo "${YEL}  WARN${RST}  $*"; }
die(){  echo "${RED}  FAIL${RST}  $*"; echo; echo "${RED}Stopped.${RST} Fix the above and re-run."; exit 1; }
step(){ echo; echo "${DIM}--- $* ---${RST}"; }

cd "$(dirname "$0")/.." || die "cannot find repo root"
echo "Repo: $(pwd)"

# ---------------------------------------------------------------- 1. the code
step "1. Code version"
BRANCH=$(git branch --show-current 2>/dev/null || echo "?")
if [ "$BRANCH" != "fix/audit-hardening" ]; then
  warn "on '$BRANCH' — the fixes live on fix/audit-hardening"
  echo "        Run: git fetch origin && git checkout fix/audit-hardening"
  echo "        Without it: test_pipeline crashes on its own fixtures, demo button 1"
  echo "        is dead, pedalboard is missing, and emoji print() can kill startup."
else
  ok "on fix/audit-hardening ($(git rev-parse --short HEAD))"
fi

# ---------------------------------------------------------------- 2. system deps
step "2. System dependencies"
command -v ffmpeg >/dev/null && ok "ffmpeg $(ffmpeg -version 2>/dev/null | head -1 | cut -d' ' -f3)" \
  || die "ffmpeg missing. Run: sudo apt install -y ffmpeg"

# nvidia-smi exits non-zero on a driver/library mismatch but still prints to stdout.
# Check the exit status and the text: an earlier version of this script printed the
# error string as a PASS and then crashed comparing it as an integer.
GPU_OK=0
if command -v nvidia-smi >/dev/null; then
  GPU=$(nvidia-smi --query-gpu=name,memory.free --format=csv,noheader 2>&1 | head -1)
  if [ $? -eq 0 ] && [ -n "$GPU" ] && ! echo "$GPU" | grep -qiE "failed|error|mismatch"; then
    ok "GPU: $GPU"
    GPU_OK=1
    FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1)
    case "${FREE:-x}" in
      ''|*[!0-9]*) : ;;
      *) [ "$FREE" -lt 3000 ] && warn "only ${FREE}MiB free — something else is on the GPU" ;;
    esac
  else
    warn "GPU present but UNUSABLE: $GPU"
    if echo "$GPU" | grep -qi mismatch; then
      echo "        Driver/library version mismatch: the loaded kernel module and the"
      echo "        userspace libraries are different versions — usually a driver update"
      echo "        without a reboot."
      echo "          sudo reboot"
      echo "        or, if nothing else is using the GPU:"
      echo "          sudo rmmod nvidia_uvm nvidia_drm nvidia_modeset nvidia && sudo modprobe nvidia"
    fi
    echo "        ${DIM}Not a blocker — the pipeline runs correctly on CPU.${RST}"
  fi
else
  warn "no nvidia-smi — running on CPU (works, just slower)"
fi

# ---------------------------------------------------------------- 3. python
step "3. Python environment"
PY=""
for c in python3.12 python3.11 python3; do
  if command -v $c >/dev/null; then
    V=$($c -c 'import sys;print("%d.%d"%sys.version_info[:2])' 2>/dev/null)
    case "$V" in 3.9|3.10|3.11|3.12|3.13) PY=$c; break;; esac
  fi
done
[ -n "$PY" ] || die "no suitable Python. faster-whisper needs 3.9-3.13 (no 3.14 wheels yet)."
ok "interpreter: $PY ($($PY -c 'import sys;print("%d.%d"%sys.version_info[:2])'))"

[ -d .venv ] || { $PY -m venv .venv || die "venv creation failed"; ok "created .venv"; }
VPY="$(pwd)/.venv/bin/python"
$VPY -m pip install -q --upgrade pip >/dev/null 2>&1

# ---------------------------------------------------------------- 4. packages
step "4. Python packages"
if ! $VPY -c "import faster_whisper, edge_tts, gradio, pydub, soundfile" 2>/dev/null; then
  echo "  installing (a few minutes)..."
  $VPY -m pip install -q -r requirements.txt || die "pip install failed"
fi
$VPY -c "import faster_whisper, edge_tts, gradio, pydub, soundfile" 2>/dev/null \
  && ok "core packages present" || die "packages still missing after install"

# pedalboard is the one that fails silently: _polish() catches ImportError and skips,
# so the voice quietly reverts to sounding like a vacuum recording.
$VPY -c "import pedalboard" 2>/dev/null && ok "pedalboard (audio polish active)" \
  || warn "pedalboard missing — voice will sound flat but the demo still runs"

# ---------------------------------------------------------------- 5. device
step "5. Compute device"
CUDA_CHECK='
import sys, numpy as np
from faster_whisper import WhisperModel
m = WhisperModel("tiny", device="cuda", compute_type="float16")
list(m.transcribe(np.zeros(4800, dtype=np.float32), language="sw", beam_size=1)[0])
print("cuda-ok")
'
if [ "${USE_CUDA:-}" = "true" ]; then
  if [ "$GPU_OK" = "1" ]; then
    # Loading succeeds even when CUDA is unusable — ctranslate2 only fails at ENCODE
    # time with "Library libcublas.so.12 is not found". So run real inference.
    if $VPY -c "$CUDA_CHECK" >/dev/null 2>&1; then
      ok "CUDA verified end-to-end (load + inference)"
    else
      warn "CUDA loads but cannot run inference — almost always missing runtime libs."
      echo "        Fix:  $VPY -m pip install nvidia-cublas-cu12 nvidia-cudnn-cu12"
      echo "        Then re-run this script with USE_CUDA=true."
      echo "        ${DIM}Using CPU meanwhile; asr_engine falls back automatically.${RST}"
      export USE_CUDA=false
    fi
  else
    warn "USE_CUDA=true but the GPU is unusable (see step 2). Using CPU."
    export USE_CUDA=false
  fi
else
  ok "running on CPU (set USE_CUDA=true once the GPU is healthy)"
fi

# ---------------------------------------------------------------- 6. models
step "6. Model weights"
# WHISPER_MODEL_SIZE defaults to small deliberately: ASR_CORRECTION_RULES were derived
# against small's error patterns, and on the demo phrases small+rules scored WER 0.139
# vs large-v3-turbo's 0.376. Turbo has better RAW WER (0.593 vs 0.699) so it likely
# wins on unseen speech -- but the rules must be re-derived for it first.
echo "  WHISPER_MODEL_SIZE=${WHISPER_MODEL_SIZE:-small}"
$VPY scripts/prefetch_models.py || die "model prefetch failed (see error above)"
ok "whisper weights cached"

# ---------------------------------------------------------------- 7. audio
step "7. Demo and fallback audio"
$VPY scripts/generate_demo_audio.py >/dev/null 2>&1
N_DEMO=$(ls assets/demo_samples/*.mp3 2>/dev/null | wc -l)
N_FB=$(ls assets/fallback/*.mp3 2>/dev/null | wc -l)
[ "$N_DEMO" -ge 6 ] && ok "$N_DEMO demo prompts" || warn "$N_DEMO/6 demo prompts (Edge-TTS needs network)"
[ "$N_FB" -ge 6 ]   && ok "$N_FB fallback replies" || warn "$N_FB/6 fallback replies"

# ---------------------------------------------------------------- 8. brain
step "8. LLM backend"
if [ -n "${OPENAI_API_KEY:-}" ]; then
  ok "OPENAI_API_KEY set -> live API backend"
else
  warn "no OPENAI_API_KEY — falls back to the offline heuristic (varied but shallow)."
  echo "        export LLM_BACKEND=openai"
  echo "        export OPENAI_API_KEY=gsk_...   # Groq, free tier"
fi

# ---------------------------------------------------------------- 9. verify
step "9. Verification"
$VPY scripts/validate_echo_guard.py >/dev/null 2>&1 && ok "echo guard" || warn "echo guard check failed"
$VPY test_pipeline.py >/tmp/sheng_test.log 2>&1 \
  && ok "test_pipeline 4/4" \
  || die "test_pipeline failed — see /tmp/sheng_test.log"

echo
echo "${GRN}Ready.${RST}  Start it with:"
echo
[ "${USE_CUDA:-false}" = "true" ] && echo "    export USE_CUDA=true"
echo "    export LLM_BACKEND=openai OPENAI_API_KEY=gsk_..."
echo "    $VPY app.py"
echo
echo "  Then open http://localhost:${PORT:-7860}"
echo "  Terminal fallback if Gradio misbehaves:  $VPY cli.py --demo"
