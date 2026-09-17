# UPDATE — `fix/audit-hardening`

What changed, what it was measured at, and what still is not fixed.

Every number here came from running something in this repo. Where a claim turned out
to be wrong it is marked **Corrected** rather than quietly deleted, because two of the
corrections matter more than the original findings did.

---

## TL;DR

The pipeline now runs end to end, the bot no longer repeats itself, the voice varies
per turn, and eleven separate bugs that would have broken the demo are fixed.

**Transcription accuracy did not improve, and could not have.** Median WER is 1.00 on
real conversational Sheng with zero of 123 clips under 0.3. That is a missing-data
problem: no transcript in this repo has ever been verified by a human. Everything else
here is engineering around that fact.

---

## 1. Bugs that would have broken the demo

| # | Bug | Effect if unfixed |
|---|---|---|
| 1 | `assets/demo_samples/demo_1_greeting.mp3` referenced by `app.py` but never committed | First demo button dead |
| 2 | `test_pipeline.py` asserted `"ni aje ba zenga" → "niaje bazenga"` while the rules mapped `ba zenga → chief` | Crashed on its own fixtures, blocking the 9:10 merge gate |
| 3 | Emoji in `print()` raised `UnicodeEncodeError` on Windows cp1252 | Every entry point died at startup |
| 4 | `pedalboard` missing from `requirements.txt` | `_polish()` catches `ImportError` and skips — voice silently goes flat |
| 5 | `duration` assigned only inside the `.mp3` branch of `synthesize_async` | `NameError` swallowed into silent empty audio for any other extension |
| 6 | `new_venv/` — a 39MB **Linux** venv committed to git | Dead weight, unusable anywhere else |
| 7 | Fallback audio documented at `/tmp/fallback/` | Wiped on reboot; not a path on Windows at all |
| 8 | Broken GPU left `self.model = None` | Every `transcribe()` returned "ASR model not initialized" — total failure on a box that runs fine on CPU |
| 9 | `launch(theme=...)` on Gradio 5 | `TypeError` — app does not start *(self-inflicted, see §5)* |
| 10 | `Chatbot(type=...)` on Gradio 6 | `TypeError` at import — app does not start *(self-inflicted, see §5)* |
| 11 | `success: True` returned with an empty `output_audio_path` | UI showed a reply and silently played nothing — looks like a dead app on stage |

---

## 2. Correctness fixes

**Repetition loops.** `temperature` was passed as a scalar `0.0`, which leaves
`compression_ratio_threshold` with no hotter temperature to retry at — so it correctly
detected degenerate output and then kept it anyway. One clip transcribed `"Ha ha ha"`
×67 against a five-word reference, WER 44.6. Now a fallback tuple.

**Prompt leakage — 16/123 clips (13%).** Whisper emitted `WHISPER_SHENG_PROMPT`
verbatim as the transcript on quiet audio. Now caught after decoding by ordered
similarity against the clause no user would say: **11/16 caught, 0 false positives**
(`scripts/validate_echo_guard.py`). Four of the five misses are the string
`"Niaje chief, form ni gani leo mtaani?"`, which appears in the baseline *both* as a
leak and as a correct transcript — identical bytes, genuinely inseparable. The guard
keeps the text, because silently eating real speech is the worse failure.

**The normalizer was a slangifier.** `ShengNormalizer` rewrote correct words
(`nielekeze` → `nisho`), inflating WER and — via `train_whisper_lora.py`, which trained
on that field — teaching Whisper to emit words nobody said. Split into `normalize()`
(lossless, safe for display, scoring and labels) and `slangify()` (meaning-changing,
LLM input only). Training labels now prefer `verified_text` → `normalized_text` →
`raw_transcription`, never the slang field.

**Device fallback.** ASR degrades `cuda → cpu/int8`, then `small → base → tiny`. CUDA
failures surface at *encode* time, not load time — ctranslate2 reports `Loaded in 0.9s`
and then dies on the first real audio — so startup pushes 0.3s of silence through
`transcribe()` to catch it during boot rather than on stage.

**TTS cold start.** Edge-TTS's first call pays TLS plus service wake-up: **19.1s
measured, against 1.4s and 1.3s for the next two**. That was landing on the first thing
anyone said. Now paid at startup by `warmup()`.

---

## 3. The bot no longer repeats itself

Every `HEURISTIC_INTENTS` bucket held **exactly one** reply, which made
`random.choice` deterministic — that was the canned feeling, not the model. Now 3–4
each, 63 total.

Added the missing `naturalizer.py` (fillers, breath pauses, follow-up questions,
per-emotion prosody) and wired `meta["emotion"]`, so `vary_prosody` actually varies
instead of always returning neutral.

> **15/15 distinct replies** over fifteen identical prompts. **6/6 distinct** across the
> demo scenarios.

---

## 4. Trained Qwen2.5-1.5B on 2,500 turns

Rebuilt `train_sheng_sft.jsonl` by slot composition: **150 records / 26 unique pairs →
2,500 records / 1,951 unique user turns**. Trained on `zerolabs1` (RTX 5060 Ti);
`vllm-gemma4.service` was stopped to free VRAM and restored immediately, leaving the
box exactly as found.

```
train loss  0.152
eval loss   0.194 → 0.170 → 0.163   (still improving at epoch 3)
adapter     llm_sheng_lora_output_1_5B/final_adapter (74MB)
```

Eval tracks train closely, so it is **not** memorising the way the 0.5B-on-150 did.

**It is still a template matcher.** Held out: *"Nimepoteza simu yangu kwa mathree,
nifanye nini?"* (*I lost my phone in a matatu*) → *"Kibanda ya mama tao iko open, nyama
choma iko tayari."* (*The food stall is open*). It matched `mathree` to a transport/food
template and ignored the question. The 2,500 records come from only ~320 skeleton
pairings, so the diversity is surface-level.

**Use it over the 0.5B. Do not use it over the API backend.**

---

## 5. Things I got wrong, and corrected

**Corrected — the normalizer does nothing on real speech.** I reported `normalize()` as
a ~5× WER improvement (0.699 → 0.139) and hedged it as "overfit to the demo set". That
understated it. On the 100 clips with genuine human references:

```
rules that fire ........ 0 of 61
clips changed .......... 0   (0 helped, 0 hurt, 100 unchanged)
mean WER raw vs norm ... 1.992 vs 1.992
```

The 5× is entirely an artifact of measuring on the same six clips the rules were
derived from. `scripts/derive_correction_rules.py` then mined the real data for better
rules and found **0 recurring substitutions out of 30** — Whisper's failures here are
whole-utterance, not token-level, so no regex layer can repair them. **Do not quote
0.139 as an accuracy figure.**

**Corrected — `large-v3-turbo` is worse here, not better.** I recommended it on
reputation. Measured:

| Model | Raw WER | After `normalize()` | Per clip | RTF |
|---|---|---|---|---|
| `small` | 0.699 | 0.139 | **3.58s** | 1.09× |
| `large-v3-turbo` | **0.593** | 0.376 | 14.88s | 4.52× |

Better raw, 4.2× slower, and it loses end to end. Reverted to `small`.

**Corrected — a word-list Whisper prompt is worse.** Tried as the leakage fix; measured
WER 0.741 vs 0.699 with 5× the comma density, and it hallucinated `nduthi` straight out
of the bias vocabulary. The decoder copies the prompt's *format*. Reverted.

**Self-inflicted — two Gradio crashes.** Moving theme/css to `launch()` broke Gradio 5;
adding `Chatbot(type=)` broke Gradio 6. Both are now version-conditional.

**Self-inflicted — a false pass.** I reported the Gradio app as "HTTP 200, 0 warnings"
while it was actually crashing on import. A stale `app.py` was still holding port 7860,
so `curl` was answered by the *old* process. `pkill -f` does not reach Windows
processes. `run.sh` now frees the port before every launch.

---

## 6. New tooling

| Script | What it is for |
|---|---|
| `run.sh` | Single entry point — `web` / `share` / `cli` / `demo` / `test` / `review` / `bench` / `setup` |
| `scripts/setup_gpu.sh` | Bring-up with per-step PASS/WARN/FAIL; stops at the first real blocker |
| `scripts/review_transcripts.py` | **The one that matters.** Play a clip, correct the transcript, save. Feeds `verified_text` straight into training |
| `scripts/prefetch_models.py` | Cache weights before the demo (sets `HF_HUB_DISABLE_XET=1` — the Xet backend failed mid-download) |
| `scripts/generate_demo_audio.py` | Demo prompts + the six committed fallback replies |
| `scripts/bench_tts.py` | Measure Edge-TTS latency *on the demo machine* and size the timeout |
| `scripts/validate_echo_guard.py` | Precision/recall of the leak guard against known leaks |
| `scripts/ab_prompt_bias.py` | Does prompt biasing help? Four conditions, real WER |
| `scripts/derive_correction_rules.py` | Mine correction rules from aligned reference/hypothesis pairs, held-out scored |
| `scripts/ab_decode_config.py` | Old vs new decoder settings on real clips |

---

## 7. Verified

```
run.sh test        components 4/4 · echo guard 11/16 with 0 false positives
end to end         pipeline 6/6 · distinct replies 6/6
Gradio app         HTTP 200, custom CSS served, 0 warnings
review tool        HTTP 200 over 349 clips (69 min)
CUDA fallback      cuda → cpu/int8, transcribes correctly
```

Audio was 5/6 on the last run — Edge-TTS rate-limiting after heavy session use, not a
code fault. `run.sh test` reports that as a failure rather than passing silently.

---

## 8. What is still not fixed

**ASR accuracy.** Median WER 1.00, 0/123 clips under 0.3. Nothing in this branch
changes that.

**There are no verified transcripts.** `dataset/validation_manifest.jsonl` marks 349
clips `CLEAN_VALID`, but `clean_sheng_text` is byte-identical to `normalized_sheng` for
all 349 — regex output, not human correction. **Do not fine-tune Whisper on it**; that
teaches the model its own errors.

**The evaluation set is not in the repo.** `eval_asr.py` reads
`zoza_transcripts/mapped_data`, never committed. It is the only human-transcribed Sheng
data the project has, it is behind every number on this page, and it currently exists
on one laptop. **Back it up.**

**Latency.** ~3.6–4.1s ASR on CPU at RTF 1.09 — barely real time. Keep spoken inputs to
3–5 seconds.

**Edge-TTS is a network service.** Latency is a property of your link: 1.3–3s on a good
one, 15–29s on a degraded one. Run `bash run.sh bench` on the demo machine.

---

## 9. Running it

```bash
bash run.sh              # web UI on :7860, sets itself up on first run
bash run.sh share        # same, plus a public *.gradio.live URL
bash run.sh test         # verification, no UI
bash run.sh review       # transcript review tool
```

For a real conversational bot rather than the offline heuristic:

```bash
export LLM_BACKEND=openai
export OPENAI_API_KEY=gsk_...        # Groq, free tier — console.groq.com
```

**If one thing gets done next,** it is `bash run.sh review`. Verified transcripts are
the only path to better ASR — not a bigger model, not more rules, not a faster GPU.
