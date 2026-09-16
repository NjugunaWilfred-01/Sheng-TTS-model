# Audit Results — Sheng S2S Agent

Deep audit of the implemented architecture against `Work_Plan.md`, plus the fixes
applied. Every number here was measured on this repo, not estimated.

**Environment audited on:** Windows 11, CPU-only (no CUDA), Python 3.11, no API keys set.

---

## 1. Work-plan status at audit time

| Plan item | Target | Found |
|---|---|---|
| Silero VAD replaces pydub | required | ❌ still `split_on_silence` |
| Pseudo-label with large-v3 | `large-v3` | ❌ still `"small"` |
| Hand-verified clips | ≥150 with `verified_text` | ❌ **0** |
| `train_sheng_asr_cleaned.jsonl` | ≥150 records | ❌ 25 |
| `naturalizer.py` | created + wired | ❌ **did not exist** |
| `meta["emotion"]` | populated | ❌ never set |
| LoRA gen params | bf16 / 120 tok / rep-penalty | ❌ fp32 / 60 / none |
| `train_sheng_sft.jsonl` | ≥2,000 lines | ❌ 150 |
| LLM LoRA base | Qwen2.5-1.5B | ❌ 0.5B |
| Whisper LoRA base | "do **not** use tiny" | ❌ trained on `whisper-tiny`, never wired in |
| System prompt rewrite | stricter prompt | ❌ old prompt |

---

## 2. The central finding: there is no verified transcription data

`dataset/validation_manifest.jsonl` holds 349 clips / 69 minutes, all marked
`CLEAN_VALID`. But:

```
clean_sheng_text != normalized_sheng :  0 / 349
clean_sheng_text != raw_transcription : 122 / 349
```

Every one of those 122 differences is a regex substitution. `clean_sheng_text` is
**machine output, not human correction**. The label `CLEAN_VALID` overstates it.

This is why no Whisper fine-tune should be run on this data: it would train the model
to reproduce its own errors. It also means the ASR ceiling cannot be raised before the
demo — only specific decoder failures can be removed.

**Separately: the only human-transcribed data is not in the repo.** `eval_asr.py` reads
`zoza_transcripts/mapped_data`, which was never committed. That is the 123-clip
reference set behind every number below, and it currently exists on one laptop. Back it
up — it is the most valuable asset the project has.

---

## 3. Baseline ASR quality (`results_baseline.csv`, 123 clips)

| Metric | Value |
|---|---|
| Median WER | **1.00** |
| Mean WER (valid refs, n=106) | 1.94 |
| Clips with WER ≤ 0.3 | **0 / 123** |
| Best single clip | 0.40 |
| Mean inference | 6,778 ms/clip |

Two decoder-level failure modes, both fixed:

**Prompt leakage — 16/123 clips (13%).** The transcript is the `initial_prompt` read
back verbatim. Whisper treats `initial_prompt` as text it already wrote and continues
it, so on quiet or non-speech audio it emits the prompt as the transcript.

**Repetition loops.** `temperature=0.0` was passed as a **scalar**, which leaves
`compression_ratio_threshold` with no hotter temperature to retry at — so detected
garbage was kept anyway. One clip: `"Ha ha ha"` ×67 against a 5-word reference (WER 44.6).

---

## 4. Experiment: does prompt biasing help? (`scripts/ab_prompt_bias.py`)

The six demo clips are TTS-synthesised from known strings, so their WER is exact.
Mean WER over all six, `whisper-small`:

| Condition | raw WER | after `normalize()` | commas/word |
|---|---|---|---|
| no prompt | 0.902 | 0.681 | 0.054 |
| **sentence prompt** | **0.699** | **0.139** | 0.082 |
| word list | 0.741 | 0.473 | **0.396** |
| disjoint sentence | 0.891 | — | 0.049 |

Three conclusions, and the second one reversed a change made during this audit:

1. **A word-list prompt is worse, not better.** It was tried as a leakage fix. The
   decoder copies the prompt's *format*, so a list prompt yields list-shaped
   transcripts — 5× the comma density, and it hallucinated `nduthi` straight out of the
   bias vocabulary. Commas also break the multi-word repair rules, which need
   `hiwe ken`, not `hiwe, ken`. **Reverted.**

2. **The sentence prompt's headline win is partly an artifact.** It literally contains
   demo clip 1, which therefore scores WER 0.000. Excluding that clip the raw spread
   collapses to noise: 0.883 / 0.838 / 0.803. Generic Sheng prompt biasing buys little
   on its own — the `disjoint` condition (same style, no shared phrasing) scored 0.891,
   essentially tying "no prompt".

3. **`normalize()` is doing the real work** — a ~5× WER reduction (0.699 → 0.139).
   *Heavily caveated:* those rules were hand-written against these exact demo phrases
   (`viatum piya`, `hiwe ken`, `pahalike`, `ikondio`), so this is overfit to the demo
   set and will not generalise at this magnitude to unseen speech.

---

## 5. Echo guard (`scripts/validate_echo_guard.py`)

Leakage is now handled after decoding rather than by weakening the prompt.

**Result: 11/16 known leaks caught, 0 false positives on 8 known-good transcripts.**

Four of the five misses are the string `"Niaje chief, form ni gani leo mtaani?"`, which
in the baseline appears *both* as a leak on silent clips *and* as the correct transcript
of real speech — the same bytes, with no signal to separate them. The guard resolves
that ambiguity towards keeping the text, because silently discarding real speech is the
worse failure on stage. The fifth miss is a four-word fragment.

Matching uses ordered similarity rather than substring or word-overlap: echoes come back
garbled (`"Nisho vile na mbogi, nisho vile tunaingia tao na nganya"`), which substrings
miss; and word-overlap is useless here because the prompt is built from exactly the
vocabulary users say.

---

## 6. Fixes applied

**ASR** — `temperature` scalar → fallback tuple; added `log_prob_threshold`; post-decode
echo guard; graceful degradation through `small → base → tiny` when a model will not
load. Model default **stayed on `small`** — see §9.

**Text pipeline** — split the single-stage map into `ASR_CORRECTION_RULES` (lossless
repairs, safe for display / WER / training labels) and `SHENG_SLANG_RULES`
(meaning-changing, LLM input only). `pipeline.py` now displays `normalize()` output and
sends `slangify()` output to the LLM. `train_whisper_lora.py` prefers
`verified_text` → `normalized_text` → `raw_transcription`, never the slang field.

**LLM** — wrote `naturalizer.py` (fillers, breath pauses, follow-ups, per-emotion
prosody) and wired it in; `meta["emotion"]` now populated, so `vary_prosody` actually
varies. Expanded every `HEURISTIC_INTENTS` bucket from **1** reply to 3–4 (63 total) —
single-element lists made `random.choice` deterministic, which *was* the canned feeling.
Backend default → API with automatic heuristic fallback. LoRA params: bf16 on CUDA,
120 tokens, `repetition_penalty=1.15`, `no_repeat_ngram_size=3`; inputs moved to device.

**TTS** — fixed a `NameError`: `duration` was assigned only inside the `.mp3` branch, so
any other extension raised and got swallowed into a silent empty-audio failure. Added a
12s timeout with 3 attempts — Edge-TTS is a network service and measured 47s / 28s /
2.2s on three identical calls.

**Data engine** — Silero VAD with pydub fallback; pseudo-labeler `small` → `large-v3`;
`train_sheng_sft.jsonl` regenerated by slot composition: **150 records / 26 unique pairs
→ 2,500 records / 1,951 unique user turns / 1,342 unique bot turns**.

**Demo blockers** — `demo_1_greeting.mp3` was referenced by `app.py` but had never been
committed, so the first demo button was dead; generated, and `app.py` now filters
missing samples instead of trusting the dict. `test_pipeline.py` asserted
`"ni aje ba zenga" → "niaje bazenga"` while the rules mapped `ba zenga → chief` — it
crashed on its own fixtures, blocking the 9:10 merge gate. Emoji in `print()` raised
`UnicodeEncodeError` on Windows cp1252 consoles, killing every entry point at startup;
`config.py` now forces UTF-8. `requirements.txt` was missing `pedalboard` (so audio
polish silently no-op'd), `peft`, `transformers`, `jiwer`. Removed `new_venv/` — a 39MB
**Linux** venv (`home = /usr/bin`) committed to git. Fallback audio moved from `/tmp/`
(wiped on reboot, not a Windows path) to committed `assets/fallback/`.

---

## 7. Verification

```
test_pipeline.py            4/4 pass
validate_echo_guard.py      11/16 leaks caught, 0 false positives
End-to-end, 3 identical inputs → 3 DISTINCT replies (was: always identical)
TTS, 3 phrases              1559 / 1500 / 1460 ms
ASR, demo_1 (small)         3.6s, transcript exact
```

---

## 8. Honest position for tomorrow

**Ready:** the pipeline runs end-to-end, the bot no longer repeats itself, the voice is
polished and varies per turn, every demo button works, fallback audio is committed, and
the merge gate passes.

**Not flawless, and cannot be by tomorrow:** the transcriber is the weak half and that is
a *data* problem, not a config one. Median WER 1.00 with zero clips under 0.3. The
decoder fixes remove specific failure modes; they do not make open-ended conversational
Sheng reliable. Closing that gap needs hand-verified transcripts, which do not exist yet.

**Biggest remaining risks on stage:**
1. **Latency.** CPU-only, ~3.6s ASR on `small` and more on `large-v3-turbo`. Keep every
   spoken input to 3–5 seconds, as `DEMO_SCRIPT.md` says.
2. **Model weights must be cached before the demo.** `scripts/prefetch_models.py` now
   sets `HF_HUB_DISABLE_XET=1` itself — the Xet backend failed mid-download here and
   left a partial cache. `asr_engine` also degrades through `small → base → tiny`
   rather than hard-failing, so a bad download costs quality, not the whole demo.
3. **Edge-TTS throttling** — bounded now, but a triple timeout still costs ~36s. The
   committed fallback audio is the answer.


---

## 9. Model swap: measured, and rejected

`large-v3-turbo` was the intended upgrade. It is worse on this hardware.

| Model | Load | Raw WER | After `normalize()` | Per clip | RTF |
|---|---|---|---|---|---|
| `small` | 2.5s | 0.699 | **0.139** | **3.58s** | 1.09x |
| `large-v3-turbo` | 5.1s | **0.593** | 0.376 | 14.88s | 4.52x |

Six demo clips, identical prompt and decode settings, CPU int8.

Turbo genuinely transcribes better raw — 0.593 vs 0.699 — and still loses end-to-end:

1. **It is 4.2x slower.** RTF 4.52 means it cannot keep up with real time. The
   glass-to-glass budget was already blown at 1.09.
2. **`ASR_CORRECTION_RULES` are coupled to `small`.** They were hand-written against
   the errors *that model* makes (`viatum piya`, `hiwe ken`, `pahalike`, `ikondio`).
   Turbo makes different mistakes, the rules do not fire, and it never receives the
   ~5x normalizer improvement that carries `small` from 0.699 to 0.139.

**The coupling is the finding worth keeping.** The correction layer is tuned to one
model's failure modes, so `WHISPER_MODEL_SIZE` cannot be changed in isolation. On a
GPU — where turbo's latency disadvantage disappears — its better raw WER would likely
win, but the rules must be re-derived against its error patterns first, and
`scripts/ab_prompt_bias.py` re-run.

The ~1.6GB turbo weights are now cached on this machine, so the swap can be re-tested
cheaply if a GPU becomes available.
