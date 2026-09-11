Work Plan — Swahili & Sheng S2S Agent (Named Roster)

Timeline: 1 working day, independent, merge tomorrow.
Team:

    🎙️ Person A = Ray — Data & Ears (ASR + dataset)

    🧠 Person B = Njugush — Sheng Brain (LLM + lexicon)

    🔊 Person C = Alphonse — Mouth & Glue (Voice + integration + demo)

Read this whole document once before you start. Then go to your own section and don't read anyone else's.
0. Rules of engagement (read once, apply all day)

You are working independently, but on a shared repository. These rules exist so that when we merge tomorrow, nothing breaks.
0.1 — You may only edit files you own
File	Owner	Who may read
data_engine/* (all files)	Ray	Njugush, Alphonse
asr_engine.py	Ray	Njugush, Alphonse
training/train_whisper_lora.py	Ray	—
training/test_lora_inference.py	Ray	—
sheng_lexicon.py	Njugush	Ray, Alphonse
llm_engine.py	Njugush	Alphonse
naturalizer.py (new file)	Njugush	Alphonse
training/train_sheng_llm.py	Njugush	—
training/test_llm_inference.py	Njugush	—
data_engine/synthetic_dialogue_generator.py	Njugush	Ray
tts_engine.py	Alphonse	—
pipeline.py	Alphonse	Ray, Njugush (read only)
app.py, cli.py	Alphonse	—
config.py	Alphonse	Ray, Njugush
README.md, requirements.txt	Alphonse	Ray, Njugush
dataset/*	Ray	Njugush

If you need a change in a file you don't own: message the owner. Do not edit it yourself. Even a one-line "quick fix" will cause a merge conflict tomorrow, and you won't have time to debug it.
0.2 — Three things are frozen (nobody changes these today)

Freeze 1 — The two pipeline interfaces:
python

# Ray → Alphonse
ShengASREngine.transcribe(audio_path) -> (raw: str, normalized: str, meta: dict)

# Njugush → Alphonse
ShengLLMEngine.generate_response(text) -> (reply: str, meta: dict)

You may add new keys to meta. You may not change the number or order of return values.

Freeze 2 — Names Ray and Alphonse import from sheng_lexicon.py:
python

WHISPER_SHENG_PROMPT       # Ray imports this in asr_engine.py
ShengNormalizer            # Ray imports this in asr_engine.py
ShengAcousticPhonetics     # Alphonse imports this in tts_engine.py
SHENG_DICTIONARY           # Ray, Alphonse read this

Njugush may add to these (new keys, new rules). Njugush may not rename or delete them.

Freeze 3 — Names Ray and Njugush import from config.py:
python

WHISPER_MODEL_SIZE, ASR_LANGUAGE, ASR_DEVICE, ASR_COMPUTE_TYPE   # Ray imports
LLM_BACKEND, OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL        # Njugush imports
OLLAMA_BASE_URL, OLLAMA_MODEL, GEMINI_API_KEY
DEFAULT_VOICE, VOICE_OPTIONS
AUDIO_TEMP_DIR, BASE_DIR

Alphonse may add keys. Alphonse may not rename or delete existing keys.
0.3 — The meta dict keys we agreed on today

Everyone reads these; nobody invents new ones without telling the group.

From ShengASREngine.transcribe (Ray):
python

{
    "detected_language": str,
    "language_probability": float,
    "duration_seconds": float,
    "inference_time_ms": float,
    "error": str,          # only present on failure
}

From ShengLLMEngine.generate_response (Njugush):
python

{
    "backend": str,
    "llm_time_ms": float,
    "response_length": int,
    "emotion": str,        # NEW — one of: "excited", "neutral", "thinking"
}

Alphonse's tts_engine.py will read meta["emotion"] to pick prosody. If it's missing, Alphonse defaults to "neutral".
0.4 — Git

Each person works on their own branch and pushes hourly:
text

main                    ← frozen, Alphonse merges into this tomorrow
├── feat/asr-data       ← Ray
├── feat/sheng-brain    ← Njugush
└── feat/voice-demo     ← Alphonse

    Commit small, commit often. A commit per task, minimum.

    Push before you take a break. If your laptop dies, we don't lose your work.

    Do not merge your branch into main today. Alphonse merges tomorrow morning.

1. 🎙️ RAY — Data & Ears (ASR + dataset)
Mission

Turn raw Kenyan Sheng audio into a working, accurate transcriber. You own the "audio → text" half of the pipeline.
Files you own

    data_engine/audio_preprocessor.py

    data_engine/pseudo_labeler.py

    data_engine/clean_ground_truth.py

    data_engine/process_raw_dataset.py

    data_engine/denoise_voice_sample.py

    asr_engine.py

    training/train_whisper_lora.py

    training/test_lora_inference.py

    dataset/* (everything in this folder)

Files you must not touch

Everything else. Especially sheng_lexicon.py (Njugush owns it), pipeline.py (Alphonse owns it), and config.py (Alphonse owns it).
Deliverable by end of day

    A working ASR engine — you can hand it an audio file and get back accurate Swahili/Sheng text.

    A cleaned dataset — dataset/train_sheng_asr_cleaned.jsonl with at least 30 minutes of hand-verified Sheng audio + transcript (~150–300 clips).

Why this matters

Whisper by default does not understand Sheng. Your WHISPER_SHENG_PROMPT biasing + hand-cleaned data + (optionally) a fine-tune is the only way to make the pipeline hear "form ni gani" instead of "formula Ni Gani."

The single biggest failure mode in the current pipeline: audio_preprocessor.py claims to use Silero VAD, but the code uses pydub.silence.split_on_silence. pydub cuts on raw energy, which means it breaks words in half and merges different speakers. Every chunk downstream inherits that error. Fix this first.
Ray's step-by-step tasks
Step 1 — Verify your environment (15 min)
bash

cd /path/to/Sheng_pipe
python -c "
from faster_whisper import WhisperModel
from pydub import AudioSegment
import soundfile, librosa, numpy
print('ok')
"
ffmpeg -version

Why: the whole pipeline shells out to ffmpeg for audio decoding. If it's missing, nothing downstream works, and you'll waste an hour thinking it's a Python bug.
Step 2 — Swap in real Silero VAD (1 hour)

Install:
bash

pip install silero-vad

Open data_engine/audio_preprocessor.py. The method chunk_audio_by_vad currently does:
python

chunks = silence.split_on_silence(audio, min_silence_len=..., silence_thresh=..., keep_silence=200)

Replace it with Silero:
python

import torch
from silero_vad import load_silero_vad, get_speech_timestamps, read_audio

def chunk_audio_by_vad(self, audio_path, output_dir):
    out_dir = Path(output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    wav = read_audio(audio_path, sampling_rate=self.target_sr)
    model = load_silero_vad()
    timestamps = get_speech_timestamps(
        wav, model,
        sampling_rate=self.target_sr,
        min_speech_duration_ms=1500,
        max_speech_duration_s=self.max_chunk_sec,
        min_silence_duration_ms=400,
        return_seconds=True,
    )
    processed = []
    for idx, ts in enumerate(timestamps):
        start, end = int(ts["start"] * self.target_sr), int(ts["end"] * self.target_sr)
        segment = wav[start:end].numpy()
        dur = (end - start) / self.target_sr
        if dur < self.min_chunk_sec or dur > self.max_chunk_sec:
            continue
        path = out_dir / f"chunk_{idx:05d}.wav"
        sf.write(str(path), segment, self.target_sr)
        processed.append({"chunk_id": idx, "file_path": str(path), "duration_sec": round(dur, 2)})
    return processed

Why: Silero is a neural VAD. It detects speech, not loudness. It won't cut mid-word when the speaker pauses for breath. Result: your ASR transcriptions get 10–20% more accurate on the same audio.
Step 3 — Re-segment the raw dataset (30 min)
bash

python data_engine/process_raw_dataset.py

Watch the log — should produce ~200+ chunks in dataset/segmented/.

Sanity check: pick 5 random chunks and listen to them. Each should be one complete thought — not a word fragment, not two people merged.
Step 4 — Pseudo-label with whisper-large-v3 (1 hour)

This is critical. Your pseudo_labeler.py currently loads model_size="small". Change it to large-v3:
python

self.asr = ShengASREngine(model_size="large-v3")

Or bypass your ASR class for this step:
python

from faster_whisper import WhisperModel
from sheng_lexicon import WHISPER_SHENG_PROMPT
model = WhisperModel("large-v3", device="cuda", compute_type="float16")
segments, info = model.transcribe(
    path, language="sw",
    initial_prompt=WHISPER_SHENG_PROMPT,
    beam_size=5, vad_filter=True,
)
text = " ".join(s.text.strip() for s in segments)

Why: large-v3 is 40× larger than tiny and dramatically better on code-switched speech. The initial_prompt biases its language model toward Sheng vocabulary.

Run:
bash

python data_engine/process_raw_dataset.py

This writes dataset/train_sheng_asr.jsonl and .csv.

Sanity check: open the CSV and read the first 20 rows. If garbage:

    Confirm WHISPER_SHENG_PROMPT is being passed.

    Try language=None (auto-detect).

Step 5 — Hand-clean 150–200 clips (3–4 hours) ⚠️ BIGGEST BLOCK

This is the part you cannot skip. Whisper's pseudo-labels are 70–85% correct at best.

Work in dataset/train_sheng_asr.csv. Add a column verified_text. Listen to each audio, read the pseudo-label, fix it. You are not transcribing from scratch — you are correcting.

Tips:

    Use ffplay -nodisp -autoexit file.wav for fast playback.

    Batch: 20 at a time, then a break.

    When unsure of a Sheng word: write it phonetically. Njugush's lexicon rules will normalize later.

    Delete clips that are silent, pure noise, cut mid-word, or two speakers.

Then run:
bash

python data_engine/clean_ground_truth.py

Produces dataset/train_sheng_asr_cleaned.jsonl.

Bug alert: in clean_ground_truth.py this rule:
python

(re.compile(r"\bniaje,\s+keze\b", re.IGNORECASE), "nielekeze")

fires on any sentence with "niaje, keze" — mapping to "nielekeze" (direct me). That's wrong. Delete it or narrow it. Since it's your file, fix it.
Step 6 — Test the ASR (30 min) ✅ PROOF OF DONE

Pick 6 audio clips from your cleaned set:
bash

python -c "
from asr_engine import ShengASREngine
asr = ShengASREngine()
for f in ['dataset/segmented/sample1/chunk_00000.wav', ...]:
    raw, norm, meta = asr.transcribe(f)
    print(f'--- {f}')
    print(f'   raw : {raw}')
    print(f'   norm: {norm}')
    print(f'   ms  : {meta[\"inference_time_ms\"]}')
"

Success criteria:

    ≥5 of 6 clips produce text a Swahili speaker would consider roughly correct.

    Latency under 3 seconds per clip.

Write down for each: raw_output | cleaned_output | what_a_human_would_say. Paste into A_RESULTS.md.
Step 7 — OPTIONAL: LoRA fine-tune Whisper (only if GPU available)

If steps 1–6 are done and you have 2 spare hours:
bash

python training/train_whisper_lora.py \
  --model_name openai/whisper-base \
  --train_jsonl dataset/train_sheng_asr_cleaned.jsonl \
  --epochs 3 --batch_size 4

Do not use whisper-tiny — too small for Sheng.

Test:
bash

python training/test_lora_inference.py path/to/test.wav

Compare to large-v3 zero-shot. If LoRA isn't clearly better, ship the large-v3 path.
Ray's Proof of Done (show tomorrow)

    dataset/train_sheng_asr_cleaned.jsonl with ≥150 records, each with verified_text.

    dataset/train_sheng_asr_cleaned.csv — human-readable.

    A_RESULTS.md with:

        The 6 test clips and their quality (raw vs. verified).

        The updated chunk_audio_by_vad code.

        Whether you shipped LoRA or zero-shot, and why.

    asr_engine.py runs and returns non-empty normalized_text for any WAV.

If Ray gets stuck

    Whisper outputs garbage: confirm initial_prompt is passed. Try language=None. Try a different model size.

    VAD produces tiny fragments: raise min_speech_duration_ms to 2000.

    Hand-cleaning takes too long: stop at 100 clips. Quality > quantity.

2. 🧠 NJUGUSH — Sheng Brain (LLM + lexicon)
Mission

Make the bot talk like a Nairobi msee on the phone — natural, varied, multi-turn. You own the "text → text" half of the pipeline.
Files you own

    sheng_lexicon.py (append-only, see Rule 0.2)

    llm_engine.py

    naturalizer.py (new file, you create it)

    training/train_sheng_llm.py

    training/test_llm_inference.py

    data_engine/synthetic_dialogue_generator.py

Files you must not touch

Everything else. Especially asr_engine.py (Ray), tts_engine.py and pipeline.py (Alphonse).
Deliverable by end of day

    A bot that doesn't repeat itself across a 5-turn conversation, uses 15+ distinct Sheng tokens naturally.

    Either a working API-backed ShengLLMEngine or a retrained LoRA on a bigger base model.

    naturalizer.py wired into generate_response.

Why this matters

Right now your LoRA is trained on 25 unique Q&A pairs. When it sees a prompt it doesn't recognize, it snaps back to the nearest memorized answer. That's the "canned" feeling.

    Fast (30 min): bypass LoRA — route generate_response to Groq/OpenAI with the Sheng system prompt.

    Slow (2–4h + GPU): retrain the LoRA on 2,000+ unique turns with a bigger base.

Ship the API path first (safety net), then attempt LoRA if time allows.
Njugush's step-by-step tasks
Step 1 — Fix _generate_lora generation params (30 min)

Open llm_engine.py. Three problems:

Problem 1 — wrong dtype:
python

torch_dtype=torch.float32,     # ← 3× memory, 2× slower

Change to:
python

torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,

Problem 2 — max_new_tokens=60 is too short. Change to 120.

Problem 3 — no repetition penalty. Change the generate call to:
python

output_ids = self._lora_model.generate(
    **inputs,
    max_new_tokens=120,
    temperature=0.8,
    top_p=0.92,
    do_sample=True,
    repetition_penalty=1.15,
    no_repeat_ngram_size=3,
    pad_token_id=self._lora_tokenizer.eos_token_id,
)

Why: repetition penalty is the single biggest fix for a small model that loops.
Step 2 — Wire the API path (30 min)

llm_engine.py already has _generate_openai. Make it the default today.

Message Alphonse to change config.py:
python

LLM_BACKEND = os.getenv("LLM_BACKEND", "openai")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.groq.com/openai/v1")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "llama-3.3-70b-versatile")

Then test:
bash

export OPENAI_API_KEY="gsk_...your-groq-key..."
export LLM_BACKEND=openai
python -c "
from llm_engine import ShengLLMEngine
eng = ShengLLMEngine(backend='openai')
reply, meta = eng.generate_response('Niaje chief, form ni gani leo mtaani?')
print(reply); print(meta)
"

Why: this is your fallback. If LoRA retrain doesn't finish, you still ship.
Step 3 — Rewrite the system prompt (30 min)

Open sheng_lexicon.py. Replace SHENG_SYSTEM_PROMPT:
text

Wewe ni msee mjanja wa Nairobi anayeongea Sheng safi, ya kisasa na Kiswahili ya mtaani.
Jina lako ni 'Nairobi Bot'.

Sheria za Maongezi:
1. Changanya Kiswahili, Sheng na Kiingereza kama msee wa mtaa — USIFASIRI maneno ya Kiingereza kwa Kiswahili sanifu. (Sema "form" si "mpango", "hustle" si "kazi ngumu".)
2. Tumia maneno kama: chief, rada, form, fiti, msee, dooh, mbogi, wazi, maze, morio, nisho, utapata, alafu, unisho, kumeet, nimeget, tunang'ang'ana, kuogopa, luku, mtaa, keja.
3. Majibu yawe MAFUPI: sentensi 1–2 kama maongezi ya kawaida ya sauti. USITOE hotuba.
4. ENDESHA MAONGEZI: maliza 40% ya majibu na swali (mfano "Rada yako?", "Wewe unasema aje?", "Uko aje upande wako?").
5. Kuwa na uchangamfu. Usiongee Kiswahili cha vitabu.

Why: the current prompt is too soft. Small models default to "safe" language — you have to explicitly say don't.
Step 4 — Write naturalizer.py (1 hour)

Create the file at project root:
python

# naturalizer.py
"""Injects fillers, pauses, follow-ups, and prosody hints into LLM output."""
import random

FILLERS = {
    "neutral":  ["Eeeh", "Aah", "Sasa", "Hmm", "Kwani", "Alafu"],
    "excited":  ["Wueeh", "Aki", "Eish", "Bana"],
    "thinking": ["Kwani", "Maze", "Alafu", "Hmm"],
}

FOLLOWUPS = [
    "Rada yako?", "Wewe unasema aje?", "Kwani wewe?",
    "Uko aje upande wako?", "Niambie zaidi.",
]

FILLER_PROB    = 0.60
PAUSE_PROB     = 0.35
FOLLOWUP_PROB  = 0.40
_ALL_LOWER = {f.lower() for fl in FILLERS.values() for f in fl}


def detect_emotion(text: str) -> str:
    t = text.lower()
    if any(w in t for w in ["party", "sherehe", "weekend", "turn up", "raha", "fiti sana", "moto"]):
        return "excited"
    if any(w in t for w in ["choka", "stress", "uchovu", "pole", "shida", "noma"]):
        return "thinking"
    return "neutral"


def humanize(text: str, emotion: str = None, add_followup: bool = True) -> str:
    if not text:
        return text
    text = text.strip()
    emotion = emotion or detect_emotion(text)

    if random.random() < FILLER_PROB:
        first = text.split()[0].rstrip(",.").lower() if text.split() else ""
        if first not in _ALL_LOWER:
            filler = random.choice(FILLERS[emotion])
            body = text[0].lower() + text[1:]
            text = f"{filler}, {body}"

    words = text.split()
    if len(words) > 8 and random.random() < PAUSE_PROB:
        i = random.randint(3, len(words) - 3)
        words.insert(i, "...")
        text = " ".join(words)

    if add_followup and random.random() < FOLLOWUP_PROB:
        if not text.rstrip().endswith("?"):
            text = text.rstrip(".!") + ". " + random.choice(FOLLOWUPS)

    return text


def vary_prosody(emotion: str = "neutral"):
    table = {
        "excited":  (random.choice(["+5%", "+8%", "+10%"]), random.choice(["+3Hz", "+5Hz", "+6Hz"])),
        "neutral":  (random.choice(["-3%", "+0%", "+2%"]),  random.choice(["-2Hz", "+0Hz", "+2Hz"])),
        "thinking": (random.choice(["-8%", "-5%", "-3%"]),  random.choice(["-4Hz", "-2Hz", "+0Hz"])),
    }
    return table.get(emotion, table["neutral"])

Wire it into llm_engine.py. Before the return in generate_response:
python

from naturalizer import humanize, detect_emotion

emotion = detect_emotion(user_message)
reply = humanize(reply, emotion=emotion)

self.conversation_history.append({"role": "user", "content": user_message})
self.conversation_history.append({"role": "assistant", "content": reply})

return reply, {
    "backend": backend_used,
    "llm_time_ms": round(duration * 1000, 2),
    "response_length": len(reply),
    "emotion": emotion,
}

Why: Even with a mediocre LLM, humanized text sounds human. And emotion in meta is what lets Alphonse's voice vary per turn.
Step 5 — Expand synthetic_dialogue_generator.py to 2,000+ turns (2 hours)

Current: 25 Q&A × 5 themes = 125, repeated to hit num_samples=150. This is the reason the LoRA memorizes.

Rewrite the generator with slot-based composition:
python

SLOTS = {
    "greeting":  ["Niaje", "Sasa", "Mambo", "Vipi", "Uko aje", "Rada iko aje"],
    "address":   ["chief", "msee", "morio", "bazenga", "maze", "chali", "mresh"],
    "topic":     ["form ya leo", "rada ya mtaa", "hustle", "mbogi", "keja", "weekend"],
    "action":    ["kuchill", "kupiga luku", "kusaka dooh", "kutea", "kumeet"],
    "context":   ["mtaani", "kejani", "tao", "kibandani", "stage"],
}

# Per theme: write 10 bot skeletons and 10 user skeletons (with {slots})
BOT_SKELETONS = {
    "Hustle, Money & Work": [
        "Kazi inasonga {fiti_word} {address}, tunang'ang'ana kusaka dooh bila kuogopa!",
        "Eeeh {address}, hustle inaendelea tu. {action} hadi dooh ivala.",
        # ... 8 more
    ],
    # ... 4 more themes
}

def generate_sft_dataset(num_samples=2500):
    seen = set()
    with open(OUTPUT_SFT_JSONL, "w") as f:
        for theme_name, _ in ALL_THEMES.items():
            for user_skel in USER_SKELETONS[theme_name]:
                for bot_skel in BOT_SKELETONS[theme_name]:
                    for _ in range(2):
                        user = user_skel.format(**{k: random.choice(v) for k, v in SLOTS.items()})
                        bot  = bot_skel.format(**{k: random.choice(v) for k, v in SLOTS.items()},
                                                fiti_word=random.choice(["fiti", "safi", "poa"]))
                        key = (user, bot)
                        if key in seen: continue
                        seen.add(key)
                        # ... write record

Why: 2,500 unique turns forces generalization instead of memorization.

Regenerate:
bash

python data_engine/synthetic_dialogue_generator.py
wc -l dataset/train_sheng_sft.jsonl    # ~2500

Step 6 — Retrain LoRA on Qwen2.5-1.5B-Instruct (2–4h if GPU)

Skip if no GPU. The API path already works.
bash

python training/train_sheng_llm.py \
  --model_name Qwen/Qwen2.5-1.5B-Instruct \
  --train_jsonl dataset/train_sheng_sft.jsonl \
  --output_dir llm_sheng_lora_output_1_5B \
  --epochs 3 --batch_size 2

Why 1.5B: same tokenizer as your 0.5B (no mismatch), 3× more capable.
Step 7 — Test the full loop (30 min) ✅ PROOF OF DONE
bash

python -c "
from llm_engine import ShengLLMEngine
eng = ShengLLMEngine()
prompts = [
    'Niaje chief, form ni gani leo mtaani?',
    'Hustle inaendeleaje leo chief?',
    'Niko na chwani nataka kubuy lunch, unapendekeza nini?',
    'Nisho pahali keja yako iko ndio nipanda nganya tao.',
    'Hii weekend form iko wapi?',
    'Hizo viatu mpya zinakutoa aje?',
]
seen = set()
for p in prompts:
    reply, meta = eng.generate_response(p)
    print(f'USER: {p}'); print(f'BOT : {reply}'); print(f'META: {meta}'); print()
    seen.add(reply)
print(f'Unique responses: {len(seen)} / {len(prompts)}')
"

Success criteria:

    Unique responses: 6 / 6

    ≥15 different Sheng words across replies

    Each reply is 1–3 sentences

    ≥2 end in a question

    meta["emotion"] is populated

Run 3 times. Write results to B_RESULTS.md.
Njugush's Proof of Done (show tomorrow)

    naturalizer.py exists and is wired into llm_engine.py.

    B_RESULTS.md with 6-prompt output and "unique count."

    Either API path is live OR llm_sheng_lora_output_1_5B/final_adapter exists.

    dataset/train_sheng_sft.jsonl has ≥2,000 lines.

If Njugush gets stuck

    API returns formal Swahili: print the messages array before the request — the system prompt isn't reaching the API.

    LoRA still repeats: check wc -l dataset/train_sheng_sft.jsonl. If it's still 150, you didn't expand.

    Model takes forever: move to API. Do not burn the day on GPU.

3. 🔊 ALPHONSE — Mouth & Glue (Voice + integration + demo)
Mission

Make the bot sound like a person on the phone, make the whole pipeline run end-to-end, own the demo. You own "text → audio" and the final integration.
Files you own

    tts_engine.py

    pipeline.py

    app.py, cli.py

    config.py

    README.md, requirements.txt

    assets/demo_samples/ folder and everything in it.

Files you must not touch

asr_engine.py (Ray), llm_engine.py and sheng_lexicon.py (Njugush), data_engine/* (Ray), training/*.
Deliverable by end of day

    A voice that sounds human.

    A Gradio app that runs end-to-end.

    A demo script and README.

Why this matters

Your current tts_engine.py calls Edge-TTS with zero post-processing and fixed prosody. Neural voices at fixed rate/pitch sound like a GPS. Three fixes:

    Post-process audio (compressor + small reverb) — "on the phone" not "in a vacuum."

    Vary rate/pitch per call — kills monotone.

    React to meta["emotion"] from Njugush — excited turns faster, thinking slower.

No GPU needed. Fits on 8 GB RAM.
Alphonse's step-by-step tasks
Step 1 — Install toolchain (20 min)
bash

pip install pedalboard pydub
python -c "from pedalboard import Pedalboard, Reverb, Compressor; print('ok')"
ffmpeg -version

If ffmpeg missing: sudo apt install ffmpeg or brew install ffmpeg.
Step 2 — Add post-processing to tts_engine.py (1 hour)

Add this method to ShengTTSEngine:
python

def _polish(self, path: str):
    """Compress + tiny reverb so the voice doesn't sound like a vacuum recording."""
    try:
        from pedalboard import Pedalboard, Reverb, Compressor, HighpassFilter, LowShelfFilter
        from pedalboard.io import AudioFile
    except ImportError:
        return
    board = Pedalboard([
        HighpassFilter(cutoff_frequency_hz=80),
        LowShelfFilter(cutoff_frequency_hz=250, gain_db=2),
        Compressor(threshold_db=-18, ratio=3.0),
        Reverb(room_size=0.15, wet_level=0.05),
    ])
    with AudioFile(path) as f:
        audio = f.read(f.frames); sr = f.samplerate
    polished = board(audio, sr)
    with AudioFile(path, "w", sr, polished.shape[0]) as f:
        f.write(polished)

In synthesize_async, right after await communicate.save(...):
python

if str(output_path).endswith(".mp3"):
    from pydub import AudioSegment
    wav_path = str(output_path).replace(".mp3", ".wav")
    AudioSegment.from_mp3(str(output_path)).export(wav_path, format="wav")
    self._polish(wav_path)
    AudioSegment.from_wav(wav_path).export(str(output_path), format="mp3")
    import os; os.remove(wav_path)

Why: compressor evens loudness, high-pass kills rumble, tiny reverb adds "room." Real humans are never recorded in silence.
Step 3 — Per-turn prosody variation (30 min)

In pipeline.py, before calling self.tts.synthesize:
python

try:
    from naturalizer import vary_prosody
except ImportError:
    def vary_prosody(emotion="neutral"):
        import random
        return random.choice(["-3%", "+0%", "+2%"]), random.choice(["-2Hz", "+0Hz", "+2Hz"])

if rate == "+0%" and pitch == "+0Hz":
    emotion = llm_meta.get("emotion", "neutral")
    rate, pitch = vary_prosody(emotion)

Why the try/except: naturalizer.py is Njugush's file. If it's not pushed yet, this keeps you unblocked.
Step 4 — Test TTS on 3 Sheng phrases (30 min)
bash

python -c "
from tts_engine import ShengTTSEngine
tts = ShengTTSEngine()
for phrase in [
    'Eeeh chief, kazi inasonga fiti sana!',
    'Hiyo chwani utapata chapo mbili safi na madondo kwa kibanda ya mtaa.',
    'Panda tu nganya pale stage, shuka tao alafu unisho kwa simu.',
]:
    path, meta = tts.synthesize(phrase, output_path=f'/tmp/test_{abs(hash(phrase))%999}.mp3')
    print(f'{path}  ({meta[\"tts_time_ms\"]}ms)')
"

Listen to all three. Honest question: person on a phone, or GPS?

If still robotic after polish, go to Step 5.
Step 5 — OPTIONAL: XTTS v2 voice clone (2h)

Biggest voice-quality jump. Only if 8 GB+ RAM and 2h spare.
bash

pip install TTS

Then:
python

from TTS.api import TTS
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cuda" if torch.cuda.is_available() else "cpu")
tts.tts_to_file(
    text="Niaje chief, form ni gani leo mtaani?",
    speaker_wav="assets/kenyan_genz_clean_ref.wav",
    language="sw",
    file_path="output.wav",
)

The reference assets/kenyan_genz_clean_ref.wav is produced by Ray's denoise_voice_sample.py. Check it exists. If not, message Ray to produce it early.

Fallback if XTTS fails: skip it, keep Edge-TTS + polish. You lose ~20% voice quality, not the demo.
Step 6 — Verify pipeline end-to-end (30 min)
bash

python -c "
from pipeline import SpeechToSpeechPipeline
p = SpeechToSpeechPipeline()
r = p.process_audio('assets/demo_samples/demo_1_greeting.mp3')
print('success:', r['success'])
print('user raw:', r['user_raw_transcription'])
print('user norm:', r['user_normalized_sheng'])
print('bot reply:', r['bot_response_text'])
print('latency:', r['latencies'])
"

Success: success: True, all fields non-empty, total_glass_to_glass_ms < 3000 (up to 5000 acceptable).

If > 5000ms, profile:

    ASR slow → message Ray.

    LLM slow → message Njugush.

    TTS slow → check you're not calling _polish on huge files and not loading model per call.

Step 7 — Run the Gradio app (20 min)
bash

python app.py

Open http://localhost:7860. Record a phrase. Verify:

    Audio auto-plays.

    Latency display shows.

    Chatbot shows both roles.

    Reset works.

If the app fails: check imports. If Njugush renamed something in sheng_lexicon.py, app.py crashes at startup. Do not fix their file — message them.
Step 8 — Write README.md and DEMO_SCRIPT.md (1 hour)

README.md:

    1-paragraph what-it-is

    Quick-start

    6 demo scenarios

    Known limitations (be honest)

DEMO_SCRIPT.md:
text

SCENARIO 1 — Greeting
  Mic: "Niaje chief, form ni gani leo mtaani?"
  Expected bot vibe: relaxed, casual, follow-up question
  Expected latency: < 3s

SCENARIO 2 — Hustle
  Mic: "Hustle inaendeleaje leo chief?"
  Expected bot vibe: energetic, mentions dooh/kazi
  ...

Step 9 — Generate fallback audio NOW (30 min) 🚨 CRITICAL

Before you sleep, generate a canned audio response for each of the 6 demo prompts and save them to /tmp/fallback/. If the live pipeline breaks on stage, you play these instead.
bash

mkdir -p /tmp/fallback
python -c "
from tts_engine import ShengTTSEngine
tts = ShengTTSEngine()
fallbacks = {
    '1_greeting': 'Eeeh chief, form ni kuchill tu kejani na mbogi, tukipiga stori za luku na hustle. Rada yako?',
    '2_hustle': 'Kazi inasonga fiti sana chief, tunang\'ang\'ana kusaka dooh bila kuogopa!',
    '3_kibanda': 'Hiyo chwani utapata chapo mbili safi na madondo kwa kibanda ya mtaa. Utashiba fiti sana!',
    '4_nganya': 'Panda tu nganya pale stage, shuka tao alafu unisho kwa simu nikupe rada safi ya street!',
    '5_weekend': 'Weekend ni kupika luku, kutokea pale alchemist na kuparty like there is no tomorrow!',
    '6_drip': 'Hizo raba zimepiga luku hatari msee, unakaa chief mwenyewe!',
}
for name, text in fallbacks.items():
    path, _ = tts.synthesize(text, output_path=f'/tmp/fallback/{name}.mp3')
    print(path)
"

Why: this turns "the demo failed" into "the demo had a pre-recorded backup." Do it before bed, no exceptions.
Step 10 — Rehearse 3 times (30 min) ✅ PROOF OF DONE

Run all 6 scenarios three times. Watch for:

    Same reply twice in a row (Njugush's variety fix isn't live).

    Latency spikes (model reloading per call).

    Audio cutting off (pydub/pedalboard dropping bytes).

Fix anything you can. If it's in Ray's or Njugush's file, message them.
Alphonse's Proof of Done (show tomorrow)

    Three audio files from Step 4 — send them in the group chat tonight.

    pipeline.py output from Step 6 with latency < 5s.

    Gradio app running on your laptop when we meet.

    DEMO_SCRIPT.md and README.md finished.

    /tmp/fallback/ has 6 audio files — verified playable.

If Alphonse gets stuck

    XTTS OOMs: fall back to Edge-TTS + polish.

    Gradio won't load: python -c "import app" — read the traceback.

    Latency > 10s: STOP and message the group. This is a blocker.

4. Tomorrow morning — merge protocol

Together, in one room, this exact order:

    9:00 — Everyone pushes their branch. No exceptions.

    9:10 — Alphonse merges into main:

        Ray's branch first (feat/asr-data), run python test_pipeline.py. If it fails, don't merge the next.

        Njugush's branch next (feat/sheng-brain), run tests again.

        Alphonse's own last.

    9:30 — Alphonse runs python app.py on main. All three watch. Record one test phrase.

    9:45 — Walk through the 6 scenarios slowly. Note rough edges.

    10:00 — Fix the ONE worst thing. Not five.

    10:30 — Rehearse full demo 3 times.

    11:30 — Freeze. No more code changes.

5. Fallback plan (if something breaks tomorrow)

Priority order:

    ASR broken → swap ShengASREngine for whisper-large-v3 zero-shot inline (Alphonse writes 10 lines).

    LLM broken → LLM_BACKEND=openai using Njugush's API path.

    TTS broken → play /tmp/fallback/ files. This is why Step 9 exists.

    Gradio broken → python cli.py --demo. Ugly but works.

The demo cannot fail if the fallback audio is generated tonight. Do it before you sleep.
6. End-of-day checklist — paste into group chat tonight

🎙️ Ray:

    □

    dataset/train_sheng_asr_cleaned.jsonl ≥150 records with verified_text
    □

    asr_engine.py transcribes 5/6 test clips correctly
    □

    A_RESULTS.md written
    □

    Branch pushed

🧠 Njugush:

    □

    naturalizer.py written and imported by llm_engine.py
    □

    6-prompt test shows "6 / 6 unique responses"
    □

    meta["emotion"] populated
    □

    dataset/train_sheng_sft.jsonl ≥2,000 lines
    □

    Either API path or 1.5B LoRA is live
    □

    B_RESULTS.md written
    □

    Branch pushed

🔊 Alphonse:

    □

    tts_engine.py runs _polish on every output
    □

    3 TTS test files produced (send in chat)
    □

    pipeline.py end-to-end test passes < 5s
    □

    Gradio app loads and records/plays
    □

    DEMO_SCRIPT.md and README.md written
    □

    /tmp/fallback/ has 6 files 🚨
    □

    Branch pushed

If any box isn't ticked at midnight, tell the group. Do not stay up debugging — a working 80% is worth more than a broken 100%.
