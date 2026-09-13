"""
generate_sheng_data.py - Synthetic Sheng dialogue bootstrap for the Sheng-TTS-model repo.

Generates multi-turn Sheng conversations by calling any OpenAI-compatible API
(GPT-4o, Groq, local Ollama, etc.) with the repo's SHENG_SYSTEM_PROMPT, then
deduplicates and filters them. Output is JSONL ready for human validation,
LoRA fine-tuning (chat format), or intent classification.

Setup:
  pip install openai requests
  export OPENAI_API_KEY=...            # or OPENAI_BASE_URL for Groq/Ollama

Usage:
  # 1. Generate conversations (each run appends, skips existing files)
  python generate_sheng_data.py --mode conversations --target 2000 --batch 10

  # 2. Generate intent-tagged utterances for the heuristic classifier
  python generate_sheng_data.py --mode intents --per_intent 60

  # 3. Filter/dedupe everything into *_filtered.jsonl
  python generate_sheng_data.py --mode filter
"""
import argparse
import json
import os
import random
import re
import time
from pathlib import Path

OUT_DIR = Path("sheng_synthetic_data")
OUT_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Import the repo's own persona prompt & lexicon so generation stays on-voice
# ---------------------------------------------------------------------------
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    from sheng_lexicon import SHENG_SYSTEM_PROMPT, SHENG_DICTIONARY, FEW_SHOT_CONVERSATIONS
except ImportError:
    sys.exit("Run this from the repo root so sheng_lexicon.py is importable.")

# Diversity axes: rotate through these so batches don't collapse into one theme
SCENARIOS = [
    "morning greetings on the way to work", "buying lunch at a kibanda",
    "matatu ride to tao", "talking about weekend plans", "asking for directions",
    "talking about money and hustle", "discussing football (local teams)",
    "complimenting someone's outfit/drip", "talking about the weather in Nairobi",
    "planning a meet-up with mbogi", "small talk with a boda rider",
    "talking about work stress", "street food at night", "phone call catch-up",
    "talking about a new nganya in town", "rent and Nairobi living costs",
    "joking about traffic jam", "asking about someone's family back home",
    "talking about music (Gengetone/Gen Z artists)", "making plans for raving",
]
TONES = ["playful", "tired but friendly", "hyped and energetic", "casual and calm",
         "sympathetic", "teasing between close friends"]
ROLES = ["two longtime friends", "colleagues at a Nairobi office",
         "cousins meeting after a while", "neighbors in the same mtaa",
         "a customer and a shop attendant"]

# Intent schema aligned with sheng_lexicon.HEURISTIC_INTENTS (extend as needed)
INTENTS = ["greeting", "weekend_plans", "work_hustle", "money", "food_lunch",
           "transport", "fashion_drip", "compliment", "farewell", "thanks",
           "smalltalk_rada", "football", "weather", "identity_who_are_you",
           "fatigue_stress", "traffic_jam"]

LEXICON_WORDS = set(SHENG_DICTIONARY.keys())


def get_client():
    from openai import OpenAI
    return OpenAI(
        api_key=os.getenv("OPENAI_API_KEY", "ollama"),
        base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
    )


def chat(client, messages, temperature=0.9, max_tokens=4096):
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    resp = client.chat.completions.create(model=model, messages=messages,
                                          temperature=temperature, max_tokens=max_tokens)
    return resp.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
def gen_conversations(client, target, batch_size):
    path = OUT_DIR / "conversations_raw.jsonl"
    existing = path.read_text().count("\n") if path.exists() else 0
    todo = target - existing
    print(f"conversations: {existing} exist, generating {todo} more")
    with path.open("a", encoding="utf-8") as f:
        for i in range(0, todo, batch_size):
            scenario = random.choice(SCENARIOS)
            tone = random.choice(TONES)
            roles = random.choice(ROLES)
            fewshot = "\n".join(f"User: {t['user']}\nBot: {t['bot']}" for t in random.sample(FEW_SHOT_CONVERSATIONS, 3))
            user_prompt = (
                f"Write {min(batch_size, todo - i)} DIFFERENT short multi-turn conversations in authentic "
                f"Nairobi Sheng between {roles}. Scenario: {scenario}. Tone: {tone}. "
                f"Each conversation must be 4-8 turns total, very colloquial, mixing Swahili, Sheng and "
                f"English naturally the way young Nairobians actually talk. Use words from the Sheng lexicon. "
                f"Output STRICT JSON array only, each element: "
                f'{{"turns": [{{"role": "user"|"bot", "text": "..."}}]}}\n\n'
                f"Example style:\n{fewshot}"
            )
            try:
                raw = chat(client, [
                    {"role": "system", "content": SHENG_SYSTEM_PROMPT + "\nYou output only valid JSON arrays."},
                    {"role": "user", "content": user_prompt},
                ])
                arr = json.loads(re.sub(r"^```(json)?|```$", "", raw.strip(), flags=re.MULTILINE))
                for conv in arr:
                    f.write(json.dumps({"scenario": scenario, "turns": conv["turns"]}, ensure_ascii=False) + "\n")
                f.flush()
                print(f"  +{len(arr)} (total {existing + i + len(arr)})")
            except Exception as e:
                error_msg = str(e)
                print(f"  batch failed: {error_msg}")
                
                # If we hit a rate limit, pause for 10 minutes to let tokens refill
                if "429" in error_msg or "rate limit" in error_msg.lower():
                    print("  -> Rate limit hit. Pausing for 10 minutes...")
                    time.sleep(600)
                else:
                    time.sleep(3) # Normal pause for random API glitches


# ---------------------------------------------------------------------------
def gen_intents(client, per_intent):
    path = OUT_DIR / "intents_raw.jsonl"
    existing_ids = set()
    if path.exists():
        existing_ids = {json.loads(l)["intent"] for l in path.read_text().splitlines() if l.strip()}
    with path.open("a", encoding="utf-8") as f:
        for intent in INTENTS:
            if intent in existing_ids:
                continue
            lex_sample = ", ".join(random.sample(sorted(LEXICON_WORDS), 15))
            prompt = (
                f"Write {per_intent} DIFFERENT short user utterances in Nairobi Sheng that all express the "
                f"intent '{intent}'. Vary length (3-15 words), phrasing, spelling style, and how much Swahili "
                f"vs English is mixed. Make some messy/colloquial the way people actually type or speak. "
                f"Use Sheng words like: {lex_sample}. "
                f'Output STRICT JSON array of strings only: ["...", "..."]'
            )
            try:
                raw = chat(client, [{"role": "user", "content": prompt}], temperature=1.0)
                arr = json.loads(re.sub(r"^```(json)?|```$", "", raw.strip(), flags=re.MULTILINE))
                for utt in arr:
                    f.write(json.dumps({"intent": intent, "text": utt}, ensure_ascii=False) + "\n")
                print(f"  {intent}: +{len(arr)}")
            except Exception as e:
                print(f"  {intent} failed: {e}")
            time.sleep(20)


# ---------------------------------------------------------------------------
def filter_all():
    """Dedupe, drop non-Sheng or degenerate outputs. Keep validation candidates."""
    for name in ["conversations", "intents"]:
        src = OUT_DIR / f"{name}_raw.jsonl"
        if not src.exists():
            continue
        seen, kept = set(), []
        for line in src.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            obj = json.loads(line)
            text = json.dumps(obj, ensure_ascii=False)
            key = re.sub(r"\s+", " ", text.lower())
            if key in seen:
                continue
            seen.add(key)
            if name == "conversations":
                turns = obj["turns"]
                if not (4 <= len(turns) <= 12):
                    continue
                all_text = " ".join(t["text"] for t in turns).lower()
            else:
                all_text = obj["text"].lower()
                if not (3 <= len(obj["text"].split()) <= 25):
                    continue
            # Require some Sheng/swahili signal: at least 1 lexicon word or common function word
            if not any(w in all_text for w in LEXICON_WORDS | {"ni", "na", "ya", "leo", "sana", "hii"}):
                continue
            kept.append(obj)
        out = OUT_DIR / f"{name}_filtered.jsonl"
        out.write_text("\n".join(json.dumps(o, ensure_ascii=False) for o in kept), encoding="utf-8")
        print(f"{name}: kept {len(kept)} / {len(seen)} unique candidates -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["conversations", "intents", "filter"])
    ap.add_argument("--target", type=int, default=2000, help="total conversations to aim for")
    ap.add_argument("--batch", type=int, default=10, help="conversations per API call")
    ap.add_argument("--per_intent", type=int, default=60)
    a = ap.parse_args()
    if a.mode == "filter":
        filter_all()
    else:
        gen_conversations(get_client(), a.target, a.batch) if a.mode == "conversations" \
            else gen_intents(get_client(), a.per_intent)
