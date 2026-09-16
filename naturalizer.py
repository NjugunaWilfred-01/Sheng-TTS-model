"""
naturalizer.py - Injects fillers, pauses, follow-ups, and prosody hints into LLM output.

Two jobs, both aimed at the same failure: a correct reply that still sounds like a
machine read it.

  humanize()     - text-level. Real speech opens with a filler, breathes mid-sentence,
                   and hands the turn back. LLMs do none of that by default.
  vary_prosody() - audio-level. Fixed rate/pitch across every turn is the single
                   loudest "this is a TTS voice" signal. Jittering per turn, steered
                   by emotion, is what breaks the GPS-voice effect.

detect_emotion() is the bridge: its return value travels in meta["emotion"] from
llm_engine to pipeline, which feeds it to vary_prosody. Keep the three labels
("excited", "neutral", "thinking") in sync with the tables below and with the
meta-dict contract in Work_Plan.md section 0.3.
"""
import random

FILLERS = {
    "neutral": ["Eeeh", "Aah", "Sasa", "Hmm", "Kwani", "Alafu"],
    "excited": ["Wueeh", "Aki", "Eish", "Bana"],
    "thinking": ["Kwani", "Maze", "Alafu", "Hmm"],
}

FOLLOWUPS = [
    "Rada yako?",
    "Wewe unasema aje?",
    "Kwani wewe?",
    "Uko aje upande wako?",
    "Niambie zaidi.",
]

FILLER_PROB = 0.60
PAUSE_PROB = 0.35
FOLLOWUP_PROB = 0.40

_ALL_LOWER = {f.lower() for group in FILLERS.values() for f in group}

_EXCITED_CUES = ("party", "sherehe", "weekend", "turn up", "raha", "fiti sana",
                 "moto", "luku", "alchemist", "sherehe")
_THINKING_CUES = ("choka", "stress", "uchovu", "pole", "shida", "noma", "jam",
                  "foleni", "sota")


def detect_emotion(text: str) -> str:
    """Classify a turn into one of the three prosody buckets."""
    if not text:
        return "neutral"
    t = text.lower()
    if any(w in t for w in _EXCITED_CUES):
        return "excited"
    if any(w in t for w in _THINKING_CUES):
        return "thinking"
    return "neutral"


def humanize(text: str, emotion: str = None, add_followup: bool = True) -> str:
    """Add a filler, a breath pause, and sometimes a follow-up question."""
    if not text:
        return text
    text = text.strip()
    emotion = emotion or detect_emotion(text)
    fillers = FILLERS.get(emotion, FILLERS["neutral"])

    words = text.split()
    if not words:
        return text

    # Filler, unless the model already opened with one.
    if random.random() < FILLER_PROB:
        first = words[0].rstrip(",.!?").lower()
        if first not in _ALL_LOWER:
            body = text[0].lower() + text[1:]
            text = f"{random.choice(fillers)}, {body}"

    # Mid-sentence breath. Edge-TTS renders "..." as a genuine pause.
    words = text.split()
    if len(words) > 8 and random.random() < PAUSE_PROB:
        i = random.randint(3, len(words) - 3)
        words.insert(i, "...")
        text = " ".join(words)

    # Hand the turn back so the conversation keeps moving.
    if add_followup and random.random() < FOLLOWUP_PROB:
        if not text.rstrip().endswith("?"):
            text = text.rstrip(".!") + ". " + random.choice(FOLLOWUPS)

    return text


def vary_prosody(emotion: str = "neutral"):
    """
    Return (rate, pitch) strings for Edge-TTS, jittered within an emotion band.

    Ranges are deliberately narrow. Beyond roughly +/-10% rate and +/-6Hz pitch the
    neural voice starts sounding processed rather than expressive.
    """
    table = {
        "excited": (["+5%", "+8%", "+10%"], ["+3Hz", "+5Hz", "+6Hz"]),
        "neutral": (["-3%", "+0%", "+2%"], ["-2Hz", "+0Hz", "+2Hz"]),
        "thinking": (["-8%", "-5%", "-3%"], ["-4Hz", "-2Hz", "+0Hz"]),
    }
    rates, pitches = table.get(emotion, table["neutral"])
    return random.choice(rates), random.choice(pitches)
