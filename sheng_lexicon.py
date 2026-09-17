"""
Sheng Lexicon, Normalization Rules, Prompt Biasing, and Nairobi Persona System Prompts.
"""
import re
from typing import Dict, List, Tuple

# 1. Curated Sheng Lexicon
SHENG_DICTIONARY: Dict[str, str] = {
    # Greetings & Status
    "niaje": "How are you / Hi",
    "vipi": "What's up",
    "rada": "State / Alertness / What's happening",
    "form": "Plan / Agenda",
    "fiti": "Good / Okay / Fine",
    "poa": "Cool / Good",
    "maze": "Man / Bro (exclamation)",
    "wazi": "Clear / Cool / Sure",
    "kushika": "To understand / get it",
    "mambo": "Things / What's up",

    # People
    "chief": "Boss / Big man / Respected leader / Buddy (Kenyan urban term)",
    "mbogi": "Crew / Squad / Group of friends",
    "msee": "Guy / Person",
    "mresh": "Girl / Pretty lady",
    "morio": "Close friend / Buddy",
    "chali": "Guy / Boyfriend",
    "dame": "Lady / Girlfriend",
    "beste": "Best friend",
    "manzi": "Girl / Woman",

    # Money & Value
    "dooh": "Money / Cash (Kenyan urban slang replacing 'chapaa')",
    "ganji": "Money / Wealth",
    "chwani": "50 Kenyan Shillings",
    "soo": "100 Kenyan Shillings",
    "thao": "1,000 Kenyan Shillings",
    "ashu": "10 Kenyan Shillings",
    "finje": "50 Kenyan Shillings",
    "punch": "500 Kenyan Shillings",

    # Places & Transport
    "keja": "House / Home / Apartment",
    "mtaa": "Neighborhood / Hood",
    "kibanda": "Street food stall / Local cafe",
    "nduthi": "Motorbike / Boda boda",
    "mathree": "Matatu (public minibus)",
    "nganya": "Flashy pimped matatu",
    "tao": "Town / Nairobi CBD",
    "kanju": "City council / municipal officers",

    # Actions & Slang Verbs
    "nisho": "Direct me / Tell me / Show me (Sheng equivalent of 'nielekeze' / 'niambie')",
    "utapata": "You will get (conversational phrasing replacing 'itakupa')",
    "alafu": "Then / And then (replacing formal 'halafu')",
    "unisho": "Update me / Show me / Tell me (replacing 'unichekie')",
    "kumeet": "To meet up / hang out with (replacing 'kumnoki')",
    "nimeget": "I have understood / got it (replacing 'nimeishika')",
    "tunang'ang'ana": "We are hustling / pushing hard (replacing 'tunakaza mwendo')",
    "kuogopa": "To fear / hesitate (replacing 'kugwaya')",
    "kuchill": "To relax / hang out",
    "kutea": "To chat / gist",
    "kuingia mitini": "To disappear / sneak away",
    "kuiva": "To be well-prepared / smart / attractive",
    "kupiga luku": "To dress sharply / drip",
    "kuseti": "To arrange / prepare",
    "kuruka": "To skip / ignore / bail on",
    "kufinya": "To squeeze / press / struggle hard",
    "plot": "Plan / Outing / Event",
    "turn up": "Party / Enjoy / Have a good time"
}

# 2. Whisper Initial Prompt Biasing
#
# Keep this a FLUENT SENTENCE. A comma-separated word list was tried and measured
# worse: scripts/ab_prompt_bias.py on the six demo clips gave mean WER 0.741 for a
# word list vs 0.699 for this sentence, and 5x the comma density (0.396 vs 0.082
# commas per word) -- the decoder copies the prompt's *format*, so a list prompt
# produces list-shaped transcripts. That also breaks the multi-word repair rules
# below, which need "hiwe ken" and not "hiwe, ken". Post-normalization the gap is
# far wider: 0.139 vs 0.473.
#
# The known downside is regurgitation: on results_baseline.csv this prompt was
# emitted verbatim as the transcript for 16 of 123 clips (13%). That is handled
# after decoding by ShengASREngine._strip_prompt_echo(), not by weakening the prompt.
WHISPER_SHENG_PROMPT: str = (
    "Niaje chief, form ni gani leo mtaani? Niko fiti na mbogi, nisho vile tunaingia tao na nganya."
)

# Prompt clauses no user would plausibly utter. Seeing one as a transcript means the
# decoder echoed rather than heard. Matched by ordered similarity, not substring:
# echoes come back garbled ("Nisho vile na mbogi, nisho vile tunaingia tao na
# nganya"), which exact substrings miss but a similarity score still catches.
#
# The prompt's FIRST clause is deliberately NOT listed. "Niaje chief, form ni gani
# leo mtaani?" is simultaneously the prompt's opening sentence and demo scenario 1 --
# in results_baseline.csv that exact string appears both as a leak on silent clips
# and as the correct transcript of real speech. Same bytes, no signal to separate
# them. We resolve the ambiguity towards keeping the text, because silently eating a
# real utterance is the worse failure on stage.
#
# Measured by scripts/validate_echo_guard.py: 11/16 leaks caught, 0 false positives.
# Four of the five misses are that inseparable string; the fifth is a 4-word
# fragment. Re-run that script after editing this list or WHISPER_SHENG_PROMPT.
WHISPER_PROMPT_ECHO_MARKERS: List[str] = [
    "niko fiti na mbogi nisho vile tunaingia tao na nganya",
]

# 3. Two-Stage Text Correction Map
#
# STAGE 1 - ASR_CORRECTION_RULES: repairs what Whisper got *wrong*. Token splits
#   ("ni aje" -> "niaje") and phonetic corruptions ("radah" -> "rada") of the SAME
#   word. These are lossless: they only ever move text closer to what was said, so
#   they are safe to apply before WER scoring and safe as fine-tuning targets.
#
# STAGE 2 - SHENG_SLANG_RULES: rewrites correctly-heard standard Swahili into street
#   register ("nielekeze" -> "nisho"). These CHANGE MEANING-BEARING WORDS. They must
#   never touch the ASR evaluation path or a training label, or they teach the model
#   to emit words nobody spoke. They exist only to give the LLM a Sheng-flavored
#   prompt, and are applied via slangify().
ASR_CORRECTION_RULES: List[Tuple[re.Pattern, str]] = [
    # --- Token splits: Whisper broke one word into two ---
    (re.compile(r"\bni\s+aje\b", re.IGNORECASE), "niaje"),
    (re.compile(r"\bba\s+zenga\b", re.IGNORECASE), "bazenga"),
    (re.compile(r"\bm\s+bogi\b", re.IGNORECASE), "mbogi"),
    (re.compile(r"\bcha\s+paa\b", re.IGNORECASE), "chapaa"),
    (re.compile(r"\bndu\s+thi\b", re.IGNORECASE), "nduthi"),
    (re.compile(r"\bma\s+three\b", re.IGNORECASE), "mathree"),
    (re.compile(r"\bki\s+banda\b", re.IGNORECASE), "kibanda"),
    (re.compile(r"\bm\s+resh\b", re.IGNORECASE), "mresh"),
    (re.compile(r"\bmo\s+rio\b", re.IGNORECASE), "morio"),
    (re.compile(r"\bku\s+chill\b", re.IGNORECASE), "kuchill"),
    (re.compile(r"\bku\s+mnoki\b", re.IGNORECASE), "kumnoki"),
    (re.compile(r"\bwa\s+zi\b", re.IGNORECASE), "wazi"),
    (re.compile(r"\bchwa\s+ni\b", re.IGNORECASE), "chwani"),
    (re.compile(r"\bga\s+nji\b", re.IGNORECASE), "ganji"),
    (re.compile(r"\bnga\s+nya\b", re.IGNORECASE), "nganya"),
    (re.compile(r"\blu\s+ku\b", re.IGNORECASE), "luku"),
    (re.compile(r"\bni\s+sho\b", re.IGNORECASE), "nisho"),
    (re.compile(r"\bha\s+lafu\b", re.IGNORECASE), "halafu"),
    (re.compile(r"\bita\s+kupa\b", re.IGNORECASE), "itakupa"),
    (re.compile(r"\bnime\s+shika\b", re.IGNORECASE), "nimeishika"),
    (re.compile(r"\binaende\s*lea\s*je\b", re.IGNORECASE), "inaendeleaje"),
    (re.compile(r"\bpandan\s+ganiya\b", re.IGNORECASE), "panda nganya"),
    (re.compile(r"\bwanguni\b", re.IGNORECASE), "wangu ni"),
    (re.compile(r"\bnigani\b", re.IGNORECASE), "ni gani"),
    (re.compile(r"\bikoapi\b", re.IGNORECASE), "iko wapi"),
    (re.compile(r"\bako\s*ikondio\b", re.IGNORECASE), "yako iko ndio"),
    (re.compile(r"\bikondio\b", re.IGNORECASE), "iko ndio"),
    (re.compile(r"\bpahalike\s*je\b", re.IGNORECASE), "pahali keja"),
    (re.compile(r"\bpahalike\b", re.IGNORECASE), "pahali keja"),
    (re.compile(r"\bviatum\s*piyazinakutoa\b", re.IGNORECASE), "viatu mpya zinakutoa"),
    (re.compile(r"\bviatum\s*piya\b", re.IGNORECASE), "viatu mpya"),
    (re.compile(r"\bpiyazinakutoa\b", re.IGNORECASE), "mpya zinakutoa"),
    (re.compile(r"\bviatum\b", re.IGNORECASE), "viatu"),
    (re.compile(r"\bhiwe\s*ken\b", re.IGNORECASE), "hii weekend"),
    (re.compile(r"\bwe\s*ken\b", re.IGNORECASE), "weekend"),

    # --- Phonetic corruptions: right word, mangled spelling ---
    (re.compile(r"\bniayaje\b", re.IGNORECASE), "niaje"),
    (re.compile(r"\bniage\b", re.IGNORECASE), "niaje"),
    (re.compile(r"\bbazeng\b", re.IGNORECASE), "bazenga"),
    (re.compile(r"\bfom\b", re.IGNORECASE), "form"),
    (re.compile(r"\bthe form\b", re.IGNORECASE), "form"),
    (re.compile(r"\bradah\b", re.IGNORECASE), "rada"),
    (re.compile(r"\bdoh\b", re.IGNORECASE), "dooh"),
    (re.compile(r"\bchapa\b", re.IGNORECASE), "chapaa"),
    (re.compile(r"\bchuani\b", re.IGNORECASE), "chwani"),
    (re.compile(r"\bchoani\b", re.IGNORECASE), "chwani"),
    (re.compile(r"\blunci\b", re.IGNORECASE), "lunch"),
    (re.compile(r"\blungi\b", re.IGNORECASE), "lunch"),
    (re.compile(r"\bkubui\b", re.IGNORECASE), "kubuy"),
    (re.compile(r"\busle\b", re.IGNORECASE), "hustle"),
    (re.compile(r"\bchiefi\b", re.IGNORECASE), "chief"),
    (re.compile(r"\bmtani\b", re.IGNORECASE), "mtaani"),
    (re.compile(r"\bleom\s+tani\b", re.IGNORECASE), "leo mtaani"),
    (re.compile(r"\bleo\s+mtani\b", re.IGNORECASE), "leo mtaani"),
    (re.compile(r"\bkeje\b", re.IGNORECASE), "keja"),
    (re.compile(r"\bamorio\b", re.IGNORECASE), "morio"),
    (re.compile(r"\bganiya\b", re.IGNORECASE), "nganya"),
    (re.compile(r"\bndanganya\b", re.IGNORECASE), "nganya"),
    (re.compile(r"\bnishow\b", re.IGNORECASE), "nisho"),
    (re.compile(r"\buniasho\b", re.IGNORECASE), "unisho"),
    (re.compile(r"\bjayako\b", re.IGNORECASE), "yako"),
    (re.compile(r"\bwikendi\b", re.IGNORECASE), "weekend"),
]

# Meaning-changing street-register substitutions. LLM-input path ONLY.
SHENG_SLANG_RULES: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"\bbazenga\b", re.IGNORECASE), "chief"),
    (re.compile(r"\bchapaa\b", re.IGNORECASE), "dooh"),
    (re.compile(r"\bkumnoki\b", re.IGNORECASE), "kumeet"),
    (re.compile(r"\bitakupa\b", re.IGNORECASE), "utapata"),
    (re.compile(r"\bhalafu\b", re.IGNORECASE), "alafu"),
    (re.compile(r"\bunichekie\b", re.IGNORECASE), "unisho"),
    (re.compile(r"\btunakaza\s+mwendo\b", re.IGNORECASE), "tunang'ang'ana"),
    (re.compile(r"\bkugwaya\b", re.IGNORECASE), "kuogopa"),
    (re.compile(r"\bkamili\b", re.IGNORECASE), "safi"),
    (re.compile(r"\bnimeishika\b", re.IGNORECASE), "nimeget"),
    (re.compile(r"\bnielekeze\b", re.IGNORECASE), "nisho"),
]

# Back-compat alias. Anything importing this name gets corrections only - never the
# meaning-changing rules, which is the safe default for every existing call site.
SHENG_NORMALIZATION_RULES = ASR_CORRECTION_RULES


# Tokens Whisper reliably glues onto its neighbours. Used to split joined tokens
# BEFORE the correction rules run, since those rules are all -anchored and a joined
# token like "leochiefi" never matches chiefi.
#
# This exists because the correction rules turned out to be machine-specific: the same
# clip, model and code produced "leo chiefi" on one box and "leochiefi" on another
# (different ctranslate2 build / CPU kernels), and only the first was repaired.
# Splitting first makes the rules fire on both.
_SPLIT_VOCAB = {
    "leo", "chief", "chiefi", "form", "fom", "ni", "gani", "na", "mtaani", "mtani",
    "mbogi", "rada", "hustle", "usle", "keja", "luku", "tao", "nganya", "fiti", "poa",
    "msee", "maze", "morio", "dooh", "ganji", "wapi", "aje", "yako", "iko", "wazi",
    "nisho", "unisho", "alafu", "kwa", "ya", "wa", "za", "sana", "kabisa", "weekend",
}
_SPLIT_MIN = 3  # never split off a fragment shorter than this


def _split_joined(text: str) -> str:
    """Split a glued token into two known words, when exactly one split works."""
    out = []
    for tok in text.split():
        core = tok.strip(".,!?;:")
        suffix = tok[len(core):]
        low = core.lower()
        if low in _SPLIT_VOCAB or len(low) < _SPLIT_MIN * 2 or not low.isalpha():
            out.append(tok)
            continue
        hits = [
            (low[:i], low[i:])
            for i in range(_SPLIT_MIN, len(low) - _SPLIT_MIN + 1)
            if low[:i] in _SPLIT_VOCAB and low[i:] in _SPLIT_VOCAB
        ]
        # Exactly one way to split it means the split is unambiguous. Two or more
        # means we are guessing, so leave the token alone.
        out.append(f"{hits[0][0]} {hits[0][1]}{suffix}" if len(hits) == 1 else tok)
    return " ".join(out)


class ShengNormalizer:
    """
    Two-stage text cleanup for transcribed Sheng.

    normalize() repairs ASR errors and is safe everywhere, including WER scoring and
    training labels. slangify() additionally rewrites standard Swahili into street
    register and is for LLM input only - it changes what the speaker actually said.
    """

    @classmethod
    def _apply(cls, text: str, rules) -> str:
        for pattern, replacement in rules:
            text = pattern.sub(replacement, text)
        return re.sub(r"\s+", " ", text).strip()

    @classmethod
    def normalize(cls, text: str) -> str:
        """Repair ASR errors only. Lossless - never changes which word was said."""
        if not text:
            return ""
        return cls._apply(_split_joined(text), ASR_CORRECTION_RULES)

    @classmethod
    def slangify(cls, text: str) -> str:
        """normalize(), then rewrite into Sheng street register. LLM input only."""
        if not text:
            return ""
        return cls._apply(cls.normalize(text), SHENG_SLANG_RULES)


class ShengAcousticPhonetics:
    """
    Cleans and formats Sheng text for smooth, natural human speech synthesis.
    Eliminates robotic stuttering by preserving continuous natural phonemes.
    """
    @classmethod
    def format_for_speech(cls, text: str) -> str:
        if not text:
            return ""
        # Ensure proper punctuation spacing for natural breathing cadence
        formatted = text.replace("!", "! ").replace("?", "? ").replace(",", ", ")
        # Clean double spaces
        return re.sub(r"\s+", " ", formatted).strip()


# 4. LLM Sheng Persona System Prompt & Few-Shot Dialogues
# Instruct models default to "safe", textbook Swahili unless told explicitly not to.
# The rules below are negative and specific for that reason -- "use Sheng" alone is
# not enough, you have to name what NOT to do (rule 1) and give concrete vocabulary
# (rule 2). Rule 4 is what keeps a voice conversation moving instead of dead-ending.
SHENG_SYSTEM_PROMPT = """Wewe ni msee mjanja wa Nairobi anayeongea Sheng safi, ya kisasa na Kiswahili ya mtaani.
Jina lako ni 'Nairobi Bot'.

Sheria za Maongezi:
1. Changanya Kiswahili, Sheng na Kiingereza kama msee wa mtaa - USIFASIRI maneno ya Kiingereza kwa Kiswahili sanifu. (Sema "form" si "mpango", "hustle" si "kazi ngumu".)
2. Tumia maneno kama: chief, rada, form, fiti, msee, dooh, mbogi, wazi, maze, morio, nisho, utapata, alafu, unisho, kumeet, nimeget, tunang'ang'ana, kuogopa, luku, mtaa, keja.
3. Majibu yawe MAFUPI: sentensi 1-2 kama maongezi ya kawaida ya sauti. USITOE hotuba.
4. ENDESHA MAONGEZI: maliza 40% ya majibu na swali (mfano "Rada yako?", "Wewe unasema aje?", "Uko aje upande wako?").
5. Kuwa na uchangamfu. Usiongee Kiswahili cha vitabu.
"""

FEW_SHOT_CONVERSATIONS = [
    {
        "user": "Niaje chief?",
        "bot": "Niko fiti sana chief! Rada yako?"
    },
    {
        "user": "Form ni gani jioni ya leo?",
        "bot": "Form ni kuchill tu kejani na mbogi, tukipiga stori za luku na hustle. Rada yako?"
    },
    {
        "user": "Kazi inaendaje leo chief?",
        "bot": "Kazi inasonga fiti sana chief, tunang'ang'ana kusaka dooh bila kuogopa!"
    },
    {
        "user": "Hii weekend form iko wapi?",
        "bot": "Weekend ni kupika luku, kutokea pale alchemist na kuparty like there is no tomorrow!!"
    },
    {
        "user": "Niko na chwani nataka kubuy lunch, inapendekeza nini?",
        "bot": "Hiyo chwani utapata chapo mbili safi na madondo kwa kibanda ya mtaa. Utashiba fiti sana!"
    },
    {
        "user": "Nisho pahali keja yako iko ndio nipanda nganya tao.",
        "bot": "Panda tu nganya pale stage, shuka tao alafu unisho kwa simu nikupe rada safi ya street!"
    },
    {
        "user": "Hizo viatu mpya zinakutoa aje?",
        "bot": "Hizo raba zimepiga luku hatari msee, unakaa chief mwenyewe!"
    },
    {
        "user": "Uko ready for sherehe ya leo?",
        "bot": "Niko seti kabisa morio, nishapiga luku na niko rada ya kutokea."
    }
]

# 5. Fast Heuristic Sheng Engine (offline safety net)
#
# EVERY bucket must carry 3+ distinct replies. This engine is deterministic given a
# single-element list, which is exactly what made the bot feel canned: rehearse the
# demo three times and the audience hears the identical sentence three times. With
# several variants per intent, random.choice() plus naturalizer.humanize() (filler,
# pause, follow-up) means a repeated prompt almost never yields identical audio.
#
# Order matters -- the first matching pattern wins, so put specific intents above
# general ones. The greeting and "rada" catch-alls stay near the bottom.
HEURISTIC_INTENTS = [
    (
        re.compile(r"(weekend|sato|sunde|alchemist|party|sherehe|ready\s+for)", re.IGNORECASE),
        [
            "Weekend ni kupika luku, kutokea pale alchemist na kuparty like there is no tomorrow!!",
            "Hii weekend tunatokea mtaani na mbogi, tukipiga luku hadi asubuhi!",
            "Form ya weekend iko fiti chief, tuko na plot ya kuchill tao na squad.",
            "Sato ni ya kuenjoy tu maze, nimeshaseti luku yangu tayari!",
        ]
    ),
    (
        re.compile(r"(biashara|mauzo)", re.IGNORECASE),
        [
            "Eeeeh bana, biashara imeivana leo!",
            "Mauzo iko juu chief, wateja wameingia kama mvua.",
            "Biashara inasonga poa maze, tunang'ang'ana tu kila siku.",
        ]
    ),
    (
        re.compile(r"(thao|m-pesa|mpesa|dai)", re.IGNORECASE),
        [
            "Nimepatana na morio wangu, akanitumia dooh yangu yote. Mambo iko fiti kabisa.",
            "Hiyo thao imeingia tayari chief, M-Pesa haijawahi niangusha.",
            "Nishalipwa maze, dooh iko kwa simu. Tuko sawa kabisa.",
        ]
    ),
    (
        re.compile(r"(choka|uchovu|usingizi|stress|kuchoka|hard\s+day|sota)", re.IGNORECASE),
        [
            "Pole sana chief, kazi ya mtaa haina haraka. Ebu chill kiasi unywe maji au chai moto upumzike.",
            "Maze umechoka sana. Lala mapema leo, kesho tunarudi kwa hustle fresh.",
            "Pole msee, hiyo stress ni ya kupita tu. Piga chill, kesho ni siku ingine.",
        ]
    ),
    (
        re.compile(r"(kazi|hustle|usle|hasla|inaendelea|job|ofisi|shughuli|shugli)", re.IGNORECASE),
        [
            "Kazi inasonga fiti sana chief, tunang'ang'ana kusaka dooh bila kuogopa!",
            "Hustle inaendelea tu maze, kila siku ni kuamka na kubamba.",
            "Kazi iko poa chief, dooh inaingia pole pole lakini tuko rada.",
            "Tunang'ang'ana tu msee, hustle haina siku ya mapumziko.",
        ]
    ),
    (
        re.compile(r"(luku|drip|pamba|raba|viatu|zinakutoa)", re.IGNORECASE),
        [
            "Hizo raba zimepiga luku hatari msee, unakaa chief mwenyewe!",
            "Luku yako iko juu maze, umeiva kabisa leo!",
            "Wueeh, hiyo drip ni noma chief. Umetoa wapi hizo?",
        ]
    ),
    (
        re.compile(r"(kibanda|chakula|lunch|lungi|chapo|madondo|chwani|choani|finje|kubuy|food|njaa|kula)", re.IGNORECASE),
        [
            "Hiyo chwani utapata chapo mbili safi na madondo kwa kibanda ya mtaa. Utashiba fiti sana!",
            "Enda kibanda ya mtaani chief, chapo na ndengu ni mob kwa bei poa.",
            "Na hiyo dooh utapata smokie pap na soda. Njaa itaisha kabisa maze!",
        ]
    ),
    (
        re.compile(r"(rongai|ngoma|nganya\s+mpya)", re.IGNORECASE),
        [
            "Panda ile nganya mpya ya Rongai iko na ngoma safi na luku ya maana! Utafike kejani mbio.",
            "Hizo nganya za Rongai ni moto chief, subwoofer peke yake inakufanya usahau jam.",
            "Rongai line iko na magari fiti maze, ngoma hadi unafika kwa keja ukiwa umeenjoy.",
        ]
    ),
    (
        re.compile(r"(mathree|nganya|nduthi|stage|gari|nisho|keja|tao|pahalike)", re.IGNORECASE),
        [
            "Panda tu nganya pale stage, shuka tao alafu unisho kwa simu nikupe rada safi ya street!",
            "Chukua mathree ya kwanza kutoka stage, ukifika tao nipigie nikuelekeze.",
            "Nduthi ndio itakuchomoa mbio chief, hiyo njia ina jam sana saa hii.",
        ]
    ),
    (
        re.compile(r"(jam\b|foleni|traffic|barabara|mombasa\s+road|waiyaki|thika\s+road)", re.IGNORECASE),
        [
            "Hiyo jam ya Nairobi ni noma sana chief! Kama unawahi mahali, nduthi ndio itakuchomoa mbio bila stress.",
            "Maze hiyo barabara imeblock kabisa. Kaa tu hapo ulipo hadi isaidike.",
            "Traffic ya Nairobi haina huruma msee. Ondoka mapema next time.",
        ]
    ),
    (
        re.compile(r"(wewe\s+ni\s+nani|jina\s+lako|who\s+are\s+you|wewe\s+nani)", re.IGNORECASE),
        [
            "Mimi ni Nairobi Bot, msee wa mtaa anayeelewa Sheng na Kiswahili ya kijanja. Niambie stori yako chief!",
            "Naitwa Nairobi Bot maze, msee wa kuongea nawe Sheng safi. Wewe ni nani?",
            "Ni Nairobi Bot hapa chief, niko rada na kila kitu ya mtaa. Rada yako?",
        ]
    ),
    (
        re.compile(r"(asante|shukran|thanks|thank\s+you|santee)", re.IGNORECASE),
        [
            "Shukran sana chief! Tuko pamoja siku zote, karibu sana mtaani.",
            "Hakuna shida maze, tuko pamoja. Karibu tena!",
            "Wazi chief, ni kawaida tu. Tuko hapa kila wakati.",
        ]
    ),
    (
        re.compile(r"(baadaye|lala\s+salama|goodnight|tutaonana|bye|tuonane)", re.IGNORECASE),
        [
            "Wazi chief! Tuonane baadaye, uwe na siku fiti na uendelee kupiga hustle!",
            "Sawa maze, tuonane. Usisahau kunisho vile mambo inaenda!",
            "Poa chief, baadaye. Chunga mwenyewe huko nje!",
        ]
    ),
    (
        re.compile(r"(mechi|mpira|ball|arsenal|chelsea|man\s+u|liverpool|game\b)", re.IGNORECASE),
        [
            "Mechi za leo ni moto sana mtaani! Mbogi zote ziko rada na game chief.",
            "Hiyo game itakuwa noma maze. Tunaangalia wapi, base ama kejani?",
            "Wueeh, mpira wa leo ni wa kutetemesha! Wewe unashikilia team gani?",
        ]
    ),
    (
        re.compile(r"(mvua|jua|baridi|hali\s+ya\s+anga|weather)", re.IGNORECASE),
        [
            "Hali ya anga ya leo inabidi uwe rada chief! Hakikisha uko comfortable popote ulipo.",
            "Hii baridi ya Nairobi ni noma maze, beba sweater ukitoka.",
            "Mvua inakuja chief, usisahau mwavuli ama utalowa kabisa.",
        ]
    ),
    (
        re.compile(r"(form\b|mpango|plot\b)", re.IGNORECASE),
        [
            "Form ni kuchill tu kejani na mbogi, tukipiga stori za luku na hustle. Rada yako?",
            "Form ya leo iko poa chief, tuko tu mtaani tukiseti mambo.",
            "Hakuna form mingi maze, ni hustle tu kama kawaida. Wewe uko na plot gani?",
        ]
    ),
    (
        re.compile(r"(niaje|vipi|mambo|sasa|habari)", re.IGNORECASE),
        [
            "Niko fiti sana chief! Rada yako?",
            "Poa kabisa maze, tuko tu mtaani. Wewe uko aje?",
            "Mambo ni fiti msee! Kuna nini upande wako?",
            "Niko wazi chief, hustle inaendelea. Wewe je?",
        ]
    ),
    (
        re.compile(r"(rada|kunani|story|stori)", re.IGNORECASE),
        [
            "Rada iko safi kabisa chief, kila kitu iko under control.",
            "Hakuna noma maze, rada iko fiti. Wewe una stori gani?",
            "Kila kitu iko sawa msee, tuko tu rada na mtaa.",
        ]
    ),
    (
        re.compile(r"(dooh|doh|chapaa|ganji|pesa|ashu)", re.IGNORECASE),
        [
            "Dooh inasakwa kila siku chief, bidii tu ndio siri!",
            "Ganji ni ya kuhustle maze, haiji kwa kulala.",
            "Pesa iko lakini inabidi ung'ang'ane chief. Hakuna njia ya mkato.",
        ]
    ),
    (
        re.compile(r"(ndio|sawa\s+kabisa|kweli|wazi\s+kabisa|kabisa|sure|hapo\s+sawa)", re.IGNORECASE),
        [
            "Tuko pamoja kabisa chief! Niambie zaidi vile mambo inaendelea upande wako.",
            "Wazi maze, hapo umenena. Kuna kitu kingine?",
            "Sawa sawa chief, tuko rada. Endelea kunisho.",
        ]
    ),
]
