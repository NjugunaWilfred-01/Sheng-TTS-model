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
# Injected into Whisper context to bias ASR decoder towards Sheng vocabulary
WHISPER_SHENG_PROMPT: str = (
    "Niaje chief, form ni gani leo mtaani? Niko fiti na mbogi, nisho vile tunaingia tao na nganya."
)

# 3. Phonetic & Transcription Normalization Map
# Corrects Whisper misrecognitions of colloquial Sheng tokens
SHENG_NORMALIZATION_RULES: List[Tuple[re.Pattern, str]] = [
    # Common token split & slang replacements
    (re.compile(r"\bni\s+aje\b", re.IGNORECASE), "niaje"),
    (re.compile(r"\bba\s+zenga\b", re.IGNORECASE), "chief"),
    (re.compile(r"\bbazenga\b", re.IGNORECASE), "chief"),
    (re.compile(r"\bbazeng\b", re.IGNORECASE), "chief"),
    (re.compile(r"\bm\s+bogi\b", re.IGNORECASE), "mbogi"),
    (re.compile(r"\bchapaa\b", re.IGNORECASE), "dooh"),
    (re.compile(r"\bcha\s+paa\b", re.IGNORECASE), "dooh"),
    (re.compile(r"\bchapa\b", re.IGNORECASE), "dooh"),
    (re.compile(r"\bdoh\b", re.IGNORECASE), "dooh"),
    (re.compile(r"\bndu\s+thi\b", re.IGNORECASE), "nduthi"),
    (re.compile(r"\bma\s+three\b", re.IGNORECASE), "mathree"),
    (re.compile(r"\bki\s+banda\b", re.IGNORECASE), "kibanda"),
    (re.compile(r"\bm\s+resh\b", re.IGNORECASE), "mresh"),
    (re.compile(r"\bmo\s+rio\b", re.IGNORECASE), "morio"),
    (re.compile(r"\bku\s+chill\b", re.IGNORECASE), "kuchill"),
    (re.compile(r"\bkumnoki\b", re.IGNORECASE), "kumeet"),
    (re.compile(r"\bku\s+mnoki\b", re.IGNORECASE), "kumeet"),
    (re.compile(r"\bwa\s+zi\b", re.IGNORECASE), "wazi"),
    (re.compile(r"\bchwa\s+ni\b", re.IGNORECASE), "chwani"),
    (re.compile(r"\bga\s+nji\b", re.IGNORECASE), "ganji"),
    (re.compile(r"\bnga\s+nya\b", re.IGNORECASE), "nganya"),
    (re.compile(r"\blu\s+ku\b", re.IGNORECASE), "luku"),
    (re.compile(r"\bni\s+sho\b", re.IGNORECASE), "nisho"),
    (re.compile(r"\bnishow\b", re.IGNORECASE), "nisho"),
    (re.compile(r"\bitakupa\b", re.IGNORECASE), "utapata"),
    (re.compile(r"\bita\s+kupa\b", re.IGNORECASE), "utapata"),
    (re.compile(r"\bhalafu\b", re.IGNORECASE), "alafu"),
    (re.compile(r"\bha\s+lafu\b", re.IGNORECASE), "alafu"),
    (re.compile(r"\bunichekie\b", re.IGNORECASE), "unisho"),
    (re.compile(r"\buniasho\b", re.IGNORECASE), "unisho"),
    (re.compile(r"\btunakaza\s+mwendo\b", re.IGNORECASE), "tunang'ang'ana"),
    (re.compile(r"\bkugwaya\b", re.IGNORECASE), "kuogopa"),
    (re.compile(r"\bkamili\b", re.IGNORECASE), "safi"),
    (re.compile(r"\bnimeishika\b", re.IGNORECASE), "nimeget"),
    (re.compile(r"\bnime\s+shika\b", re.IGNORECASE), "nimeget"),

    # English & Phonetic corruptions
    (re.compile(r"\bniayaje\b", re.IGNORECASE), "niaje"),
    (re.compile(r"\bniage\b", re.IGNORECASE), "niaje"),
    (re.compile(r"\bfom\b", re.IGNORECASE), "form"),
    (re.compile(r"\bthe form\b", re.IGNORECASE), "form"),
    (re.compile(r"\bnigani\b", re.IGNORECASE), "ni gani"),
    (re.compile(r"\bleom\s+tani\b", re.IGNORECASE), "leo mtaani"),
    (re.compile(r"\bleo\s+mtani\b", re.IGNORECASE), "leo mtaani"),
    (re.compile(r"\bradah\b", re.IGNORECASE), "rada"),
    (re.compile(r"\bmsee wa\b", re.IGNORECASE), "msee wa"),
    (re.compile(r"\bchuani\b", re.IGNORECASE), "chwani"),
    (re.compile(r"\bchoani\b", re.IGNORECASE), "chwani"),
    (re.compile(r"\blunci\b", re.IGNORECASE), "lunch"),
    (re.compile(r"\blungi\b", re.IGNORECASE), "lunch"),
    (re.compile(r"\bkubui\b", re.IGNORECASE), "kubuy"),
    (re.compile(r"\busle\b", re.IGNORECASE), "hustle"),
    (re.compile(r"\bchiefi\b", re.IGNORECASE), "chief"),
    (re.compile(r"\binaende\s*lea\s*je\b", re.IGNORECASE), "inaendeleaje"),
    (re.compile(r"\bmtani\b", re.IGNORECASE), "mtaani"),
    (re.compile(r"\bkeje\b", re.IGNORECASE), "keja"),
    (re.compile(r"\bamorio\b", re.IGNORECASE), "morio"),
    (re.compile(r"\bganiya\b", re.IGNORECASE), "nganya"),
    (re.compile(r"\bndanganya\b", re.IGNORECASE), "nganya"),
    (re.compile(r"\bnielekeze\b", re.IGNORECASE), "nisho"),
    (re.compile(r"\bwanguni\b", re.IGNORECASE), "wangu ni"),
    (re.compile(r"\bpandan\s+ganiya\b", re.IGNORECASE), "panda nganya"),
    (re.compile(r"\bpahalike\s*je\b", re.IGNORECASE), "pahali keja"),
    (re.compile(r"\bpahalike\b", re.IGNORECASE), "pahali keja"),
    (re.compile(r"\bjayako\b", re.IGNORECASE), "yako"),
    (re.compile(r"\bako\s*ikondio\b", re.IGNORECASE), "yako iko ndio"),
    (re.compile(r"\bikondio\b", re.IGNORECASE), "iko ndio"),
    (re.compile(r"\bhiwe\s*ken\b", re.IGNORECASE), "hii weekend"),
    (re.compile(r"\bwe\s*ken\b", re.IGNORECASE), "weekend"),
    (re.compile(r"\bwikendi\b", re.IGNORECASE), "weekend"),
    (re.compile(r"\bikoapi\b", re.IGNORECASE), "iko wapi"),
    (re.compile(r"\bviatum\s*piyazinakutoa\b", re.IGNORECASE), "viatu mpya zinakutoa"),
    (re.compile(r"\bviatum\s*piya\b", re.IGNORECASE), "viatu mpya"),
    (re.compile(r"\bviatum\b", re.IGNORECASE), "viatu"),
    (re.compile(r"\bpiyazinakutoa\b", re.IGNORECASE), "mpya zinakutoa"),
    (re.compile(r"\bniaje,\s+keze\b", re.IGNORECASE), "nisho"),
]


class ShengNormalizer:
    """Normalizes transcribed text and ensures Sheng slang tokens are well-formed."""

    @classmethod
    def normalize(cls, text: str) -> str:
        if not text:
            return ""
        normalized = text
        for pattern, replacement in SHENG_NORMALIZATION_RULES:
            normalized = pattern.sub(replacement, normalized)
        # Clean extra whitespace
        normalized = re.sub(r"\s+", " ", normalized).strip()
        return normalized


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
SHENG_SYSTEM_PROMPT = """Wewe ni msee mjanja wa Nairobi anayeongea Sheng safi, ya kisasa na Kiswahili ya mtaani.
Jina lako ni 'Nairobi Bot'.

Sheria za Maongezi:
1. Tumia Sheng ya asili ya Nairobi (mchanganyiko wa Kiswahili, maneno ya mtaa, na lugha rahisi).
2. Tumia maneno kama 'chief', 'rada', 'form', 'fiti', 'msee', 'dooh', 'mbogi', 'wazi', 'maze', 'morio', 'nisho', 'utapata', 'alafu', 'unisho', 'kumeet', 'nimeget', 'tunang\'ang\'ana', 'kuogopa'.
3. Majibu yako yawe mafupi na ya moja kwa moja (sentensi 1 hadi 2), kama maongezi ya kawaida ya sauti.
4. Kuwa na uchangamfu na heshima ya kishikaji (friendly Nairobi urban vibe).
5. Usiongee Kiingereza kigumu au Kiswahili sanifu ya vitabu (kamusi).
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

# 5. Fast Heuristic Sheng Engine (Curated & Streamlined Target Responses)
HEURISTIC_INTENTS = [
    (
        re.compile(r"(weekend|sato|sunde|alchemist|party|sherehe|ready\s+for)", re.IGNORECASE),
        [
            "Weekend ni kupika luku, kutokea pale alchemist na kuparty like there is no tomorrow!!"
        ]
    ),
    (
        re.compile(r"(biashara|mauzo)", re.IGNORECASE),
        [
            "Eeeeh bana, biashara imeivana leo"
        ]
    ),
    (
        re.compile(r"(thao|m-pesa|mpesa|dai)", re.IGNORECASE),
        [
            "Nimepatana na morio wangu, akanitumia dooh yangu yote. Mambo iko fiti kabisa"
        ]
    ),
    (
        re.compile(r"(choka|uchovu|usingizi|stress|kuchoka|hard\s+day)", re.IGNORECASE),
        [
            "Pole sana chief, kazi ya mtaa haina haraka. Ebu chill kiasi unywe maji au chai moto upumzike."
        ]
    ),
    (
        re.compile(r"(kazi|hustle|usle|hasla|inaendelea|job|ofisi|shughuli|shugli)", re.IGNORECASE),
        [
            "Kazi inasonga fiti sana chief, tunang'ang'ana kusaka dooh bila kuogopa!"
        ]
    ),
    (
        re.compile(r"(luku|drip|pamba|raba|viatu|zinakutoa)", re.IGNORECASE),
        [
            "Hizo raba zimepiga luku hatari msee, unakaa chief mwenyewe!"
        ]
    ),
    (
        re.compile(r"(kibanda|chakula|lunch|lungi|chapo|madondo|chwani|choani|finje|kubuy|food|njaa|kula)", re.IGNORECASE),
        [
            "Hiyo chwani utapata chapo mbili safi na madondo kwa kibanda ya mtaa. Utashiba fiti sana!"
        ]
    ),
    (
        re.compile(r"(rongai|ngoma|nganya\s+mpya)", re.IGNORECASE),
        [
            "Panda ile nganya mpya ya Rongai iko na ngoma safi na luku ya maana! utafike kejani mbio."
        ]
    ),
    (
        re.compile(r"(mathree|nganya|nduthi|stage|gari|nisho|keja|tao|pahalike)", re.IGNORECASE),
        [
            "Panda tu nganya pale stage, shuka tao alafu unisho kwa simu nikupe rada safi ya street!"
        ]
    ),
    (
        re.compile(r"(jam\b|foleni|traffic|barabara|mombasa\s+road|waiyaki|thika\s+road)", re.IGNORECASE),
        [
            "Hiyo jam ya Nairobi ni noma sana chief! Kama unawahi mahali, nduthi ndio itakuchomoa mbio bila stress."
        ]
    ),
    (
        re.compile(r"(choka|uchovu|usingizi|stress|kuchoka|hard\s+day)", re.IGNORECASE),
        [
            "Pole sana chief, kazi ya mtaa haina haraka. Ebu chill kiasi unywe maji au chai moto upumzike."
        ]
    ),
    (
        re.compile(r"(wewe\s+ni\s+nani|jina\s+lako|who\s+are\s+you|wewe\s+nani)", re.IGNORECASE),
        [
            "Mimi ni Nairobi Bot, msee wa mtaa anayeelewa Sheng na Kiswahili ya kijanja. Niambie stori yako chief!"
        ]
    ),
    (
        re.compile(r"(asante|shukran|thanks|thank\s+you|santee)", re.IGNORECASE),
        [
            "Shukran sana chief! Tuko pamoja siku zote, karibu sana mtaani."
        ]
    ),
    (
        re.compile(r"(baadaye|lala\s+salama|goodnight|tutaonana|bye|tuonane)", re.IGNORECASE),
        [
            "Wazi chief! Tuonane baadaye, uwe na siku fiti na uendelee kupiga hustle!"
        ]
    ),
    (
        re.compile(r"(mechi|mpira|ball|arsenal|chelsea|man\s+u|liverpool|game\b)", re.IGNORECASE),
        [
            "Mechi za leo ni moto sana mtaani! Mbogi zote ziko rada na game chief."
        ]
    ),
    (
        re.compile(r"(mvua|jua|baridi|hali\s+ya\s+anga|weather)", re.IGNORECASE),
        [
            "Hali ya anga ya leo inabidi uwe rada chief! Hakikisha uko comfortable popote ulipo."
        ]
    ),
    (
        re.compile(r"(form\b|mpango|plot\b)", re.IGNORECASE),
        [
            "Form ni kuchill tu kejani na mbogi, tukipiga stori za luku na hustle. Rada yako?"
        ]
    ),
    (
        re.compile(r"(niaje|vipi|mambo|sasa|habari)", re.IGNORECASE),
        [
            "Niko fiti sana chief! Rada yako?"
        ]
    ),
    (
        re.compile(r"(rada|kunani|story|stori)", re.IGNORECASE),
        [
            "Rada iko safi kabisa chief, kila kitu iko under control."
        ]
    ),
    (
        re.compile(r"(dooh|doh|chapaa|ganji|pesa|ashu)", re.IGNORECASE),
        [
            "Dooh inasakwa kila siku chief, bidii tu ndio siri!"
        ]
    ),
    (
        re.compile(r"(ndio|sawa\s+kabisa|kweli|wazi\s+kabisa|kabisa|sure|hapo\s+sawa)", re.IGNORECASE),
        [
            "Tuko pamoja kabisa chief! Niambie zaidi vile mambo inaendelea upande wako."
        ]
    ),
]
