"""
Synthetic Sheng dialogue generator for LLM fine-tuning (ShareGPT / ChatML format).

WHY THIS IS SLOT-BASED
----------------------
The previous version held 26 hand-written (user, bot) pairs and sampled them WITH
REPLACEMENT 150 times. The adapter therefore saw the same handful of answers dozens
of times each and memorised them verbatim -- which is exactly the "canned" feeling:
give the model a prompt it has not seen and it snaps to the nearest memorised reply.
Three of those 26 pairs even shared an identical bot answer.

The fix is composition, not more hand-writing. Each theme carries user and bot
skeletons with {slot} placeholders; filling the slots from vocabulary lists yields
thousands of distinct surface forms over the same intents, which forces the model to
generalise the PATTERN rather than memorise the STRING.

Usage:
    python data_engine/synthetic_dialogue_generator.py
    python data_engine/synthetic_dialogue_generator.py --num_samples 4000
"""
import argparse
import json
import logging
import random
import sys
from pathlib import Path
from typing import Dict, List

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("SyntheticGenerator")

OUTPUT_SFT_JSONL = BASE_DIR / "dataset" / "train_sheng_sft.jsonl"

SYSTEM_PROMPT = (
    "Wewe ni msee mjanja wa Nairobi anayeongea Sheng safi na Kiswahili ya mtaani. "
    "Jibu kwa kifupi na uchangamfu wa kishikaji."
)

# Interchangeable vocabulary. Every skeleton draws from these, so one skeleton
# expands into hundreds of distinct sentences.
SLOTS: Dict[str, List[str]] = {
    "greeting": ["Niaje", "Sasa", "Mambo", "Vipi", "Uko aje", "Rada iko aje", "Kunani"],
    "address": ["chief", "msee", "morio", "maze", "chali", "bro", "bazenga", "beste"],
    "topic": ["form ya leo", "rada ya mtaa", "hustle", "mbogi", "keja", "weekend",
              "plot ya leo", "stori za mtaa"],
    # Singular-agreeing subset. Swahili concord: "stori za mtaa ZIKO", not "IKO", so
    # any skeleton of the form "{...} iko ..." must draw from here, not from topic.
    "sing_topic": ["form ya leo", "rada ya mtaa", "hustle", "keja", "weekend",
                   "plot ya leo"],
    "action": ["kuchill", "kupiga luku", "kusaka dooh", "kutea", "kumeet", "kuseti mambo"],
    "context": ["mtaani", "kejani", "tao", "kibandani", "stage", "base", "kwa mtaa"],
    "good": ["fiti", "safi", "poa", "wazi", "noma", "moto"],
    "money": ["dooh", "ganji", "chapaa", "pesa"],
    "food": ["chapo na madondo", "smokie pap", "chai na mandazi", "nyama choma",
             "chapo dondo", "githeri ya mtaa"],
    "coin": ["chwani", "finje", "soo", "ashu", "thao", "punch"],
    "ride": ["nganya", "mathree", "nduthi", "gari ya mtaa"],
    "time": ["leo", "jioni", "asubuhi", "weekend", "kesho", "saa hii"],
}

USER_SKELETONS: Dict[str, List[str]] = {
    "Greetings & State of the Hood (Mtaa)": [
        "{greeting} {address}, rada ya {time} iko aje?",
        "{greeting} {address}, {topic} ni gani {time}?",
        "{greeting} {address}, mambo iko aje {context}?",
        "{greeting} {address}, mbogi iko wapi {time}?",
        "Kunani {context} {time} {address}?",
        "{greeting} {address}, kuna nini {context}?",
        "Rada iko aje upande wako {address}?",
        "{greeting} {address}, uko {context} ama?",
    ],
    "Food, Lunch & Kibanda": [
        "Niko na {coin} nataka kubuy lunch, unapendekeza nini?",
        "Lunch ya {time} ni gani kibandani?",
        "Niko na {coin} pekee, itatosha {address}?",
        "{food} iko wapi {good} {context}?",
        "Njaa imenimaliza {address}, nikule nini?",
        "Kibanda gani iko open {time} {context}?",
        "Nataka {food}, ni {coin} ngapi?",
        "Chakula {good} kiko wapi {context} {address}?",
    ],
    "Matatu, Nganya & Transport": [
        "Panda {ride} gani kuelekea tao {address}?",
        "Kondaa anaitisha {money} ngapi {context}?",
        "Kuna jam {context} {time}, nifanye aje?",
        "Nisho pahali keja yako iko ndio nipande {ride} tao.",
        "Fare ya {ride} {time} iko aje?",
        "{ride} ya {context} inaondoka saa ngapi?",
        "Nafika {context} aje kutoka hapa {address}?",
        "Ni {ride} gani iko {good} kwa hii route?",
    ],
    "Hustle, Money (Dooh/Ganji) & Work": [
        "Hustle inaendeleaje {time} {address}?",
        "Rada ya kazi {time} iko aje?",
        "{money} inapatikana aje siku hizi {address}?",
        "Ulipata ile {coin} uliyokuwa unadai?",
        "Biashara imeenda aje {time} {address}?",
        "Kazi iko aje {context} {time}?",
        "Nimechoka na hii hustle {address}, nifanye aje?",
        "{money} ya {time} imeingia ama bado?",
    ],
    "Fashion, Luku & Weekend Vibes": [
        "Plot ya weekend na mbogi iko aje {address}?",
        "Hii weekend {topic} iko wapi?",
        "{time} nimepiga luku {good} sana {address}!",
        "Hizo viatu mpya zinakutoa aje?",
        "Uko ready for sherehe ya {time} {address}?",
        "Tunatokea wapi {time} {address}?",
        "Luku yangu ya {time} iko aje {address}?",
        "Sherehe ya {time} iko {context} ama wapi?",
    ],
}

BOT_SKELETONS: Dict[str, List[str]] = {
    "Greetings & State of the Hood (Mtaa)": [
        "Niko {good} sana {address}! Rada yako?",
        "Poa kabisa {address}, tuko tu {context}. Wewe uko aje?",
        "Mambo ni {good} {address}! Kuna nini upande wako?",
        "Rada iko {good} kabisa {address}, kila kitu iko under control.",
        "Niko wazi {address}, hustle inaendelea. Wewe je?",
        "Mbogi imechill pale base {context}, fika haraka tutee stori.",
        "{sing_topic} iko {good} {address}, hakuna stress. Unasemaje?",
        "Tuko tu {context} tuki{action}. Wewe uko wapi?",
    ],
    "Food, Lunch & Kibanda": [
        "Hiyo {coin} utapata {food} kwa kibanda ya mtaa. Utashiba {good} sana!",
        "Enda kibanda ya {context} {address}, {food} ni mob kwa bei poa.",
        "{food} ndio mpango kamili wa {time} {address}, supu iko moto.",
        "Na hiyo {coin} utapata {food}. Njaa itaisha kabisa!",
        "Tupitie ile joint ya {context}, {food} iko freshi {address}.",
        "Kibanda ya mama {context} iko open, {food} iko tayari.",
        "Hiyo {coin} inatosha {food} {address}. Usiwe na stress.",
        "Chukua {food} {address}, hiyo ndio inashika {time}.",
    ],
    "Matatu, Nganya & Transport": [
        "Panda ile {ride} mpya iko na ngoma {good}! Utafike kejani mbio.",
        "Stage ni {coin} saa ya rush lakini ukichill kiasi itashuka.",
        "Chukua {ride} ikuchomoe kwa jam haraka ufike keja bila kuchelewa.",
        "Panda tu {ride} pale stage, shuka tao alafu unisho kwa simu.",
        "Fare iko poa sana {time}, ni {coin} tu kutoka {context}.",
        "{ride} ndio itakuchomoa mbio {address}, hiyo njia ina jam.",
        "Toka mapema {address}, {ride} za {time} zinajaa haraka.",
        "Shuka {context} alafu ubadilishe {ride}, utafika {good}.",
    ],
    "Hustle, Money (Dooh/Ganji) & Work": [
        "Kazi inasonga {good} sana {address}, tunang'ang'ana kusaka {money} bila kuogopa!",
        "Hustle inaendelea tu {address}, kila siku ni kuamka na kubamba.",
        "{money} inasakwa kila siku {address}, bidii tu ndio siri!",
        "Nimepatana na morio wangu, akanitumia {money} yangu yote. Mambo iko {good}.",
        "Biashara imeivana {time} {address}! Wateja wameingia kama mvua.",
        "Kazi iko {good} {address}, {money} inaingia pole pole lakini tuko rada.",
        "Pole {address}, hiyo hustle ni ngumu. Chill kiasi alafu urudi {time}.",
        "{money} ni ya kuhustle {address}, haiji kwa kulala. Tuko pamoja?",
    ],
    "Fashion, Luku & Weekend Vibes": [
        "Weekend ni kupika luku, kutokea {context} na kuparty like there is no tomorrow!",
        "Hii weekend tunatokea {context} na mbogi, tukipiga luku hadi asubuhi!",
        "Luku lazima iive {address}! Unaenda kumeet nani na hiyo drip yote?",
        "Hizo raba zimepiga luku hatari {address}, unakaa chief mwenyewe!",
        "Niko seti kabisa {address}, nishapiga luku na niko rada ya kutokea.",
        "{topic} iko {good} {address}, tuko na plot ya {action} {context}.",
        "Luku yako iko juu {address}, umeiva kabisa {time}!",
        "Tunatokea {context} {time} {address}. Uko ndani ama uko nje?",
    ],
}

USER_PREFIXES = ["", "", "", "Yo, ", "Maze, ", "Bana, ", "Eeeh, ", "Aki "]


def _fill(skeleton: str, rng: random.Random) -> str:
    """Fill every {slot} in a skeleton from SLOTS, then tidy capitalisation."""
    text = skeleton.format(**{k: rng.choice(v) for k, v in SLOTS.items()})
    return text[0].upper() + text[1:] if text else text


def generate_sft_dataset(num_samples: int = 2500, seed: int = 7) -> List[dict]:
    rng = random.Random(seed)
    themes = sorted(USER_SKELETONS)

    # Enumerate the full skeleton cross-product per theme, then fill each pairing a
    # few times with different slot draws. Deduplicated on the exact (user, bot)
    # string so no record is ever repeated -- repetition is what caused memorisation.
    combos = []
    for theme in themes:
        for user_skel in USER_SKELETONS[theme]:
            for bot_skel in BOT_SKELETONS[theme]:
                combos.append((theme, user_skel, bot_skel))
    rng.shuffle(combos)

    seen = set()
    records = []
    attempts_per_combo = max(1, (num_samples // max(len(combos), 1)) + 2)

    for theme, user_skel, bot_skel in combos:
        for _ in range(attempts_per_combo):
            if len(records) >= num_samples:
                break
            user = f"{rng.choice(USER_PREFIXES)}{_fill(user_skel, rng)}".strip()
            bot = _fill(bot_skel, rng)
            key = (user, bot)
            if key in seen:
                continue
            seen.add(key)
            records.append({
                "id": f"sheng_sft_{len(records) + 1:05d}",
                "topic": theme,
                "conversations": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user},
                    {"role": "assistant", "content": bot},
                ],
            })
        if len(records) >= num_samples:
            break

    OUTPUT_SFT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_SFT_JSONL, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    unique_users = len({r["conversations"][1]["content"] for r in records})
    unique_bots = len({r["conversations"][2]["content"] for r in records})
    logger.info(f"Wrote {len(records)} turns to {OUTPUT_SFT_JSONL}")
    logger.info(f"  unique user turns: {unique_users}  unique bot turns: {unique_bots}")
    logger.info(f"  skeleton combinations available: {len(combos)}")
    if len(records) < num_samples:
        logger.warning(
            f"Only {len(records)} unique turns available from {len(combos)} skeleton "
            f"combinations. Add skeletons or slot values to reach {num_samples}."
        )
    return records


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--num_samples", type=int, default=2500)
    ap.add_argument("--seed", type=int, default=7)
    generate_sft_dataset(**vars(ap.parse_args()))
