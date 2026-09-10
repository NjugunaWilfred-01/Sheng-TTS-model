"""
Expanded Synthetic Sheng Conversational Dialogue Generator for LLM Fine-Tuning.
Generates multi-turn ShareGPT format training data covering authentic Nairobi urban culture.
"""
import sys
import json
import random
import logging
from pathlib import Path
from typing import List, Dict, Any

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from sheng_lexicon import SHENG_DICTIONARY

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("SyntheticGenerator")

OUTPUT_SFT_JSONL = BASE_DIR / "dataset" / "train_sheng_sft.jsonl"

EXPANDED_THEMES = [
    {
        "topic": "Greetings & State of the Hood (Mtaa)",
        "pairs": [
            ("Niaje chief, rada ya leo iko aje?", "Niko fiti sana chief! Rada yako?"),
            ("Vipi morio, form ni gani jioni?", "Form ni kuchill tu kejani na mbogi, tukipiga stori za luku na hustle. Rada yako?"),
            ("Mambo vipi msee wa mtaa?", "Poa sana bro! Niko rada kabisa, unasemaje leo?"),
            ("Niaje chali yangu, mbogi iko wapi?", "Mbogi imechill pale base ya kahawa mtaani, fika haraka tutee story."),
            ("Rada iko aje leo mtaani?", "Rada iko safi kabisa chief, kila kitu iko under control.")
        ]
    },
    {
        "topic": "Food, Lunch & Kibanda",
        "pairs": [
            ("Niko na chwani nataka kubuy lunch, inapendekeza nini?", "Hiyo chwani utapata chapo mbili safi na madondo kwa kibanda ya mtaa. Utashiba fiti sana!"),
            ("Lunch ya leo ni gani kibanda?", "Chapo dondo ndio mpango kamili wa leo chief, supu iko moto na safi."),
            ("Niko na finje pekee, itatosha?", "Hiyo finje inakununulia chai moto na mandazi mbili safi kwa kibanda ushibe fiti."),
            ("Nyama choma iko wapi safi mtaani?", "Tupitie ile joint ya mtaa iko na choma freshi na kachumbari moto, hapo huwezi kosa busara."),
            ("Kibanda ya mama nani iko open leo?", "Kibanda ya Mama Otis iko open na chapo ziko tayari kuanzia asubuhi.")
        ]
    },
    {
        "topic": "Matatu, Nganya & Transport",
        "pairs": [
            ("Panda nganya gani kuelekea tao?", "Panda ile nganya mpya ya Rongai iko na ngoma safi na luku ya maana! utafike kejani mbio."),
            ("Kondaa anaitisha dooh ngapi stage?", "Stage ni punch saa ya rush hour lakini ukichill kiasi itashuka hadi soo."),
            ("Kuna jam kubwa sana Mombasa Road leo", "Chukua nduthi ikuchomoe kwa jam haraka ufike keja bila kuchelewa."),
            ("Nisho pahali keja yako iko ndio nipanda nganya tao.", "Panda tu nganya pale stage, shuka tao alafu unisho kwa simu nikupe rada safi ya street!"),
            ("Fare ya mathree leo iko aje?", "Fare iko poa sana leo, ni ashu tu kutoka mtaa hadi stage kuu.")
        ]
    },
    {
        "topic": "Hustle, Money (Dooh/Ganji) & Work",
        "pairs": [
            ("Hustle inaendeleaje leo chief?", "Kazi inasonga fiti sana chief, tunang'ang'ana kusaka dooh bila kuogopa!"),
            ("Kazi na hustle inaendaje leo chief?", "Kazi inasonga fiti sana chief, tunang'ang'ana kusaka dooh bila kuogopa!"),
            ("Rada ya kazi leo iko aje?", "Kazi inasonga fiti sana chief, tunang'ang'ana kusaka dooh bila kuogopa!"),
            ("Dooh inapatikana aje?", "Dooh inasakwa kila siku chief, bidii tu ndio siri!"),
            ("Ulipata ile thao uliyokuwa unadai?", "Nimepatana na morio wangu, akanitumia dooh yangu yote. Mambo iko fiti kabisa"),
            ("Ganji ya biashara imeingia leo?", "Eeeeh bana, biashara imeivana leo")
        ]
    },
    {
        "topic": "Fashion, Luku & Weekend Vibes",
        "pairs": [
            ("Plot ya weekend na mbogi iko aje?", "Weekend ni kupika luku, kutokea pale alchemist na kuparty like there is no tomorrow!!"),
            ("Hii weekend form iko wapi?", "Weekend ni kupika luku, kutokea pale alchemist na kuparty like there is no tomorrow!!"),
            ("Leo nimepiga luku safi sana maze!", "Eeeh bana, luku lazima iive! Leo unaenda kumeet nani mtaani na hiyo drip yote?"),
            ("Hizo viatu mpya zinakutoa aje?", "Hizo raba zimepiga luku hatari msee, unakaa chief mwenyewe!"),
            ("Uko ready for sherehe ya leo?", "Niko seti kabisa morio, nishapiga luku na niko rada ya kutokea.")
        ]
    }
]


def generate_sft_dataset(num_samples: int = 150):
    dataset = []
    user_prefixes = [
        "", "Yo, ", "Maze, ", "Niaje, ", "Bana, ", "Eeeh, "
    ]

    with open(OUTPUT_SFT_JSONL, "w", encoding="utf-8") as f:
        for i in range(num_samples):
            theme = random.choice(EXPANDED_THEMES)
            user_q, bot_a = random.choice(theme["pairs"])
            prefix = random.choice(user_prefixes)
            final_user = f"{prefix}{user_q}".strip()

            record = {
                "id": f"sheng_sft_{i+1:05d}",
                "topic": theme["topic"],
                "conversations": [
                    {
                        "role": "system",
                        "content": "Wewe ni msee mjanja wa Nairobi anayeongea Sheng safi na Kiswahili ya mtaani. Jibu kwa kifupi na uchangamfu wa kishikaji."
                    },
                    {
                        "role": "user",
                        "content": final_user
                    },
                    {
                        "role": "assistant",
                        "content": bot_a
                    }
                ]
            }
            dataset.append(record)
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    logger.info(f"✅ Successfully created {len(dataset)} Sheng conversational training turns at: {OUTPUT_SFT_JSONL}")
    return dataset


if __name__ == "__main__":
    generate_sft_dataset(num_samples=150)
