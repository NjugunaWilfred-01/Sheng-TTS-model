# Live Demo Script — Nairobi Bot

**CRITICAL INSTRUCTION FOR THE SPEAKER:** 
Keep your audio inputs to **3 to 5 seconds max**. The ASR is running on a CPU. Short clips will process in < 5 seconds. Long clips will stall the demo.

---

### SCENARIO 1 — Greeting
* **Mic:** "Niaje chief, form ni gani leo mtaani?"
* **Expected Bot Vibe:** Relaxed, casual, ends with a follow-up question (e.g., "Rada yako?").
* **Expected Latency:** < 5s

### SCENARIO 2 — Hustle
* **Mic:** "Hustle inaendeleaje leo chief?"
* **Expected Bot Vibe:** Energetic, mentions looking for money (dooh/kazi).
* **Expected Latency:** < 5s

### SCENARIO 3 — Kibanda (Food)
* **Mic:** "Niko na chwani nataka kubuy lunch, unapendekeza nini?"
* **Expected Bot Vibe:** Helpful, street-smart, mentions chapati or madondo.
* **Expected Latency:** < 5s

### SCENARIO 4 — Nganya (Transport)
* **Mic:** "Nisho pahali keja yako iko ndio nipande nganya tao."
* **Expected Bot Vibe:** Directive, gives instructions mentioning stage/tao.
* **Expected Latency:** < 5s

### SCENARIO 5 — Weekend
* **Mic:** "Hii weekend form iko wapi?"
* **Expected Bot Vibe:** Excited, mentions sherehe, luku, or kuparty.
* **Expected Latency:** < 5s

### SCENARIO 6 — Drip (Fashion)
* **Mic:** "Hizo raba mpya zimepiga luku hatari msee."
* **Expected Bot Vibe:** Confident, hyping up the look, acting like the chief.
* **Expected Latency:** < 5s

---

## 🚨 EMERGENCY FALLBACK PROTOCOL
If the Wi-Fi drops, the API fails, or the CPU stalls during the presentation, **do not panic**.

Pre-generated bot replies for all 6 scenarios live in **`assets/fallback/`** — committed
to the repo, so they survive a reboot and a fresh clone. (They used to be documented as
living in `/tmp/fallback/`, which is wiped on restart and is not a real path on Windows.)

```
assets/fallback/1_greeting.mp3     assets/fallback/4_drip.mp3
assets/fallback/2_kibanda.mp3      assets/fallback/5_hustle.mp3
assets/fallback/3_nganya.mp3       assets/fallback/6_weekend.mp3
```

Open the folder and play the matching file. Regenerate any time with:

```bash
python scripts/generate_demo_audio.py --force
```

### Degradation ladder — try in this order

1. **LLM API fails** → nothing to do, `ShengLLMEngine` drops to the offline heuristic by itself.
2. **ASR returns garbage** → switch the backend dropdown to heuristic and drive it from `python cli.py`.
3. **Pipeline throws** → play `assets/fallback/*.mp3`.
4. **Gradio won't load** → `python cli.py --demo`.

### Before you start, confirm
- [ ] `python scripts/prefetch_models.py` has run (no 1.6GB download mid-demo)
- [ ] `assets/fallback/` has all 6 files and they play
- [ ] `OPENAI_API_KEY` is exported in the shell you launch `app.py` from
- [ ] `python test_pipeline.py` passes
