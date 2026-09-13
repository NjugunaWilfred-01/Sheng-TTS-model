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

Pre-generated audio files for all 6 of these exact scenarios are saved on the machine at `/tmp/fallback/`. 
If the live pipeline breaks, simply open your file explorer, navigate to `/tmp/fallback/`, and manually play the corresponding `.mp3` file for the audience.
