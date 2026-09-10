"""
Terminal Interactive CLI for the Swahili & Sheng Speech-to-Speech (S2S) Pipeline.
Enables 100% terminal-based interaction, mic recording, file processing, and demo tracks.
"""
import os
import sys
import time
import argparse
import subprocess
from pathlib import Path

from pipeline import SpeechToSpeechPipeline
from tts_engine import ShengTTSEngine
from config import VOICE_OPTIONS, DEFAULT_VOICE, AUDIO_TEMP_DIR
from sheng_lexicon import SHENG_DICTIONARY

# ANSI Color Codes for Terminal Styling
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BOLD = "\033[1m"
RESET = "\033[0m"


def play_audio_terminal(audio_path: str):
    """Attempts to play audio directly from the terminal using available system players."""
    players = ["ffplay", "mpv", "aplay", "paplay", "mpg123", "mplayer"]
    for player in players:
        try:
            if player == "ffplay":
                cmd = ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", audio_path]
            else:
                cmd = [player, audio_path]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            return True
        except (FileNotFoundError, subprocess.CalledProcessError):
            continue
    return False


def run_text_prompt(text: str, voice: str = DEFAULT_VOICE):
    """Processes a text prompt directly through LLM and TTS."""
    print(f"\n{BOLD}{CYAN}=== Terminal S2S: Text Input ==={RESET}")
    print(f"{YELLOW}User Input:{RESET} {text}")
    
    tts = ShengTTSEngine()
    from llm_engine import ShengLLMEngine
    llm = ShengLLMEngine()

    # Step 1: LLM
    bot_reply, llm_meta = llm.generate_response(text)
    print(f"{GREEN}Bot Reply ({llm_meta['backend']}):{RESET} {BOLD}{bot_reply}{RESET}")
    print(f"⏱️  LLM Latency: {llm_meta['llm_time_ms']} ms")

    # Step 2: TTS
    out_file = str(AUDIO_TEMP_DIR / f"cli_resp_{int(time.time()*1000)}.mp3")
    audio_path, tts_meta = tts.synthesize(bot_reply, output_path=out_file, voice=voice)
    print(f"🔊 Spoken Audio Saved: {audio_path}")
    print(f"⏱️  TTS Latency: {tts_meta['tts_time_ms']} ms")

    # Audio Playback
    print(f"🎵 Playing spoken response in terminal...")
    if not play_audio_terminal(audio_path):
        print(f"{YELLOW}Note: No audio player (ffplay/mpv/aplay) found. File saved at {audio_path}{RESET}")


def run_audio_file(audio_path: str, voice: str = DEFAULT_VOICE):
    """Processes an audio file end-to-end."""
    print(f"\n{BOLD}{CYAN}=== Terminal S2S: Audio File Processing ==={RESET}")
    print(f"📁 Input Audio: {audio_path}")

    pipeline = SpeechToSpeechPipeline()
    result = pipeline.process_audio(audio_path, voice=voice)

    if not result.get("success"):
        print(f"{RED}Error:{RESET} {result.get('error')}")
        return

    print(f"\n{BOLD}Results:{RESET}")
    print(f"🎙️  {YELLOW}Raw Whisper ASR:{RESET}    {result['user_raw_transcription']}")
    print(f"✨ {YELLOW}Normalized Sheng:{RESET}   {result['user_normalized_sheng']}")
    print(f"🧠 {GREEN}Bot Sheng Reply:{RESET}    {BOLD}{result['bot_response_text']}{RESET}")
    print(f"🔊 {CYAN}Output Audio:{RESET}       {result['output_audio_path']}")
    
    lat = result["latencies"]
    print(f"\n{BOLD}⚡ Latency Breakdown:{RESET}")
    print(f"  • ASR Latency:            {lat['asr_ms']} ms")
    print(f"  • LLM Latency:            {lat['llm_ms']} ms")
    print(f"  • TTS Latency:            {lat['tts_ms']} ms")
    print(f"  • {BOLD}Total Turnaround:{RESET}        {BOLD}{GREEN}{lat['total_glass_to_glass_ms']} ms{RESET}")

    # Playback
    print(f"\n🎵 Playing response audio in terminal...")
    play_audio_terminal(result["output_audio_path"])


def run_demo_tracks():
    """Runs the 3 Saturday Demo scenarios sequentially in the terminal."""
    print(f"\n{BOLD}{CYAN}===================================================={RESET}")
    print(f"{BOLD}{GREEN}🇰🇪 SWAHILI & SHENG S2S: SATURDAY DEMO TRACKS (CLI){RESET}")
    print(f"{BOLD}{CYAN}===================================================={RESET}\n")

    tracks = [
        ("Track 1: Casual Nairobi Greeting", "Niaje chief, form ni gani leo mtaani?", "demo_1_greeting.mp3"),
        ("Track 2: Work & Hustle", "Hustle inaendeleaje leo chief?", "demo_5_work.mp3"),
        ("Track 3: Food & Lunch at a Kibanda", "Niko na chwani nataka kubuy lunch, unapendekeza nini?", "demo_2_food.mp3"),
        ("Track 4: Matatu & Transport Direction", "Nisho pahali keja yako iko ndio nipanda nganya tao.", "demo_3_transport.mp3"),
        ("Track 5: Weekend Vibes", "Hii weekend form iko wapi?", "demo_6_weekend.mp3"),
        ("Track 6: Drip & Fashion", "Hizo viatu mpya zinakutoa aje?", "demo_4_drip.mp3")
    ]

    pipeline = SpeechToSpeechPipeline()
    from config import BASE_DIR

    for idx, (title, prompt_text, filename) in enumerate(tracks, 1):
        print(f"{BOLD}{YELLOW}[Scenario {idx}/6] {title}{RESET}")
        sample_path = BASE_DIR / "assets" / "demo_samples" / filename

        if not sample_path.exists():
            tts = ShengTTSEngine()
            tts.synthesize(prompt_text, output_path=str(sample_path), voice="sw-KE-RafikiNeural")

        print(f"🗣️  Input Audio: {sample_path.name} (\"{prompt_text}\")")

        # Process end-to-end
        result = pipeline.process_audio(str(sample_path), voice="sw-KE-ZuriNeural")

        if not result.get("success"):
            print(f"{RED}❌ Error: {result.get('error')}{RESET}\n")
            continue

        print(f"🎙️  ASR Recognized: \"{result['user_normalized_sheng']}\"")
        print(f"🤖 Bot Spoken Reply: \"{BOLD}{result['bot_response_text']}{RESET}\"")
        lat = result["latencies"]
        print(f"⚡ Glass-to-Glass Latency: {GREEN}{lat['total_glass_to_glass_ms']} ms{RESET} (ASR: {lat['asr_ms']}ms | LLM: {lat['llm_ms']}ms | TTS: {lat['tts_ms']}ms)")
        print(f"🔊 Output: {result['output_audio_path']}\n")
        play_audio_terminal(result["output_audio_path"])
        time.sleep(1)

    print(f"{BOLD}{GREEN}🎉 All 6 Demo Tracks Executed Successfully!{RESET}\n")


def interactive_terminal_repl():
    """Interactive command-line conversational REPL."""
    print(f"\n{BOLD}{CYAN}===================================================={RESET}")
    print(f"{BOLD}{GREEN}🇰🇪 Nairobi Sheng S2S Terminal Interactive REPL{RESET}")
    print(f"{CYAN}Type your Sheng phrase and get spoken audio responses!{RESET}")
    print(f"{YELLOW}Commands: 'exit' to quit, 'dict' for Sheng slang cheat sheet{RESET}")
    print(f"{BOLD}{CYAN}===================================================={RESET}\n")

    tts = ShengTTSEngine()
    from llm_engine import ShengLLMEngine
    llm = ShengLLMEngine()

    while True:
        try:
            user_text = input(f"{BOLD}{GREEN}You (Sheng) > {RESET}").strip()
            if not user_text:
                continue
            if user_text.lower() in ["exit", "quit", "q"]:
                print("Tuonane baadaye chief! (Goodbye!)")
                break
            if user_text.lower() == "dict":
                print(f"\n{BOLD}📚 Top Sheng Terms:{RESET}")
                for k, v in list(SHENG_DICTIONARY.items())[:20]:
                    print(f"  • {BOLD}{k}{RESET}: {v}")
                print()
                continue

            bot_reply, llm_meta = llm.generate_response(user_text)
            print(f"{BOLD}{CYAN}Bot > {RESET}{BOLD}{bot_reply}{RESET}")

            # Synthesize audio
            out_file = str(AUDIO_TEMP_DIR / f"repl_{int(time.time()*1000)}.mp3")
            audio_path, tts_meta = tts.synthesize(bot_reply, output_path=out_file)
            print(f"⚡ [LLM: {llm_meta['llm_time_ms']}ms | TTS: {tts_meta['tts_time_ms']}ms | Audio: {audio_path}]")
            play_audio_terminal(audio_path)
            print()

        except (KeyboardInterrupt, EOFError):
            print("\nTuonane baadaye!")
            break


def record_microphone_terminal(duration: int = 5, voice: str = DEFAULT_VOICE):
    """Records audio directly from the terminal microphone and processes it."""
    out_file = str(AUDIO_TEMP_DIR / f"mic_input_{int(time.time()*1000)}.wav")
    print(f"\n{BOLD}{CYAN}=== 🎤 Terminal Microphone Recording ==={RESET}")
    print(f"{YELLOW}Recording for {duration} seconds...{RESET}")
    print(f"{BOLD}{GREEN}>>> SPEAK NOW (Ongea Kiswahili / Sheng)! <<<{RESET}")

    # Try recording via ffmpeg or arecord
    rec_cmd = ["ffmpeg", "-y", "-f", "pulse", "-i", "default", "-t", str(duration), "-ar", "16000", "-ac", "1", out_file]
    
    try:
        subprocess.run(rec_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    except Exception:
        try:
            # Fallback to ALSA
            rec_cmd = ["arecord", "-d", str(duration), "-f", "cd", "-r", "16000", "-c", "1", out_file]
            subprocess.run(rec_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        except Exception as e:
            print(f"{RED}Could not record from mic: {e}{RESET}")
            print(f"{YELLOW}Tip: You can also record a .wav/.mp3 file and pass it with: python cli.py --file your_audio.wav{RESET}")
            return

    print(f"{GREEN}✓ Recording finished! Processing through S2S pipeline...{RESET}")
    run_audio_file(out_file, voice=voice)


def main():
    parser = argparse.ArgumentParser(description="Terminal CLI for Swahili & Sheng S2S Pipeline")
    parser.add_argument("--demo", action="store_true", help="Run the 3 Saturday Demo scenarios end-to-end")
    parser.add_argument("--mic", action="store_true", help="Record directly from your microphone (5 seconds)")
    parser.add_argument("--record", type=int, default=0, help="Record from microphone for specified seconds (e.g. --record 6)")
    parser.add_argument("--text", type=str, help="Process a text prompt and output spoken audio")
    parser.add_argument("--file", type=str, help="Process an audio file through the S2S pipeline")
    parser.add_argument("--voice", type=str, default=DEFAULT_VOICE, choices=list(VOICE_OPTIONS.values()), help="TTS Voice ID")

    args = parser.parse_args()

    if args.mic or args.record > 0:
        duration = args.record if args.record > 0 else 5
        record_microphone_terminal(duration=duration, voice=args.voice)
    elif args.demo:
        run_demo_tracks()
    elif args.text:
        run_text_prompt(args.text, voice=args.voice)
    elif args.file:
        run_audio_file(args.file, voice=args.voice)
    else:
        interactive_terminal_repl()


if __name__ == "__main__":
    main()
