from pathlib import Path
import shutil
import re

# ---------- Configuration ----------
BASE_DIR        = Path.home() / "Desktop" / "zoza_transcripts"
TRANSCRIPTS_DIR = BASE_DIR / "transcripts"       # holds transcripts.txt
AUDIO_DIR       = BASE_DIR / "audio_clips"       # holds clip_XXXX.mp3 files
OUTPUT_DIR      = BASE_DIR / "mapped_data"       # new folder that will be created
# -----------------------------------


def load_transcript_blocks(transcripts_dir: Path) -> list[str]:
    """Find the transcript file (.txt or .csv) and split it into blocks
    separated by blank lines."""
    candidates = [p for p in transcripts_dir.iterdir()
                  if p.suffix.lower() in {".txt", ".csv"}]
    if not candidates:
        raise FileNotFoundError(f"No .txt or .csv file found in {transcripts_dir}")
    if len(candidates) > 1:
        print(f"Warning: multiple transcript files found, using {candidates[0].name}")

    text = candidates[0].read_text(encoding="utf-8")

    # Split on blank lines
    blocks = [b.strip() for b in re.split(r"\n\s*\n", text)]
    return [b for b in blocks if b]


def main():
    transcripts = load_transcript_blocks(TRANSCRIPTS_DIR)
    audios = sorted(AUDIO_DIR.glob("clip_*.mp3"))

    print(f"Found {len(transcripts)} transcript blocks")
    print(f"Found {len(audios)} audio clips")

    n_pairs = min(len(transcripts), len(audios))
    if len(transcripts) != len(audios):
        print(f"Note: counts differ — mapping the first {n_pairs} pairs only.")

    out_audio = OUTPUT_DIR / "audios"
    out_trans = OUTPUT_DIR / "transcripts"
    out_audio.mkdir(parents=True, exist_ok=True)
    out_trans.mkdir(parents=True, exist_ok=True)

    for i in range(n_pairs):
        idx = i + 1  # 1-based numbering for the naming convention
        audio_name = f"audio_{idx}_for_script_{idx}.mp3"
        trans_name = f"transcript_{idx}_for_audio_{idx}.txt"

        shutil.copy2(audios[i], out_audio / audio_name)
        (out_trans / trans_name).write_text(transcripts[i], encoding="utf-8")

    # Manifest tracing each pair back to the original clip file
    manifest = OUTPUT_DIR / "manifest.csv"
    with manifest.open("w", encoding="utf-8") as f:
        f.write("pair,original_clip,paired_audio,paired_transcript\n")
        for i in range(n_pairs):
            idx = i + 1
            f.write(f"{idx},{audios[i].name},"
                    f"audio_{idx}_for_script_{idx}.mp3,"
                    f"transcript_{idx}_for_audio_{idx}.txt\n")

    print(f"Done! {n_pairs} pairs written to {OUTPUT_DIR}")
    print(" - audios/       -> audio_1_for_script_1.mp3, ...")
    print(" - transcripts/  -> transcript_1_for_audio_1.txt, ...")
    print(" - manifest.csv  -> maps each pair to the original clip file")


if __name__ == "__main__":
    main()
