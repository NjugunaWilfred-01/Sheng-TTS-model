"""
make_subset.py - create a mapped_data-style folder containing only specific ids.
Usage:
  python make_subset.py --src zoza_transcripts/mapped_data --dst heldout_eval --ids 3,7,12,...
"""
import argparse, re, shutil
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--src", default="zoza_transcripts/mapped_data")
ap.add_argument("--dst", required=True)
ap.add_argument("--ids", required=True, help="comma-separated ids, e.g. 3,7,12")
a = ap.parse_args()

ids = {int(x) for x in a.ids.split(",")}
src, dst = Path(a.src), Path(a.dst)
for i in ids:
    au = src / "audios" / f"audio_{i}_for_script_{i}.mp3"
    tr = src / "transcripts" / f"transcript_{i}_for_audio_{i}.txt"
    if au.exists() and tr.exists():
        (dst / "audios").mkdir(parents=True, exist_ok=True)
        (dst / "transcripts").mkdir(parents=True, exist_ok=True)
        shutil.copy2(au, dst / "audios" / au.name)
        shutil.copy2(tr, dst / "transcripts" / tr.name)
print(f"Copied {len(ids)} pairs to {dst}")
