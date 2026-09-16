"""
review_transcripts.py - Local web tool for hand-verifying Sheng transcripts.

This exists because the single thing holding ASR quality back is that NO transcript
in this repo has been corrected by a human. `dataset/validation_manifest.jsonl` marks
349 clips CLEAN_VALID, but clean_sheng_text is byte-identical to normalized_sheng for
all 349 -- it is regex output. Fine-tuning on it teaches Whisper its own mistakes.

The tool loads each clip, plays it, shows the machine guess, and lets you correct it.
Corrections are written to dataset/verified_transcripts.jsonl as you go (one JSON
object per clip, saved on every edit -- close the tab whenever you like, nothing is
lost). train_whisper_lora.py already prefers a `verified_text` field over everything
else, so verified clips feed straight into training with no further plumbing.

Usage:
    python scripts/review_transcripts.py
    python scripts/review_transcripts.py --port 7870 --manifest dataset/review/review.jsonl

Keyboard: Space play/pause · Ctrl+Enter save and advance · Ctrl+K mark unusable
"""
import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs, unquote

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST = ROOT / "dataset" / "validation_manifest.jsonl"
OUTPUT = ROOT / "dataset" / "verified_transcripts.jsonl"


def load_clips(manifest_path: Path):
    """
    Read the manifest and resolve each clip to a file that exists on THIS machine.

    The manifests carry absolute Linux paths from the box they were built on
    (/home/ray/...), so `relative_path` is the only field that resolves anywhere.
    Fall back to matching the basename under dataset/segmented when it is missing.
    """
    clips = []
    with open(manifest_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            path = None
            rel = row.get("relative_path")
            if rel and (ROOT / rel).exists():
                path = (ROOT / rel)
            else:
                for key in ("audio_filepath", "clean_audio_path", "absolute_path"):
                    name = row.get(key)
                    if name:
                        hits = list((ROOT / "dataset" / "segmented").rglob(Path(name).name))
                        if hits:
                            path = hits[0]
                            break
            if not path:
                continue
            clips.append({
                "id": row.get("id") or path.stem,
                "path": str(path.relative_to(ROOT)).replace("\\", "/"),
                "duration": round(float(row.get("duration", 0) or 0), 1),
                "guess": (row.get("clean_sheng_text") or row.get("normalized_text")
                          or row.get("raw_transcription") or "").strip(),
            })
    return clips


def load_done():
    done = {}
    if OUTPUT.exists():
        with open(OUTPUT, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    done[row["id"]] = row
    return done


def save_all(done):
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUTPUT.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for row in done.values():
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(OUTPUT)


PAGE = r"""<!doctype html><html><head><meta charset="utf-8">
<title>Sheng Transcript Review</title><style>
:root{--bg:#0f1413;--surf:#18201e;--line:#2a3532;--ink:#e7edea;--dim:#8fa09b;
--acc:#3fb5ae;--ok:#6fbf8e;--warn:#d9a55e}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 ui-sans-serif,system-ui,sans-serif}
.wrap{max-width:860px;margin:0 auto;padding:28px 20px 80px}
h1{font-size:19px;margin:0 0 4px;letter-spacing:-.01em}
.sub{color:var(--dim);font-size:13px;margin:0 0 20px}
.bar{height:5px;background:var(--line);border-radius:3px;overflow:hidden;margin-bottom:22px}
.bar i{display:block;height:100%;background:var(--acc);transition:width .2s}
.card{background:var(--surf);border:1px solid var(--line);border-radius:8px;padding:20px;margin-bottom:14px}
.meta{display:flex;gap:14px;font:11px ui-monospace,monospace;color:var(--dim);
letter-spacing:.06em;text-transform:uppercase;margin-bottom:14px;flex-wrap:wrap}
audio{width:100%;margin-bottom:16px}
label{display:block;font:11px ui-monospace,monospace;letter-spacing:.08em;
text-transform:uppercase;color:var(--dim);margin-bottom:6px}
.guess{background:#131a19;border:1px solid var(--line);border-radius:5px;padding:11px 13px;
color:var(--dim);font-size:14px;margin-bottom:16px;white-space:pre-wrap;min-height:20px}
textarea{width:100%;min-height:86px;background:#131a19;color:var(--ink);
border:1px solid var(--acc);border-radius:5px;padding:11px 13px;font:15px inherit;resize:vertical}
textarea:focus{outline:2px solid var(--acc);outline-offset:1px}
.row{display:flex;gap:9px;margin-top:14px;flex-wrap:wrap}
button{background:var(--acc);color:#05100f;border:0;border-radius:5px;padding:9px 16px;
font-weight:650;font-size:14px;cursor:pointer}
button:hover{filter:brightness(1.12)}
button.ghost{background:transparent;color:var(--dim);border:1px solid var(--line)}
button.bad{background:transparent;color:var(--warn);border:1px solid var(--warn)}
.hint{color:var(--dim);font-size:12px;margin-top:14px}
kbd{background:var(--line);border-radius:3px;padding:1px 5px;font:11px ui-monospace,monospace}
.done{color:var(--ok)}
</style></head><body><div class="wrap">
<h1>Sheng Transcript Review</h1>
<p class="sub">Correct the machine guess to what you actually hear. Saved automatically to
<code>dataset/verified_transcripts.jsonl</code>.</p>
<div class="bar"><i id="bar"></i></div>
<div id="app"></div>
<p class="hint"><kbd>Space</kbd> play/pause · <kbd>Ctrl</kbd>+<kbd>Enter</kbd> save &amp; next ·
<kbd>Ctrl</kbd>+<kbd>K</kbd> unusable clip · edits save as you type</p>
</div><script>
let clips=[],done={},i=0;
const app=document.getElementById('app'),bar=document.getElementById('bar');
async function boot(){
  const r=await fetch('/api/clips');const d=await r.json();
  clips=d.clips;done=d.done;
  i=clips.findIndex(c=>!done[c.id]); if(i<0)i=0;
  render();
}
function render(){
  const c=clips[i]; if(!c){app.innerHTML='<div class="card">No clips resolved on this machine.</div>';return;}
  const prev=done[c.id]||{};
  const n=Object.values(done).filter(x=>x.verified_text||x.unusable).length;
  bar.style.width=(100*n/clips.length)+'%';
  app.innerHTML=`<div class="card">
    <div class="meta"><span>${i+1} / ${clips.length}</span><span>${c.id}</span>
      <span>${c.duration}s</span><span class="done">${n} verified</span>
      ${prev.unusable?'<span style="color:var(--warn)">MARKED UNUSABLE</span>':''}</div>
    <audio id="au" src="/audio/${encodeURI(c.path)}" controls autoplay></audio>
    <label>Machine guess</label><div class="guess">${esc(c.guess)||'(empty)'}</div>
    <label>What you actually hear</label>
    <textarea id="tx" placeholder="Type the correct transcript...">${esc(prev.verified_text||c.guess)}</textarea>
    <div class="row">
      <button onclick="save(1)">Save &amp; next</button>
      <button class="ghost" onclick="go(-1)">Back</button>
      <button class="ghost" onclick="go(1)">Skip</button>
      <button class="bad" onclick="unusable()">Unusable</button>
    </div></div>`;
  const tx=document.getElementById('tx'); tx.focus();
  // Autosave on pause so nothing is lost if the tab closes mid-clip.
  let t; tx.oninput=()=>{clearTimeout(t);t=setTimeout(()=>save(0),600);};
}
function esc(s){return (s||'').replace(/[&<>"]/g,m=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[m]));}
async function post(body){await fetch('/api/save',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});}
async function save(adv){
  const c=clips[i],v=document.getElementById('tx').value.trim();
  done[c.id]={id:c.id,audio_filepath:c.path,duration:c.duration,
    machine_guess:c.guess,verified_text:v,unusable:false};
  await post(done[c.id]);
  if(adv){go(1);}else{const n=Object.values(done).filter(x=>x.verified_text||x.unusable).length;
    bar.style.width=(100*n/clips.length)+'%';}
}
async function unusable(){
  const c=clips[i];
  done[c.id]={id:c.id,audio_filepath:c.path,duration:c.duration,
    machine_guess:c.guess,verified_text:'',unusable:true};
  await post(done[c.id]); go(1);
}
function go(d){i=Math.max(0,Math.min(clips.length-1,i+d));render();}
document.addEventListener('keydown',e=>{
  if(e.ctrlKey&&e.key==='Enter'){e.preventDefault();save(1);}
  else if(e.ctrlKey&&(e.key==='k'||e.key==='K')){e.preventDefault();unusable();}
  else if(e.code==='Space'&&e.target.tagName!=='TEXTAREA'){e.preventDefault();
    const a=document.getElementById('au'); a.paused?a.play():a.pause();}
});
boot();
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    clips = []
    done = {}

    def log_message(self, *a):
        pass  # keep the console readable

    def _send(self, code, ctype, body):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = unquote(urlparse(self.path).path)
        if path == "/":
            return self._send(200, "text/html; charset=utf-8", PAGE.encode("utf-8"))
        if path == "/api/clips":
            payload = json.dumps({"clips": self.clips, "done": self.done}).encode("utf-8")
            return self._send(200, "application/json", payload)
        if path.startswith("/audio/"):
            rel = path[len("/audio/"):]
            target = (ROOT / rel).resolve()
            # Never serve outside the repo, whatever the client asks for.
            if ROOT.resolve() not in target.parents or not target.exists():
                return self._send(404, "text/plain", b"not found")
            ctype = "audio/wav" if target.suffix == ".wav" else "audio/mpeg"
            return self._send(200, ctype, target.read_bytes())
        return self._send(404, "text/plain", b"not found")

    def do_POST(self):
        if urlparse(self.path).path != "/api/save":
            return self._send(404, "text/plain", b"not found")
        row = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        Handler.done[row["id"]] = row
        save_all(Handler.done)
        return self._send(200, "application/json", b'{"ok":true}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    ap.add_argument("--port", type=int, default=7870)
    args = ap.parse_args()

    manifest = Path(args.manifest)
    if not manifest.exists():
        sys.exit(f"Manifest not found: {manifest}")

    Handler.clips = load_clips(manifest)
    Handler.done = load_done()
    if not Handler.clips:
        sys.exit(f"No clips in {manifest} resolve to audio on this machine.")

    verified = sum(1 for r in Handler.done.values() if r.get("verified_text") or r.get("unusable"))
    total_min = sum(c["duration"] for c in Handler.clips) / 60
    print(f"{len(Handler.clips)} clips ({total_min:.0f} min) from {manifest.name}")
    print(f"{verified} already verified -> {OUTPUT.name}")
    print(f"\n  Open http://localhost:{args.port}\n\nCtrl-C to stop. Progress is saved as you go.")
    HTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
