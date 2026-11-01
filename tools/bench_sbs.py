#!/usr/bin/env python3
"""Side-by-side frames for the UE5 bench.

    python3 tools/bench_sbs.py <out> [--scenes bench/scenes.txt] [--refs DIR]

Reads <out>/ours/<id>.png (from `silt capture --bench <out>`) and the
reference named in bench/scenes.txt, writes <out>/sbs/NN-<id>.jpg (Silt left,
UE5 right, each half fitted into 960x540) and <out>/index.html with a 1-5
score field per scene. A scene without a still gets a card that says why.
References are third-party screenshots and are not in the repo: they are
read from --refs, $BENCH_REFS or bench/refs/ (gitignored). A missing
reference gets a placeholder half, so CI still produces the Silt side.
Needs Pillow only. No network.
"""
import argparse
import html
import json
import os
import sys

from PIL import Image, ImageDraw, ImageFont

HALF_W, HALF_H, BAR = 960, 540, 34
BG = (18, 18, 20)
FG = (235, 235, 235)
DIM = (150, 150, 150)


def parse(path):
    scenes = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(None, 11)
            if len(parts) < 11:
                sys.exit(f"bad scene line: {line}")
            scenes.append({
                "id": parts[0],
                "preset": parts[1],
                "ref": parts[10],
                "title": parts[11] if len(parts) > 11 else parts[0],
            })
    return scenes


def font(size):
    for name in ("DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def fit(img):
    img = img.convert("RGB")
    img.thumbnail((HALF_W, HALF_H), Image.LANCZOS)
    canvas = Image.new("RGB", (HALF_W, HALF_H), BG)
    canvas.paste(img, ((HALF_W - img.width) // 2, (HALF_H - img.height) // 2))
    return canvas


def missing_card(text, head="NO SILT SCENE YET", tint=(40, 30, 30)):
    canvas = Image.new("RGB", (HALF_W, HALF_H), tint)
    d = ImageDraw.Draw(canvas)
    f = font(26)
    words, lines, cur = text.split(), [], ""
    for w in words:
        trial = (cur + " " + w).strip()
        if d.textlength(trial, font=f) > HALF_W - 80:
            lines.append(cur)
            cur = w
        else:
            cur = trial
    lines.append(cur)
    y = HALF_H // 2 - len(lines) * 18 - 30
    d.text((40, y), head, font=font(30), fill=(255, 170, 120))
    for i, ln in enumerate(lines):
        d.text((40, y + 50 + i * 36), ln, font=f, fill=FG)
    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--scenes", default="bench/scenes.txt")
    ap.add_argument("--refs", default=os.environ.get("BENCH_REFS", ""))
    ap.add_argument("--label", default=os.environ.get("BENCH_LABEL", ""))
    args = ap.parse_args()
    refs = args.refs or os.path.join(os.path.dirname(os.path.abspath(args.scenes)), "refs")
    scenes = parse(args.scenes)
    sbs_dir = os.path.join(args.out, "sbs")
    os.makedirs(sbs_dir, exist_ok=True)
    rows = []
    for n, s in enumerate(scenes, 1):
        ours_path = os.path.join(args.out, "ours", s["id"] + ".png")
        have = os.path.exists(ours_path)
        left = fit(Image.open(ours_path)) if have else missing_card(s["title"])
        ref_path = os.path.join(refs, s["ref"])
        have_ref = os.path.exists(ref_path)
        right = fit(Image.open(ref_path)) if have_ref else missing_card(
            f"{s['ref']} is not in this checkout. UE5 screenshots stay out of the public repo; "
            "put them in bench/refs/ or pass --refs.",
            head="REFERENCE NOT AVAILABLE", tint=(28, 30, 40))
        frame = Image.new("RGB", (HALF_W * 2, HALF_H + BAR), BG)
        frame.paste(left, (0, BAR))
        frame.paste(right, (HALF_W, BAR))
        d = ImageDraw.Draw(frame)
        f = font(20)
        d.text((12, 6), f"Silt  {s['id']}  {args.label}".rstrip(), font=f, fill=FG)
        d.text((HALF_W + 12, 6), f"UE5 reference  {os.path.basename(s['ref'])}", font=f, fill=DIM)
        d.line([(HALF_W, 0), (HALF_W, HALF_H + BAR)], fill=(80, 80, 80), width=2)
        name = f"{n:02d}-{s['id']}.jpg"
        frame.save(os.path.join(sbs_dir, name), quality=88)
        rows.append({**s, "file": "sbs/" + name, "rendered": have, "ref_found": have_ref})
        print(f"{name} {'rendered' if have else 'missing'}{'' if have_ref else ' (no ref)'}")
    with open(os.path.join(args.out, "bench.json"), "w", encoding="utf-8") as f:
        json.dump({"label": args.label, "scenes": rows}, f, indent=1)
    write_index(args.out, rows, args.label)
    done = sum(r["rendered"] for r in rows)
    print(f"bench: {done}/{len(rows)} scenes rendered, index {os.path.join(args.out, 'index.html')}")


def write_index(out, rows, label):
    cards = []
    for r in rows:
        status = "" if r["rendered"] else '<span class="miss">no Silt scene yet</span>'
        cards.append(f"""
<section data-id="{html.escape(r['id'])}">
  <h2>{html.escape(r['id'])} {status}</h2>
  <p>{html.escape(r['title'])}</p>
  <a href="{r['file']}"><img src="{r['file']}" loading="lazy" alt="{html.escape(r['id'])}"></a>
  <label>Score 1-5 <input type="number" min="1" max="5" step="1"></label>
  <input class="note" type="text" placeholder="note">
</section>""")
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Silt vs UE5 bench</title>
<style>
:root {{ color-scheme: dark; }}
body {{ background:#121214; color:#e8e8e8; font:15px system-ui,sans-serif; margin:0 16px 40px; }}
h1 {{ font-size:20px; }} h2 {{ font-size:16px; margin:0 0 4px; }}
section {{ margin:24px 0; padding-top:12px; border-top:1px solid #333; }}
img {{ width:100%; max-width:1920px; display:block; margin:8px 0; }}
.miss {{ color:#ffaa78; font-weight:400; font-size:13px; margin-left:8px; }}
input {{ background:#222; color:#eee; border:1px solid #444; padding:4px; }}
.note {{ width:min(600px,100%); margin-left:12px; }}
button {{ padding:6px 12px; }}
</style></head><body>
<h1>Silt vs UE5 bench {html.escape(label)}</h1>
<p>Silt left, UE5 reference right. Scores stay in this browser; "Export" copies them as JSON.</p>
<button id="export">Export scores</button> <span id="avg"></span>
{''.join(cards)}
<script>
const key = 'silt-bench-scores';
let saved = {{}};
try {{ saved = JSON.parse(localStorage.getItem(key) || '{{}}'); }} catch (e) {{}}
const secs = [...document.querySelectorAll('section')];
function collect() {{
  const out = {{}};
  for (const s of secs) {{
    const [score, note] = s.querySelectorAll('input');
    out[s.dataset.id] = {{ score: score.value ? +score.value : null, note: note.value }};
  }}
  return out;
}}
function refresh() {{
  const v = Object.values(collect()).map(x => x.score).filter(x => x);
  document.getElementById('avg').textContent = v.length ? `average ${{(v.reduce((a, b) => a + b, 0) / v.length).toFixed(2)}} over ${{v.length}}` : '';
}}
for (const s of secs) {{
  const [score, note] = s.querySelectorAll('input');
  const prev = saved[s.dataset.id] || {{}};
  if (prev.score) score.value = prev.score;
  if (prev.note) note.value = prev.note;
  for (const el of [score, note]) el.addEventListener('input', () => {{
    try {{ localStorage.setItem(key, JSON.stringify(collect())); }} catch (e) {{}}
    refresh();
  }});
}}
refresh();
document.getElementById('export').onclick = () => {{
  const text = JSON.stringify(collect(), null, 1);
  navigator.clipboard?.writeText(text).catch(() => {{}});
  prompt('Scores JSON', text);
}};
</script></body></html>
"""
    with open(os.path.join(out, "index.html"), "w", encoding="utf-8") as f:
        f.write(page)


if __name__ == "__main__":
    main()
