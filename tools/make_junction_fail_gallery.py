"""Gallery of DTLD frames where the junction-lights + pictogram + color rule fails (tools/audit_junction_rule.py).

50 frames, at most one per session (seed 0): 30 with red and green circles, 12 with circles that
disagree with yellow involved (often heads switching a moment apart), 8 with arrows of one maneuver
disagreeing. Each image shows:
- junction lamps: thick box in the lamp's color, labelled "R" (DTLD relevant) or "J" (non-relevant
  lamp the size rule adds to the junction), the pictogram and the state;
- other front-facing vehicle lamps (left out by the size rule): thin grey box;
- a zoom strip of the junction lamps.
The page asks, per frame, what the failure is (real conflict / wrong junction lamp / timing / label
error / unsure), keeps the answers in the browser and exports them as JSON. Writes data/junction_fail/.
"""
import collections
import json
import os
import random
import statistics
import sys

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.audit_junction_rule import ARROWS, DTLD_ROOT, STATE, classify  # noqa: E402

JPG = os.path.join(os.environ.get("DATASETS", "../traffic-light-detection/datasets"), "DTLD_jpg")
OUT = "data/junction_fail"
QUOTA = {"red+green circles": 30, "circles with yellow": 12, "arrows disagree": 8}
RGB = {"red": (235, 50, 50), "yellow": (245, 200, 0), "green": (40, 200, 80)}
WIDTH, ZOOM = 1400, 110
SHORT = {"circle": "circle", "arrow_left": "arrow L", "arrow_right": "arrow R", "arrow_straight": "arrow S",
         "arrow_straight_left": "arrow SL", "unknown": "unknown"}


def font(size):
    for f in ("arialbd.ttf", "arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(f, size)
        except OSError:
            pass
    return ImageFont.load_default()


def candidates():
    out = collections.defaultdict(list)
    for split in ("train", "test"):
        for im in json.load(open(os.path.join(DTLD_ROOT, f"DTLD_{split}.json")))["images"]:
            front = [l for l in im["labels"] if l["attributes"]["direction"] == "front"
                     and l["attributes"]["pictogram"] in set(ARROWS) | {"circle", "unknown"}]
            rel = [l for l in front if l["attributes"]["relevance"] == "relevant"]
            if not rel:
                continue
            h = statistics.median(l["h"] for l in rel)
            nrel = [l for l in front if l["attributes"]["relevance"] != "relevant" and l["h"] >= 0.6 * h]
            lights = [("unreadable" if l["attributes"]["pictogram"] == "unknown" else l["attributes"]["pictogram"],
                       STATE[l["attributes"]["state"]]) for l in rel + nrel if l["attributes"]["state"] in STATE]
            if not lights:
                continue
            cat, _ = classify(lights)
            if not cat.startswith("fail"):
                continue
            circles = {c for p, c in lights if p == "circle"}
            kind = ("red+green circles" if {"red", "green"} <= circles else "circles with yellow") \
                if cat == "fail_circles_disagree" else "arrows disagree" if cat == "fail_arrows_disagree" else cat
            path = im["image_path"].replace("\\", "/")
            out[kind].append({"split": split, "session": "/".join(path.split("/")[-4:-1]),
                              "jpg": os.path.join(JPG, split, os.path.basename(path).rsplit(".", 1)[0] + ".jpg"),
                              "rel": rel, "nrel": nrel, "other": [l for l in front if l not in rel and l not in nrel]})
    return out


def render(c, path):
    img = Image.open(c["jpg"]).convert("RGB")
    s = WIDTH / img.width
    im = img.resize((WIDTH, round(img.height * s)))
    d = ImageDraw.Draw(im)
    f = font(14)
    for l in c["other"]:
        d.rectangle([l["x"] * s, l["y"] * s, (l["x"] + l["w"]) * s, (l["y"] + l["h"]) * s], outline=(150, 150, 150), width=1)
    lamps = [("R", l) for l in c["rel"]] + [("J", l) for l in c["nrel"]]
    for tag, l in lamps:
        a = l["attributes"]
        col = RGB.get(STATE.get(a["state"]), (200, 200, 200))
        x1, y1, x2, y2 = l["x"] * s, l["y"] * s, (l["x"] + l["w"]) * s, (l["y"] + l["h"]) * s
        d.rectangle([x1 - 3, y1 - 3, x2 + 3, y2 + 3], outline=col, width=3 if tag == "R" else 2)
        label = f"{tag} {SHORT.get(a['pictogram'], a['pictogram'])} {a['state']}"
        tw = d.textlength(label, font=f)
        ty = y1 - 22 if y1 > 24 else y2 + 4
        d.rectangle([x1 - 3, ty, x1 + tw + 3, ty + 18], fill=(0, 0, 0))
        d.text((x1, ty + 1), label, fill=col, font=f)
    per_row = WIDTH // (ZOOM + 4)
    rows = (len(lamps) + per_row - 1) // per_row
    strip = Image.new("RGB", (WIDTH, rows * (ZOOM + 22)), (30, 30, 30))
    sd = ImageDraw.Draw(strip)
    for n, (tag, l) in enumerate(lamps):
        cx, cy, side = l["x"] + l["w"] / 2, l["y"] + l["h"] / 2, max(l["w"], l["h"], 8) * 2.0
        crop = img.crop((round(cx - side / 2), round(cy - side / 2), round(cx + side / 2), round(cy + side / 2))).resize((ZOOM, ZOOM))
        ImageDraw.Draw(crop).rectangle([0, 0, ZOOM - 1, ZOOM - 1], outline=RGB.get(STATE.get(l["attributes"]["state"]), (200, 200, 200)), width=3)
        x, y = (n % per_row) * (ZOOM + 4) + 2, (n // per_row) * (ZOOM + 22)
        strip.paste(crop, (x, y + 2))
        sd.text((x + 2, y + ZOOM + 4), f"{tag} {SHORT.get(l['attributes']['pictogram'], '?')}", fill=(230, 230, 230), font=font(12))
    out = Image.new("RGB", (WIDTH, im.height + strip.height))
    out.paste(im, (0, 0))
    out.paste(strip, (0, im.height))
    out.save(path, quality=88)


PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>DTLD Rule Failures</title><style>
body{margin:0;background:#111;color:#eee;font:15px/1.4 system-ui,sans-serif}
header{position:sticky;top:0;z-index:5;background:#181818;border-bottom:1px solid #333;padding:10px 16px;display:flex;flex-wrap:wrap;gap:10px 18px;align-items:center}
header h1{font-size:17px;margin:0}.grow{flex:1}
button{background:#262626;color:#eee;border:1px solid #444;border-radius:6px;padding:5px 10px;font:inherit;cursor:pointer}
main{max-width:1440px;margin:auto;padding:16px}.help{color:#aaa;font-size:13.5px}
.card{background:#1b1b1b;border:2px solid #333;border-radius:10px;margin:0 0 26px;padding:12px;scroll-margin-top:70px}
.card.focus{border-color:#4c9bff}.card.done{box-shadow:inset 6px 0 0 #4c9bff}
.meta{display:flex;flex-wrap:wrap;gap:6px 14px;margin-bottom:8px}.tag{font-size:12.5px;padding:1px 7px;border-radius:10px;background:#2a2a2a;color:#ccc}
img{width:100%;display:block;border-radius:4px}.ctl{margin-top:8px;display:flex;flex-wrap:wrap;gap:8px}
.ctl button.on{background:#4c9bff;color:#000}textarea{width:100%;min-height:32px;margin-top:8px;background:#141414;color:#eee;border:1px solid #444;border-radius:6px;padding:6px;font:inherit}
@media (max-width:700px){main{padding:16px}}
</style></head><body>
<header><h1>DTLD: frames where the junction rule fails</h1><span id="progress"></span><span class="grow"></span><button id="export">Export JSON</button></header>
<main><p class="help">Thick box, <b>R</b>: lamp DTLD marks relevant for the route. Box, <b>J</b>: non-relevant front lamp the size rule adds to the junction.
Thin grey: other front-facing vehicle lamps, left out. Label: pictogram and state. Strip: zoom on the junction lamps.<br>
For each frame, what is the failure? Keys on the highlighted card: <b>1</b> real conflict (lights of this approach really disagree),
<b>2</b> wrong junction lamp (a J lamp belongs to another road or junction), <b>3</b> timing (heads changing a moment apart),
<b>4</b> label error, <b>5</b> unsure, <b>j</b>/<b>k</b> next / previous, <b>n</b> note.</p><div id="list"></div></main>
<script>
const ITEMS=/*ITEMS*/[];const KEY="junction_fail_v1";const V=["real conflict","wrong junction lamp","timing","label error","unsure"];
let st={};try{st=JSON.parse(localStorage.getItem(KEY)||"{}")}catch(e){}
const save=()=>{try{localStorage.setItem(KEY,JSON.stringify(st))}catch(e){}};const rec=id=>(st[id]||={verdict:null,note:""});let focus=0;
function draw(){document.getElementById("list").innerHTML=ITEMS.map((it,i)=>`<section class="card" id="c${i}" data-i="${i}">
<div class="meta"><b>${i+1}. ${it.kind}</b><span class="tag">${it.session}</span><span class="tag">${it.split}</span><span class="tag">${it.file}</span></div>
<img src="${it.id}.jpg" alt="${it.kind} ${it.file}"><div class="ctl">${V.map((v,k)=>`<button data-v="${v}">${k+1} ${v}</button>`).join("")}</div>
<textarea placeholder="note (optional)"></textarea></section>`).join("");ITEMS.forEach((it,i)=>{paint(i);document.querySelector(`#c${i} textarea`).value=rec(it.id).note});setFocus(0,false)}
function paint(i){const r=rec(ITEMS[i].id),el=document.getElementById("c"+i);el.classList.toggle("done",!!r.verdict);
el.querySelectorAll(".ctl button").forEach(b=>b.classList.toggle("on",b.dataset.v===r.verdict));
const n=ITEMS.filter(x=>rec(x.id).verdict).length;document.getElementById("progress").textContent=`${n} / ${ITEMS.length} reviewed`}
function setV(i,v){const r=rec(ITEMS[i].id);r.verdict=r.verdict===v?null:v;save();paint(i)}
function setFocus(i,scroll=true){i=Math.max(0,Math.min(ITEMS.length-1,i));document.querySelectorAll(".card").forEach(c=>c.classList.remove("focus"));
const c=document.getElementById("c"+i);c.classList.add("focus");focus=i;if(scroll)c.scrollIntoView({block:"start"})}
document.getElementById("list").addEventListener("click",e=>{const c=e.target.closest(".card");if(!c)return;focus=+c.dataset.i;setFocus(focus,false);if(e.target.dataset.v)setV(focus,e.target.dataset.v)});
document.getElementById("list").addEventListener("input",e=>{if(e.target.tagName==="TEXTAREA"){rec(ITEMS[+e.target.closest(".card").dataset.i].id).note=e.target.value;save()}});
document.addEventListener("keydown",e=>{if(e.target.tagName==="TEXTAREA"){if(e.key==="Escape")e.target.blur();return}if(e.ctrlKey||e.metaKey||e.altKey)return;const k=e.key.toLowerCase();
if(k==="j")setFocus(focus+1);else if(k==="k")setFocus(focus-1);else if("12345".includes(k)&&k)setV(focus,V[+k-1]);
else if(k==="n"){e.preventDefault();document.querySelector(`#c${focus} textarea`).focus()}else return;e.preventDefault()});
document.getElementById("export").addEventListener("click",()=>{const a=document.createElement("a");
a.href=URL.createObjectURL(new Blob([JSON.stringify(ITEMS.map(it=>({...it,...rec(it.id)})),null,1)],{type:"application/json"}));a.download="junction_fail_v1.json";a.click()});
draw();
</script></body></html>"""


def main():
    os.makedirs(OUT, exist_ok=True)
    cands = candidates()
    rng = random.Random(0)
    items = []
    for kind, n in QUOTA.items():
        by_session = collections.defaultdict(list)
        for c in cands[kind]:
            by_session[c["session"]].append(c)
        sessions = rng.sample(sorted(by_session), min(n, len(by_session)))
        print(f"{kind}: {len(cands[kind])} frames in {len(by_session)} sessions -> {len(sessions)}", flush=True)
        for sname in sessions:
            c = rng.choice(by_session[sname])
            fid = f"f{len(items):02d}"
            render(c, os.path.join(OUT, f"{fid}.jpg"))
            items.append({"id": fid, "kind": kind, "split": c["split"], "session": c["session"], "file": os.path.basename(c["jpg"])})
    with open(os.path.join(OUT, "index.html"), "w", encoding="utf-8") as f:
        f.write(PAGE.replace("/*ITEMS*/[]", json.dumps(items)))
    print(len(items), "frames ->", OUT)


if __name__ == "__main__":
    main()
