"""Check page for the rebuilt DTLD junction label (dtsu/junction_dtld.py) before it is used for training.

50 frames, one per session (seed 0):
- 25 where circles of the junction still disagree in color under the new label (what is left of the
  failures: real lane conflicts, or labelling mistakes of the rule);
- 25 "borderline" frames with at least one lamp added to the junction and at least one front vehicle
  lamp left out whose track ratio is 0.5-0.85 (just below the threshold).
Each image: R = route lamp (DTLD relevant, thick), J = lamp added as same junction, X = front vehicle
lamp left out (thin); J and X show their track ratio. The page asks whether the J / X split is right.
Writes data/junction_label/.
"""
import collections
import json
import os
import random
import sys

from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dtsu.junction_dtld import RATIO, front_vehicle, junction_role, session, track_ratios  # noqa: E402
from tools.audit_junction_rule import DTLD_ROOT, STATE  # noqa: E402
from tools.make_junction_fail_gallery import JPG, PAGE, RGB, SHORT, WIDTH, ZOOM, font  # noqa: E402

OUT = "data/junction_label"


def render(im, roles, ratios, jpg, path):
    img = Image.open(jpg).convert("RGB")
    s = WIDTH / img.width
    out = img.resize((WIDTH, round(img.height * s)))
    d = ImageDraw.Draw(out)
    f = font(14)
    shown = []
    for l, role in zip(im["labels"], roles):
        if l not in front_vehicle(im):
            continue
        tag = {"route": "R", "junction": "J"}.get(role, "X")
        a = l["attributes"]
        col = RGB.get(STATE.get(a["state"]), (200, 200, 200))
        x1, y1, x2, y2 = l["x"] * s, l["y"] * s, (l["x"] + l["w"]) * s, (l["y"] + l["h"]) * s
        d.rectangle([x1 - 3, y1 - 3, x2 + 3, y2 + 3], outline=col if tag != "X" else (170, 170, 170), width={"R": 3, "J": 2, "X": 1}[tag])
        r = ratios.get((session(im), l["track_id"]))
        label = f"{tag} {SHORT.get(a['pictogram'], a['pictogram'])} {a['state']}" + (f" {r:.2f}" if tag != "R" and r is not None else "")
        tw = d.textlength(label, font=f)
        ty = y1 - 22 if y1 > 24 else y2 + 4
        d.rectangle([x1 - 3, ty, x1 + tw + 3, ty + 18], fill=(0, 0, 0))
        d.text((x1, ty + 1), label, fill=col if tag != "X" else (190, 190, 190), font=f)
        shown.append((tag, l, col))
    per_row = WIDTH // (ZOOM + 4)
    rows = (len(shown) + per_row - 1) // per_row
    strip = Image.new("RGB", (WIDTH, rows * (ZOOM + 22)), (30, 30, 30))
    sd = ImageDraw.Draw(strip)
    for n, (tag, l, col) in enumerate(shown):
        cx, cy, side = l["x"] + l["w"] / 2, l["y"] + l["h"] / 2, max(l["w"], l["h"], 8) * 2.0
        crop = img.crop((round(cx - side / 2), round(cy - side / 2), round(cx + side / 2), round(cy + side / 2))).resize((ZOOM, ZOOM))
        ImageDraw.Draw(crop).rectangle([0, 0, ZOOM - 1, ZOOM - 1], outline=col if tag != "X" else (170, 170, 170), width=3)
        x, y = (n % per_row) * (ZOOM + 4) + 2, (n // per_row) * (ZOOM + 22)
        strip.paste(crop, (x, y + 2))
        sd.text((x + 2, y + ZOOM + 4), f"{tag} {SHORT.get(l['attributes']['pictogram'], '?')}", fill=(230, 230, 230), font=font(12))
    full = Image.new("RGB", (WIDTH, out.height + strip.height))
    full.paste(out, (0, 0))
    full.paste(strip, (0, out.height))
    full.save(path, quality=88)


def main():
    os.makedirs(OUT, exist_ok=True)
    images = [(sp, im) for sp in ("train", "test") for im in json.load(open(os.path.join(DTLD_ROOT, f"DTLD_{sp}.json")))["images"]]
    ratios = track_ratios([im for _, im in images])
    groups = {"circles still disagree": collections.defaultdict(list), "borderline": collections.defaultdict(list)}
    for sp, im in images:
        roles = junction_role(im, ratios)
        lamps = [(l, r) for l, r in zip(im["labels"], roles) if r]
        if not lamps:
            continue
        circles = {STATE[l["attributes"]["state"]] for l, _ in lamps if l["attributes"]["pictogram"] == "circle" and l["attributes"]["state"] in STATE}
        if len(circles) > 1:
            groups["circles still disagree"][session(im)].append((sp, im, roles))
        elif any(r == "junction" for _, r in lamps) and any(
                r is None and 0.5 <= ratios.get((session(im), l["track_id"]), 0) < RATIO
                for l, r in zip(im["labels"], roles) if l in front_vehicle(im)):
            groups["borderline"][session(im)].append((sp, im, roles))
    rng = random.Random(0)
    used, items = set(), []
    for kind, by_ses in groups.items():
        pool = sorted(s for s in by_ses if s not in used)
        print(f"{kind}: {sum(map(len, by_ses.values()))} frames in {len(by_ses)} sessions", flush=True)
        for ses in rng.sample(pool, min(25, len(pool))):
            used.add(ses)
            sp, im, roles = rng.choice(by_ses[ses])
            stem = os.path.basename(im["image_path"].replace("\\", "/")).rsplit(".", 1)[0]
            fid = f"f{len(items):02d}"
            render(im, roles, ratios, os.path.join(JPG, sp, stem + ".jpg"), os.path.join(OUT, f"{fid}.jpg"))
            items.append({"id": fid, "kind": kind, "split": sp, "session": "/".join(ses.split("/")[-3:]), "file": stem + ".jpg"})
    page = (PAGE.replace("DTLD Rule Failures", "Junction Label Check")
            .replace("DTLD: frames where the junction rule fails", f"DTLD: junction label check (track ratio &ge; {RATIO})")
            .replace("junction_fail_v1", "junction_label_v1")
            .replace('const V=["real conflict","wrong junction lamp","timing","label error","unsure"]',
                     'const V=["labels right","a J lamp is another junction","an X lamp is my junction","both kinds of error","unsure"]'))
    help_start = page.index('<p class="help">')
    help_end = page.index("</p>", help_start) + 4
    page = page[:help_start] + (
        '<p class="help">Thick box, <b>R</b>: route lamp (DTLD relevant). Box, <b>J</b>: lamp the new rule adds to my junction. '
        'Thin grey, <b>X</b>: front vehicle lamp left out (another junction, or not facing my approach). J and X show the track ratio '
        f'(lamp height / route-lamp height, median over the track); the rule keeps J when it is &ge; {RATIO}.<br>'
        'Is the J / X split right? Keys on the highlighted card: <b>1</b> labels right, <b>2</b> a J lamp belongs to another junction, '
        '<b>3</b> an X lamp belongs to my junction, <b>4</b> both kinds of error, <b>5</b> unsure, <b>j</b>/<b>k</b> next / previous, '
        '<b>n</b> note.</p>') + page[help_end:]
    with open(os.path.join(OUT, "index.html"), "w", encoding="utf-8") as f:
        f.write(page.replace("/*ITEMS*/[]", json.dumps(items)))
    print(len(items), "frames ->", OUT)


if __name__ == "__main__":
    main()
