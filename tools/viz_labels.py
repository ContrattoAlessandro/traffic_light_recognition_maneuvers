"""Draw maneuver labels and traffic-light boxes on front images for spot-checking."""
import argparse
import json
import os

from PIL import Image, ImageDraw

BOX_RGB = {"red": (255, 40, 40), "green": (40, 220, 40), "yellow": (255, 200, 0)}


def render(rec, image_root, out_path, width=775):
    img = Image.open(os.path.join(image_root, rec["image"])).convert("RGB")
    draw = ImageDraw.Draw(img)
    for b in rec["boxes"]:
        # relevant lights get a thick box, ignored ones a thin one
        draw.rectangle(b["box"], outline=BOX_RGB[b["color"]], width=8 if b["relevant"] else 2)
    text = "  ".join(f"{d}:{rec['color'][d]}/{rec['pass'][d]}" for d in rec["color"])
    text += f"   kept={rec['kept']} conflict={rec['conflict']}"
    scale = width / img.width
    img = img.resize((width, round(img.height * scale)))
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, width, 18], fill=(0, 0, 0))
    draw.text((4, 3), text, fill=(255, 255, 255))
    img.save(out_path, quality=85)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--image-root", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--every", type=int, default=1)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    with open(args.labels) as f:
        records = [json.loads(l) for l in f]
    for k, rec in enumerate(records[:: args.every]):
        render(rec, args.image_root, os.path.join(args.out_dir, f"{k:04d}_{rec['segment_id']}_{rec['timestamp']}.jpg"))


if __name__ == "__main__":
    main()
