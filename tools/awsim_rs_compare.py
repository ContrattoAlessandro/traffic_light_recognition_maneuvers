"""R vs S on the AWSIM 854px clip: frame counts and a side-by-side sample sheet (R left, S right)."""
import collections
import json

import cv2
import numpy as np

P = "runs/awsim/run_full_clean_picto_{}"
L = {m: [json.loads(x) for x in open(P.format(m) + ".jsonl")] for m in ("R", "S")}
out = {}
for m, rows in L.items():
    J = [[d for d in r["detections"] if d["junction"]] for r in rows]
    out[m] = {"frames": len(rows), "any_det": sum(bool(r["detections"]) for r in rows), "junction_frames": sum(bool(j) for j in J),
              "dets": sum(len(r["detections"]) for r in rows), "junction_dets": sum(map(len, J)),
              "picto": dict(collections.Counter(d["picto"] for j in J for d in j)),
              "color": dict(collections.Counter(d["color"] for j in J for d in j)),
              **{k: dict(collections.Counter(r["directions"][k] for r in rows)) for k in ("straight", "left", "right")}}
    print(m, json.dumps(out[m]))
R, S = L["R"], L["S"]
dis = [i for i, (a, b) in enumerate(zip(R, S)) if a["directions"]["straight"] != b["directions"]["straight"]]
out["straight_disagree"] = len(dis)
pairs = collections.Counter((R[i]["directions"]["straight"], S[i]["directions"]["straight"]) for i in dis)
out["straight_disagree_pairs"] = {f"{a}->{b}": n for (a, b), n in pairs.most_common()}
print("straight disagree", len(dis), out["straight_disagree_pairs"])


# sample frames: the longest disagreement runs (one frame from each), filled with evenly spaced frames
def runs(ix):
    seg, cur = [], []
    for i in ix:
        if cur and i != cur[-1] + 1:
            seg.append(cur)
            cur = []
        cur.append(i)
    return seg + ([cur] if cur else [])


segs = sorted(runs(dis), key=len, reverse=True)
picks = sorted(s[len(s) // 2] for s in segs[:3])
picks.append(next(i for i in range(len(R) // 2, len(R)) if R[i]["directions"]["straight"] == S[i]["directions"]["straight"] != "none"))
out["sample_frames"] = picks
print("sample frames", picks, "longest runs", [len(s) for s in segs[:3]])
caps = {m: cv2.VideoCapture(P.format(m) + ".mp4") for m in ("R", "S")}
rows = []
for f in picks:
    ims = []
    for m in ("R", "S"):
        caps[m].set(1, f)
        ok, im = caps[m].read()
        assert ok, (m, f)
        cv2.rectangle(im, (300, 0), (520, 30), (0, 0, 0), -1)
        cv2.putText(im, f"{m}  frame {f}", (308, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
        ims.append(im)
    rows.append(np.hstack(ims))
sheet = cv2.resize(np.vstack(rows), None, fx=0.7, fy=0.7)
cv2.imwrite("runs/awsim/run_full_clean_picto_RS_sample.jpg", sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])
json.dump(out, open("runs/awsim/run_full_clean_picto_RS_counts.json", "w"), indent=1)
print(sheet.shape)
