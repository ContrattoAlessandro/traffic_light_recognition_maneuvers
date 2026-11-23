# Experiment log: maneuver-level traffic signal understanding

This is the chronological research log of the project. Each experiment's question, data, arms, endpoints and
decision rule were written down **before** any code or result for that experiment existed; results, deviations
and amendments were added afterwards as dated entries, never by editing the design in place. Every number in it
comes from the scripts named next to it.

**Where the code is.** The scripts and modules named in this log (`tools/…`, `dtsu/picto_rules.py`, …) are those of
git tag `pre-cleanup`. On 2026-11-20 the repository was reorganised around the main model R (last entry of this
log); the current code is described in `README.md`, and its outputs were checked against the results below.

The task follows Zou et al., *One Perception, All Maneuvers: Directional Traffic Signal Understanding for
Maneuver-Level Signal Intent Prediction* (arXiv 2609.32316): from one front-camera image, predict for each ego
maneuver the color of the signal that governs it and whether the maneuver may proceed. An early independent
re-implementation of that paper's Swin-Tiny direction-aware model was removed from the repository on
2026-11-20, because the models studied here diverged from it; it remains in the git history (commit `366a767`).

## Task and metrics

From one front-view image, predict a color c ∈ {none, red, yellow, green} and a passability p ∈ {stop, go}
for each maneuver d ∈ {straight, left, right}. Passability follows the color: red → stop, everything else → go.

- Acc_color and Acc_pass are averaged over all frame × direction states.
- EM: a frame counts only if all colors and all passabilities are correct.
- TSS = (Acc_color + Acc_pass + EM) / 3.
- False-Go = #(gt stop, pred go) / #(gt stop), pooled over directions.
- False-Stop = #(gt go, pred stop) / #(gt go), pooled over directions.

The paper also scores a U-turn maneuver. It is not an output here: in the rebuilt labels U-turn takes the
left-turn signal in 99.9% of frames, and DTLD and LISA have no U-turn signal, so it would only count left a
second time. Model selections made before U-turn was dropped (2026-11-19) were kept as made, and the reported
evaluations were re-scored over three directions (`tools/rescore_three_dirs.sh`). Implementation:
`dtsu/metrics.py`, unit-tested.

## OpenLane-V2 labels (2026-10-26)

**Data.** OpenLane-V2 subset_A (Argoverse 2, front camera `ring_front_center`, 1550 × 2048 portrait) and
subset_B (nuScenes, `CAM_FRONT`, 1600 × 900). Traffic elements are annotated only on the front view, as 2D
boxes. The official val split is the evaluation set; the test split has no usable annotations. The image tars
bundle all cameras, so every tar has to be downloaded and only the front camera is extracted
(`tools/extract_front.sh`).

**Label rules** (`dtsu/labels.py`). The paper releases neither labels nor code, so per-frame, per-direction
labels are rebuilt from the annotations:

| # | Gap in the paper | Interpretation |
|---|-----|----------------|
| 1 | Light-to-maneuver association | `topology_lcte` links lights to lanes; each lane is followed (`topology_lclc`) to the first junction lane being entered and classified straight / left / right / U-turn by its heading change. Only lanes of the ego approach count (aligned heading, nearest junction). |
| 2 | Which light governs a maneuver | The most specific light wins; left without its own light inherits straight; right takes only a dedicated right-only signal (US rules), otherwise "none". |
| 3 | Frame filtering | Keep frames with ≥ 1 relevant colored light and no conflict between equally specific lights. The paper's manual QC cannot be replicated. |
| 4 | Passability | Per the paper's Fig. 2 legend: red → stop; green, yellow and none → go. |
| 5 | Frame counts | The paper reports 23,904 train / 5,147 val, more than subset_A holds before filtering (22,477 / 4,806). A+B fits its counts and its label distribution (below), so A+B is used. |

**Data scope: A+B with US rules.** With US driving rules applied to both subsets, including the Singapore
scenes of subset_B, the label distribution reproduces the paper's Fig. 3 closely: straight red/green
37.3/53.9 vs 40.7/49.7, right none 96.7 vs 96.9, U-turn = left 99.9%. Mirroring the rules for left-hand
traffic does not match (right none 85.7%). The left-hand-traffic variant of the labels is kept in
`data/labels/variants/` for analysis only.

| Kept frames | train | val |
|---|---|---|
| subset_A | 10,141 / 22,477 | 2,220 / 4,806 |
| subset_B | 6,806 / 27,968 | 1,565 / 6,019 |
| A+B | 16,947 | 3,785 |
| paper | 23,904 | 5,147 |

About 71% as many frames as the paper are kept; the remaining difference is unexplained.

**Label spot-check.** 30 val frames, 6 per tricky case: turn arrows, straight = none, dedicated right,
crossing the junction, conflicts.
- It found one bug: when the ego stands at the stop line, the junction lane starts slightly behind it, and the
  builder skipped to the next junction. The fix (`ENTERING_X = -3 m`) changed about 140 subset_B frames and
  2 subset_A frames.
- The rest looked right or plausible.
- Conflicts are mostly a near light plus a farther intersection's light, and stay dropped (596 frames).
- In Singapore a green right arrow gives right = none. That follows from the US-rule convention that matches Fig. 3.

**Tuning split.** `data/labels/tune/` holds 10% of train scenes (70 per subset): 1,185 A + 785 B kept
frames (`tools/make_tune_split.py`). Hyperparameters, epochs and score thresholds are chosen on these, never
on val.

## Joint-class RF-DETR baseline (2026-10-29)

`dtsu/rfdetr_baseline.py`, `tools/make_rfdetr_coco.py`, `tools/train_rfdetr.py`, `tools/eval_rfdetr.py`.

- **Classes.** Every annotated light becomes one detection class `<moves>|<color>`, where `<moves>` is the
  set of ego maneuvers the light controls according to the label builder (e.g. `left+straight`, `generic`) or
  `irrelevant` for lights of other lanes or junctions: 46 classes on OpenLane-V2 A+B.
- **Decoding.** Detections above a score threshold go through class-agnostic NMS (IoU 0.5) and the label
  builder's own assignment rule (`assign_scored`), with the highest score breaking ties. Ground-truth boxes
  through this rule reproduce every label (oracle TSS 1.0000), so all errors are detection, class and
  relevance errors.
- **Recipe.** RF-DETR Medium (34.6M parameters) fine-tuned from the released weights with the library's
  recipe, except: no horizontal flip and no random crop (they corrupt left/right and can cut out the light
  that sets a label), one square resolution of 1024, brightness/contrast jitter. lr 1e-4 / encoder 1.5e-4,
  batch 4 × grad-accum 4.
- **Tuning** (fit split, scored on the tune split, four directions): Medium at the default lr 0.905, Medium
  at half the lr 0.893, Large 0.896. Tune TSS plateaus after two epochs, best threshold 0.20. The tuning run
  `runs/tune_rfdetr/r1_m_default` (fit split, epoch 1, threshold 0.20) is the "A+B" model of the later
  experiments.
- **Full-train run** (`runs/rfdetr_m_AB`, 4 epochs, threshold 0.20, last epoch reported): TSS 0.899 on A val
  and 0.915 on B val (three directions). Right-turn red is not learned (False-Go 0.64 on A val, 14 frames).

## Extra datasets: DTLD and LISA (2026-10-29)

Goal: train on more varied data (German cities, US night driving) for robustness. Labels follow
the same conventions as `dtsu/labels.py` (US rules, right = dedicated signal only, U-turn = left), so
the sets can be mixed. Build everything with `bash tools/build_ext_labels.sh`.

- **DTLD v2.0** (`dtsu/labels_dtld.py`). DTLD's `relevant` flag follows the driven route, so the
  lamps for each route come from the reviewed counterfactual rule of traffic-light-detection's
  `route_labels_20261023`: front-facing vehicle lamps at the ego intersection; an arrow for the route
  if there is one, else the circles. red_yellow counts as red. Frames are dropped for conflicting
  lamps (619 train), off/unknown lamps (656), rule mismatch against the reviewed route (484), or no
  relevant vehicle lamp (949).
  - **Right turns follow German rules:** a right arrow if there is one, otherwise the circle (no right
    on red). Records carry `near_turn: "signal"`. OpenLane-V2 and LISA keep the paper's US convention
    (right = dedicated signal only), so right-turn labels mean different things across datasets.
  - DTLD train right: 63% green / 27% red / 3% yellow / 7% none (US convention: 97% none).
- **LISA** (`dtsu/labels_lisa.py`). It has no relevance info, so lights at the ego intersection are
  found by size: at least 0.6 × the tallest light *of the same kind* (ball / left arrow / straight
  arrow). Comparing per kind keeps far-side balls next to a big near-side left-arrow pole (spot-check).
  Frames are dropped for conflicts (130 train), or for showing only arrows (LISA leaves some visible
  balls unannotated). Right is always none.
- **Splits.** test = official test (DTLD test; LISA day/nightSequence1–2). val = held out from the
  official train by whole session/clip: `data/labels/tune/{dtld,lisa}_train_{fit,tune}.jsonl`.

| Kept frames | train (fit) | val (tune) | test | canvas |
|---|---|---|---|---|
| DTLD | 23,222 | 2,706 | 11,347 | 1024×512 |
| LISA | 16,240 | 1,431 (dayClip9/10, nightClip1) | 17,612 | 1024×768 |

- Image roots: `<datasets>/DTLD_jpg/{train,test}` and `<datasets>/LISA`.
- LISA frames are consecutive video frames, so they are heavily redundant. `--stride n` keeps every n-th frame.
- Spot-checked about 50 frames across turn arrows, straight=none, dedicated right and conflicts. Two bugs were found and fixed: an inflated relevant-light count in DTLD, and the LISA size reference.
- **RF-DETR on DTLD** (`bash tools/run_rfdetr_dtld.sh` → `runs/rfdetr_m_dtld/`).
  - Same recipe as the OpenLane baseline (Medium, 1024 square, default lr), but 15 epochs with a
    per-epoch overfitting report (`train_rfdetr.py --report` → `epoch_report.jsonl`): eval-mode loss and
    direction metrics on the tune split vs a fixed 1,000-frame train probe. `touch <out>/STOP` ends
    training after the current epoch.
  - Trained on DTLD fit. The tune split picks the score threshold, and DTLD test is only reported.
  - 24 "<moves>|<color>" classes. `assign_scored` takes `near_turn`. GT-box oracle TSS = 1.0000 on fit/tune/test, and the OpenLane oracle is unchanged.
  - **Result (2026-10-30).** Best val TSS 0.9595 at epoch 4. After that, the probe loss kept falling
    (4.61 → 3.43) while the val loss rose (6.50 → 6.78), so it was overfitting. DTLD test, epoch 4 @ 0.25:
    TSS 0.9584, Acc_color 0.968, Acc_pass 0.979, EM 0.928, False-Go 0.030, False-Stop 0.016. Epoch 14
    gives the same TSS. Right is the weakest direction (Acc_color 0.950). About 5 epochs is enough.
    Latency on an RTX 5070 (bf16, batch 1): 22 ms network, 37 ms with JPEG decode and resize.

## D*_US: DTLD with the OpenLane-V2 label protocol (2026-11-05)

`tools/make_dtld_us.py` → `data/labels/dtld_us_{train,test}.jsonl` (+ `tune/dtld_us_train_{fit,tune}.jsonl`),
audit in `data/labels/dtld_us_audit.json`. Each record goes through `labels_dtld.to_openlane_protocol`.

**Protocol audit, OpenLane-V2 (`dtsu/labels.py`) vs DTLD (`dtsu/labels_dtld.py`).**

| Step | OpenLane-V2 | DTLD (German labels) | D*_US |
|---|---|---|---|
| Light → maneuvers | `topology_lcte` links + lane geometry | lamps serving each route (`serving`: pictogram, relevance flag, size rule) | same as DTLD: these per-lamp links are the analogue of `lcte` |
| Relevant lights | lights of ego-direction lanes at the nearest junction | front vehicle lamps at the ego intersection (≥ 0.6 × median relevant height) | same as DTLD |
| Straight / left / U-turn | `assign_directions`: most specific light wins; left ← straight; U-turn ← left | own resolver: arrows before circles; left ← straight; U-turn = left | **`assign_directions`** |
| Right | dedicated right-only signal, else none (US) | arrow, else circle (German) | **dedicated only (US)** |
| Conflict | disagreement among the most specific lights | any disagreement among a route's lamps | **most specific only** |
| Lights without a color | unknown-attribute elements are not lights (469/470 unlinked in a 300-frame sample) | frame dropped | **ignored** |
| red_yellow | does not occur | red | red |
| Frame filter | ≥ 1 relevant colored light, no conflict | + no unknown, + route review (`rule_ok`) | ≥ 1 relevant colored light, no conflict, + route review |

What the audit shows (test; train is the same in proportion):
- On the 11,347 frames both versions keep, straight, left and U-turn are identical. Only right changes (green → none 6,814, red → none 3,081, yellow → none 311).
- 154 frames are added (11,501 kept): 131 that had an off lamp, 23 conflicts that the specificity rule settles. None are lost.
- Right is 96.5% none, against 96.6% in OpenLane A+B. Straight is 65% green / 29% red / 2.4% none (OpenLane: 54 / 37 / 6.5).
- Oracle: ground-truth lamps through `assign_scored` with the US rule reproduce all D*_US labels (0 mismatches).
- What stays different, because it comes from the annotations, not the rules:
  - the link source (HD-map topology vs pictogram + route);
  - OpenLane's `generic` and U-turn lights, which DTLD doesn't have;
  - box conventions, which matter for detection metrics, not for TSS.
- The run trained on subset_B only has no `right|yellow` class. That touches 30 D*_US test frames, which stay in its test set (`--keep-all`), so every run is scored on the same frames.

## Cross-subset / cross-dataset protocol (2026-11-05, `tools/run_rfdetr_cross.sh`)

| Train | Test | Measures |
|---|---|---|
| A | A, B, D*_US | in-domain A + A→B + A→D* |
| B | B, A, D*_US | in-domain B + B→A + B→D* |
| A+B | A, B, D*_US | mixed in-domain + zero-shot D* |

Every run uses the same settings:
- source fit split, RF-DETR Medium at 1024, default lr, 4 epochs;
- the threshold is picked on the source's own tune split, per epoch;
- the reported epoch is the best on source tune TSS;
- images are squeezed to the square, as in training.

A+B = `runs/tune_rfdetr/r1_m_default` (already this recipe). A-only and B-only use `make_rfdetr_coco.py --subsets`; their classes come from that subset's train split (45 and 35 classes).

### Results (2026-11-05)

Each run is reported at its best source-tune epoch and that epoch's source-tune threshold. The D*_US scores use that same threshold. Scores are TSS / Acc_color / False-Go. D*_US is the official DTLD test split (11,501 frames). "no-right" leaves out right, which is 96.5% none.

| Train (epoch, thr) | A val | B val | D*_US | D*_US no-right |
|---|---|---|---|---|
| A (2, 0.20) | **0.900** / 0.915 / 0.178 | 0.789 / 0.822 / 0.288 | 0.755 / 0.774 / 0.323 | 0.731 / 0.715 / 0.313 |
| B (2, 0.15) | 0.827 / 0.839 / 0.327 | **0.904** / 0.921 / 0.102 | 0.782 / 0.796 / 0.219 | 0.767 / 0.746 / 0.208 |
| A+B (1, 0.20) | **0.905** / 0.917 / 0.145 | **0.918** / 0.932 / 0.098 | **0.827** / 0.852 / 0.213 | **0.814** / 0.814 / 0.201 |

Color accuracy per direction (straight / left / right):

| Train | A val | B val | D*_US |
|---|---|---|---|
| A | 0.886 / 0.899 / 0.978 | 0.724 / 0.815 / 0.935 | 0.700 / 0.722 / 0.953 |
| B | 0.804 / 0.789 / 0.976 | 0.888 / 0.923 / 0.950 | 0.743 / 0.747 / 0.948 |
| A+B | 0.891 / 0.902 / 0.975 | 0.916 / 0.930 / 0.952 | 0.824 / 0.810 / 0.964 |

Findings:
- **The cross-subset gap is large and roughly symmetric.** A→B loses 11.5 TSS points against B in-domain; B→A loses 7.8 against A in-domain. Moving to the other subset also doubles False-Go (A→B 0.29, B→A 0.33).
- **Mixing helps both subsets.** A+B matches A in-domain on A (+0.5 TSS) and beats B in-domain on B (+1.4), with the lowest False-Go on each, despite also having the fewest epochs (best at epoch 1).
- **Zero-shot to Germany (D*_US) costs 8–15 TSS points.** The ranking is A+B (0.827) > B (0.782) > A (0.755). Data diversity matters more than any single source. Straight and left lose the most, while right is easy because it is almost always none.
- **Tuning the threshold on DTLD barely helps.** It changes TSS by at most +0.001, so the drop is not a calibration problem.
- **The drop is mostly False-Go** (0.21–0.32 against 0.10–0.18 in-domain). It could come from wrong colors on detected German lamps or from missed lamps; the per-light OOD perception test should separate the two.

## OOD perception per light (2026-11-05, `tools/run_rfdetr_perception.sh`)

These datasets have no relevance labels, so each annotated light is matched to the run's detections without looking at the class. Matching is greedy by score. Each run uses its best source-tune epoch and threshold: A e2 / 0.20, B e2 / 0.15, A+B e1 / 0.20, DTLD e4 / 0.25.

Two matching criteria:
- **center**: each box's center lies inside the other box.
- **IoU ≥ 0.5**: the usual overlap test.

LISA and BDD100K boxes are about twice as wide as the lamp-only boxes of OpenLane, DTLD and ATLAS. IoU is strict for them, so center matching is the main number. Off, wait-on and none lights are ignored: neither misses nor false positives.

Detection: AP with center matching (AP with IoU ≥ 0.5 in brackets), then recall and precision at the run's threshold.

| Model | ATLAS (6,169) | S2TLD (13,317) | BDD100K val (18,538) | LISA test (57,649) |
|---|---|---|---|---|
| A | 0.483 (0.333) R 0.63 P 0.57 | 0.687 (0.649) R 0.89 P 0.63 | 0.642 (0.385) R 0.63 P 0.73 | 0.514 (0.232) R 0.44 P 0.75 |
| B | 0.509 (0.422) R 0.64 P 0.51 | 0.630 (0.588) R 0.94 P 0.49 | 0.647 (0.344) R 0.83 P 0.44 | 0.555 (0.164) R 0.86 P 0.44 |
| A+B | 0.487 (0.359) R 0.65 P 0.56 | 0.608 (0.560) R 0.93 P 0.54 | **0.662** (0.366) R 0.78 P 0.57 | 0.573 (0.151) R 0.78 P 0.58 |
| DTLD | **0.697** (0.567) R 0.69 P 0.71 | **0.812** (0.712) R 0.84 P 0.64 | 0.461 (0.103) R 0.53 P 0.47 | **0.750** (0.064) R 0.74 P 0.69 |

Color of matched lights (center matching): accuracy, then recall of yellow.

| Model | ATLAS | S2TLD | BDD100K | LISA |
|---|---|---|---|---|
| A | 0.951 / 0.13 | 0.934 / 0.32 | 0.944 / 0.09 | 0.941 / 0.12 |
| B | 0.935 / 0.79 | 0.940 / 0.36 | 0.934 / 0.20 | 0.972 / 0.29 |
| A+B | 0.959 / 0.59 | 0.958 / 0.61 | 0.956 / 0.18 | 0.913 / 0.68 |
| DTLD | 0.900 / 0.93 | **0.985** / 0.83 | **0.968** / 0.48 | **0.985** / 0.51 |

Breakdowns (center AP / recall):
- **LISA day / night.** A drops from 0.733 / 0.87 to 0.427 / 0.25 at night, B from 0.752 / 0.90 to 0.467 / 0.85, A+B from 0.750 / 0.90 to 0.487 / 0.73, DTLD from 0.838 / 0.86 to 0.710 / 0.69.
- **BDD100K.** Day AP is 0.70–0.75 for the OpenLane models (DTLD 0.60); night AP is 0.57–0.61 (DTLD 0.34).
- **ATLAS cameras.** The wide camera is hardest for every model, with AP 0.43–0.45 for OpenLane and 0.61 for DTLD.
- **S2TLD.** The 1920×1080 images are much harder than the 1280×720 ones, worst for DTLD (0.47 vs 0.87).

Findings:
- **Detection transfers by camera and box convention more than by country.** The DTLD model is the best detector wherever lights are drawn as lamps and seen up close: ATLAS (Germany, same convention), S2TLD and LISA. It is the worst on BDD100K, which has small dashcam lights and many night frames. The OpenLane models are flatter across datasets (center AP 0.48–0.69).
- **No OpenLane model is the best detector everywhere.** A leads on S2TLD, B on ATLAS, and A+B on BDD100K and LISA, all within 0.08 AP of each other. B runs at higher recall and lower precision on S2TLD, BDD and LISA, partly because its threshold is lower (0.15).
- **Green is near-solved everywhere (recall ≥ 0.94), and red mostly is (≥ 0.85, except A+B at LISA night, 0.77). Yellow is the weak color.**
  - The A model calls most yellow lights red (yellow recall 0.09–0.32).
  - B, A+B and DTLD vary from 0.18 to 0.93 by dataset.
  - BDD100K yellow is poor for every model (≤ 0.48).
  - DTLD is the most stable (0.48–0.93).
- **Night is the main detection failure.** The A model loses most of its LISA recall at night, and DTLD loses about 45% of its BDD AP (0.60 → 0.34). Small (10–20 px) LISA night lights separate the models most: A finds 1% of them, B 73%, A+B 46%, DTLD 37%.
- **IoU ≥ 0.5 understates LISA and BDD by 2–10×.** For DTLD on LISA it gives 0.064 against 0.750 with center matching. Use center AP when comparing across datasets, and IoU AP only within a box convention.

### Arrow direction (ATLAS, LISA)

The test covers annotated arrows that a run matches (center criterion) and calls relevant. It checks the run's predicted moves, without uturn and generic, against the pictogram's directions:
- **exact**: the moves equal the pictogram's directions;
- **contain**: the moves include them.

`gt_to_pred_moves` in the JSON counts each (arrow, moves) pair.

| Model | ATLAS exact / contain (n) | LISA day exact / contain (n) | LISA night exact / contain (n) |
|---|---|---|---|
| A | 0.19 / 0.59 (1,041) | 0.17 / 0.72 (2,065) | 0.00 / 0.08 (348) |
| B | 0.23 / 0.53 (897) | 0.01 / 0.80 (1,851) | 0.00 / 0.03 (577) |
| A+B | 0.18 / 0.48 (1,127) | 0.08 / 0.68 (1,994) | 0.00 / 0.01 (692) |
| DTLD | **0.43 / 0.84** (1,560) | **0.49** / 0.61 (1,743) | 0.00 / 1.00 (786) |

Findings:
- **The OpenLane models don't read the pictogram; they guess the lane set.**
  - A light's class is the set of maneuvers of the lanes it controls (HD-map topology), so a left arrow usually comes out as left+straight. Its color still applies to left.
  - In 40–50% of cases on ATLAS, the predicted moves don't even include the arrow's direction: left → straight, left → right+straight, right → straight. Those errors would put the arrow's color on the wrong direction.
- **The DTLD model is the only one with pictogram-based moves**, because a DTLD lamp's moves come from the route its arrow serves. It gets the exact moves 2–3× more often. Its main error is calling an arrow a circle (left → left+right+straight), which keeps the arrow's direction (contain 0.84 on ATLAS).
- **At night no model recognizes arrows.** DTLD calls 784 of 786 LISA night arrows a circle (left+right+straight). The OpenLane models give moves without left (A+B: generic 383, straight 164, right+straight 131 out of 692).
- Per light, moves are much weaker than color (color accuracy ≥ 0.9 against exact moves ≤ 0.5). This motivates a factorized pictogram head (shape: circle / left / straight / right / …) that can be trained on any dataset with pictograms (DTLD, ATLAS, LISA), separately from the relevance and moves heads.

## Pictogram + rules vs joint classes: pre-registered protocol (2026-11-06)

This section was written before any pictogram was annotated and before any new model was trained. Its hypotheses, metrics, data use and decision rules are fixed. Any later change will be added below as a dated amendment with its reason, never edited in place.

### Question

Does reading each light's pictogram and applying fixed traffic rules (with a learned fallback) predict per-direction signals better than the current joint `<moves>|<color>` classes, whose moves come from HD-map topology? "Better" has three parts:
- in-domain accuracy;
- transfer to other subsets and countries;
- per-light behavior that matches what the lamp shows.

### What is already known, without annotation (`tools/audit_pictogram_rules.py`)

These use ground-truth relevance and color; only each light's moves are replaced.

| | A val TSS | B val TSS |
|---|---|---|
| every light is a circle | 0.9465 | 0.9629 |
| every turn-only light is an arrow of its own directions (ceiling) | 0.9940 | 1.0000 |
| frames whose label depends on the pictogram | 189 / 2,220 (20 segments) | 114 / 1,565 (13 segments) |

The rest of A val (27 frames) cannot be fixed by turn-only arrows. Those frames have lights with different topology moves that all cover straight: lane-specific circle heads or straight arrows.

### H1: Rule oracle on OpenLane-V2 (annotation)

**Annotation sample.** Built by `tools/make_pictogram_pool.py` with seed 0 and fixed before labelling. Three strata:
- **D:** every relevant light in every dependent val frame (A 189, B 114). This stratum is complete, so its result is exact.
- **N:** a simple random sample of 150 of the other kept val frames, split in proportion to subset size (A 87 of 2,031, B 63 of 1,451), with all their relevant lights. It estimates how often a real arrow breaks a frame that the all-circle rule gets right.
- **QC:** 5% of the items are shown a second time, to measure the annotator's self-agreement.

**Blinding.** The annotator sees only the crop, a wider context crop, and the light's annotated color (needed to pick which lamp of a head is meant). The topology moves, the stratum, the subset and the frame's labels are hidden, and items are shuffled.

**Classes.** circle, arrow_left, arrow_straight, arrow_right, arrow_straight_left, arrow_straight_right, arrow_uturn, other (not a vehicle signal: pedestrian, bike, bus or tram), unclear.

**Rules.** US, `assign_scored` with `near_turn = dedicated`, mapping each pictogram to moves:
- circle → generic (governs straight, and left / U-turn by inheritance);
- each arrow → its own directions;
- other → light dropped;
- unclear → circle in the primary analysis. As a sensitivity check, frames with an unclear light are excluded instead.

**Primary metric.** Oracle TSS on A val and B val, estimated as:
- the exact result on stratum D, plus
- the N-sample mean, weighted by the size of the non-dependent part.

The 95% interval comes from a segment-cluster bootstrap of the N sample (10,000 draws, seed 0).

**Also reported:**
- the share of turn-only lights that are arrows;
- the oracle failures by type (circle-only dedicated head, lane-specific circles, T-junction, …);
- the QC self-agreement (Cohen's κ).

The oracle is an upper bound for a pictogram method with perfect perception. It is not a model result.

### H2–H4: Learned models

**Arms.**
- **J (joint):** the current RF-DETR joint classes. This is the baseline already run (tag `baseline-joint-classes`).
- **F (factorized hybrid):** RF-DETR with these heads:
  - light;
  - color (3-way);
  - relevance;
  - moves (multi-label: straight / left / right / U-turn / generic, as in J);
  - pictogram (the classes above).

  Decoding:
  - pictogram + rules where the pictogram head is confident and the light is an arrow;
  - the learned moves head for circles;
  - the learned moves head wherever the pictogram is unclear or below the confidence threshold.

  The confidence threshold is picked on the tune split.
- **J′ (control):** J's existing scores decoded through F's marginalization (no training). It separates the effect of the decoding from that of the new heads.

**Pictogram supervision.** DTLD train and the ATLAS train split. On OpenLane and DTLD-US frames the pictogram loss is masked. The OpenLane pictogram annotations are never used for training. ATLAS test and LISA stay pure out-of-distribution sets.

**Kept identical to J:**
- RF-DETR Medium at 1024, squeeze resize, 4 epochs, lr 1e-4 / encoder 1.5e-4, batch 4 × grad-accum 4;
- the same fit / tune splits;
- epoch and threshold picked on the source tune TSS only.

The val sets and the DTLD test set are never used to choose anything.

**Seeds.** 3 seeds per arm and training set (J will be re-run with 2 more seeds, so both arms have 3).

**Hypotheses and tests.** All are paired on the same frames: the difference of the seed-averaged per-frame scores between F and J, with a segment-cluster bootstrap 95% interval (10,000 draws).

| | Training | Test | Primary metric | F is "better" if |
|---|---|---|---|---|
| H2 in-domain | A+B | A val, B val | TSS | the CI of ΔTSS is above 0, or above −0.005 (non-inferior) with H3 or H4 met |
| H3 transfer | A, B, A+B | the other subset, D*_US test | TSS | the CI of ΔTSS is above 0 on at least 2 of the 4 transfer cells and below 0 on none |
| H4 per-light OOD | A+B, DTLD | ATLAS test, LISA test | exact arrow-moves accuracy (center matching) | the CI of Δ is above 0 |

**Secondary metrics:** False-Go (the safety metric; F must not increase it beyond the CI), Acc_color, EM, per-direction accuracy, and center-AP. Center-AP checks that the extra heads don't cost detection.

**Multiple comparisons.** H2–H4 are judged together. The paper claims an improvement only if H2 holds and at least one of H3 or H4 holds. Everything else is reported as exploratory.

**Reported whatever the outcome:**
- all cells, including the ones where F is worse;
- the oracle (H1) next to the learned results, to separate "the rules are limited" from "the pictogram head is not accurate enough";
- all seeds.

**Pool as built** (`tools/make_pictogram_pool.py`, seed 0, before labelling). It has 1,231 items:
- A: D 527, N 174;
- B: D 315, N 156;
- QC repeats: 59.

Overlapping boxes (IoU ≥ 0.8) are merged into one item only when they have the same color, because a head with a red ball and a green arrow lit can be two OpenLane elements.

The key `data/annotation/pictogram_v1/items.jsonl` has sha256 prefix `d4ddf7951e3fad36`. Labels go to `annotations/openlane_pictograms_v1.json` (tracked) and are scored by `tools/score_pictogram_oracle.py`, written and tested on synthetic labels before any real label existed:
- all-circle labels reproduce the "every light is a circle" row exactly (0.9465 / 0.9629);
- topology-derived arrows give 0.9895 / 0.9988;
- random labels give 0.75 / 0.82 with κ ≈ 0.

### H1 results: rule oracle (2026-11-06)

**Order of events.** These file timestamps (local time) show that the protocol, the pool and the scoring code all existed before the labels:
- protocol: 09:30
- pool: 09:29
- scoring code: 09:28
- labels exported: 09:50

The scoring ran unchanged as pre-registered. Labels: `annotations/openlane_pictograms_v1.json`, sha256 prefix `b6a485e0b9254c5c`. All 1,231 items are labelled. Full output: `results/pictogram_oracle_v1.json`.

| | A val | B val |
|---|---|---|
| every light a circle (no pictogram) | 0.9465 | 0.9629 |
| **pictogram + rules oracle (primary)** | **0.9667** (bootstrap CI degenerate: 0 / 87 N-frames fail) | **0.9589** [0.9380, 0.9675] |
| conservative bound: one-sided 95% upper bound on the N failure rate (exact binomial) | ≥ 0.9489 | ≥ 0.9317 |
| sensitivity: frames with an unclear light left out | 0.9658 | 0.9573 |
| ceiling: every turn-only light an arrow | 0.9940 | 1.0000 |
| topology labels (what J learns) | 1 | 1 |
| learned J model (A+B, for scale) | 0.905 | 0.918 |

**Labels.** circle 962, arrow_left 112, arrow_straight 41, other 39, unclear 17, arrow_uturn 1, out of 1,172 lights. No right or two-way arrows. QC self-agreement: 59 / 59 (κ = 1.0).

Findings:
- **Most dedicated turn signals in OpenLane-V2 are not arrows.** Only 28% of the lights whose lanes only turn are arrows on A, and 16% on B. Light instances linked only to left lanes: arrow_left 66 against circle 59 on A, 26 against 65 on B. None of those linked only to right lanes is an arrow (A: 78 circles; B: 68 circles).
  - The share of arrows grows with box height: 12% under 25 px, 16% at 25–40 px, 33% at 40–60 px, 39% at 60 px and above. Part of this may be arrows too small to read at OpenLane's resolution. Even at 60 px and above, 61% are circles.
  - So circle-only dedicated heads, which the pictogram cannot tell apart from the main signal, are the norm here. This is the US case predicted in the review.
- **The pictogram recovers only part of what topology gives.** On A it closes 43% of the gap between "all circles" and the ceiling (+0.020 TSS). On B it closes none: 0.9589 against 0.9629, inside the CI. That B point estimate comes from a single N frame (segment 11048) where a light labelled other was dropped, weighted 23× by the extrapolation.
- **Oracle failures** (dependent frames still wrong):

  | Cause | A | B |
  |---|---|---|
  | circle-only head on a left-turn lane | 39 | 40 |
  | circle-only head on a right-turn lane | 28 | 50 |
  | lane-specific heads | 25 | 2 |
  | T-junction light (no straight) | 20 | 2 |
  | light labelled other | 11 | 5 |
  | arrow disagreeing with topology | 6 | 0 |

**Implications for H2–H4.** The protocol is unchanged; this is the reading of H1.
- A pure pictogram + rules system is capped about 3–4 TSS points below the topology labels on OpenLane-V2. It cannot replace the learned moves there.
- The hybrid (F) is therefore the right design: rules for arrows, a learned moves head for circles.
- The cap (≈ 0.96) is well above today's learned models (≈ 0.91), so it doesn't limit F yet.
- In the random N frames, arrows are only 3 / 174 (A) and 5 / 156 (B) relevant lights. They are about 13% only in the sample as a whole, which is enriched with turn-only lights. Any in-domain gain from the pictogram head (H2) should therefore be small. If F helps, it should show mostly in transfer and OOD arrow reading (H3, H4), where datasets have more arrows and no topology.

### Amendment 1 (2026-11-06): two-stage arm J+P, before F

Written before any J+P code or result existed. Reason: OpenLane-V2 has no pictogram labels for training, and H1 showed that arrows are rare there. The cheapest clean test of "do pictogram rules help" changes only the decoding of the existing J models. F is built only if J+P shows a gain (decision rule below). Seeds are deferred: every J+P result below is single-seed and preliminary until they are run.

**P (pictogram classifier).**
- Model: ResNet-18 trained from scratch.
- Input: a square crop of side 1.25 × the box's long side, resized to 64 × 64, plus the light's color as a one-hot input. The color picks the lamp when a head has two lamps lit.
- 7 classes: circle, arrow_left, arrow_straight, arrow_right, arrow_straight_left, arrow_straight_right, other (DTLD pedestrian / bicycle / tram). Unknown pictograms are left out.
- Training data: DTLD train (fit sessions of `data/labels/tune/dtld_train_fit.jsonl`) and ATLAS train (85% of sessions). It uses ground-truth boxes, jittered in scale (0.8–1.5 per side) and shift (±15%), with color jitter. A horizontal flip swaps left and right.
- Model selection (epoch): macro accuracy on DTLD tune sessions plus the held-out ATLAS train sessions.
- It never sees OpenLane, LISA, ATLAS test or DTLD test.
- P's accuracy on the 1,172 hand-labelled OpenLane lights (ground-truth boxes and colors) is reported, never used.

**Decoding J+P.** Each J run keeps its own epoch and score threshold, picked on its source tune split.
- After threshold and NMS, every detection J calls relevant is cropped from the source image and given J's color.
- If P's top class is an arrow with probability ≥ τ, the detection's moves become the arrow's directions. Otherwise J's moves are kept. Colors are always J's, and "other" is not used.
- τ ∈ {0.5, 0.6, 0.7, 0.8, 0.9, 0.95, off} is picked on the same source tune split, by TSS. "off" reproduces J exactly, so J is nested in J+P.

**Matrix.** Rows: J models A, B, A+B, DTLD.
- Per-direction columns: A val, B val, D*_US test, with the same metrics as before.
- Arrow columns: exact arrow-moves accuracy (center matching) on ATLAS test and LISA test. ATLAS test is in-domain for P, since P trains on ATLAS train; LISA is the out-of-distribution arrow test.

**Comparison.**
- J vs J+P is paired on the same detections.
- 95% CIs of ΔTSS and ΔFalse-Go come from a segment-cluster bootstrap (10,000 draws, seed 0). For the arrow columns, the bootstrap is over images grouped by sequence or camera.

**Decision rule (build F or not).** F is built if all of the following hold:
1. At least one transfer cell has a ΔTSS CI above 0. The transfer cells are A→B, B→A, A→D*_US, B→D*_US and A+B→D*_US.
2. Or, instead of 1, the LISA arrow-accuracy gain has a CI above 0.
3. In-domain is non-inferior: on A→A val, B→B val and A+B on A val and B val, the lower ΔTSS CI bound is above −0.005.
4. No cell has a ΔFalse-Go point estimate above +0.01.

Otherwise F is not built, and the paper reports J+P as the test of the pictogram idea.

**Implementation notes for amendment 1** (recorded before any J+P result):
- **P's training crops** (`tools/make_pictogram_crops.py`): 77,784 for training and 22,836 for validation. Circles are capped at 15,000 and others at 8,000 per source; lamps with a long side under 6 px are left out. Evaluation crops: ATLAS test 6,156, LISA test 21,525 (circles capped at 15,000), and the 1,154 hand-labelled OpenLane lights. Unclear lights and the single U-turn arrow are left out, because neither is a P class.
- **ATLAS validation has only 2 straight-right arrows**, so that class's recall makes the macro accuracy used for epoch selection noisy.
- **Deviation:** the arrow-column CIs bootstrap over images, not over sequences or cameras. ATLAS test has 3 cameras and LISA test 4 sequences, too few clusters for a cluster bootstrap. Images from one sequence are correlated, so these CIs are too narrow, and the LISA criterion should be read with that in mind.
- **Tie-break for τ:** when several τ give the same source-tune TSS, the most conservative is taken (the largest, then off). In the smoke test, τ = 0.8 … off all tied.

### Amendment 1 results: J vs J+P (2026-11-06, single seed, preliminary)

P = `runs/pictogram_p/best.pth`, epoch 9 (sha256 prefix `2fd9aed0693abe75`). Outputs: `results/hybrid_v1/`.

**P on its own** (ground-truth boxes and colors):

| Set | Accuracy | Arrow recall | Arrow precision | Exact type on true arrows |
|---|---|---|---|---|
| val (DTLD + ATLAS, selection) | 0.856 | 0.948 | 0.873 | 0.893 |
| ATLAS test (in-domain for P) | 0.705 | 0.891 | 0.704 | 0.716 |
| OpenLane hand labels | 0.685 | 0.719 | 0.415 | 0.582 |
| LISA test | 0.515 | 0.261 | 0.287 | 0.158 |

On OpenLane, 144 of 962 circles are called arrows (87 of them arrow_straight), and 126 are called other. LISA's wider boxes take P far outside its training crops.

**τ chosen on each source tune split:**

| Run | τ | Effect |
|---|---|---|
| A | off | J+P = J everywhere |
| A+B | off | J+P = J everywhere |
| B | 0.8 | overrides applied |
| DTLD | 0.8 | overrides applied |

On the OpenLane tune splits, tune TSS falls as more lights are overridden: P's false arrows on US and Singapore circles cost more than its true arrows gain.

**Per-direction matrix** (ΔTSS = J+P − J, with the segment-cluster bootstrap 95% CI):

| Run → test | J TSS | J+P TSS | ΔTSS [95% CI] | ΔFalse-Go | frames changed |
|---|---|---|---|---|---|
| A → A val | 0.8995 | 0.8995 | 0 | 0 | 0 |
| A → B val | 0.7893 | 0.7893 | 0 | 0 | 0 |
| A → D*_US | 0.7545 | 0.7545 | 0 | 0 | 0 |
| B → A val | 0.8273 | 0.8255 | −0.0018 [−0.0054, +0.0006] | +0.0047 | 28 |
| B → B val | 0.9039 | 0.9020 | −0.0019 [−0.0081, +0.0025] | −0.0005 | 30 |
| **B → D*_US** | 0.7816 | **0.8131** | **+0.0315 [+0.0237, +0.0398]** | **−0.0353** | 987 |
| A+B → A val | 0.9048 | 0.9048 | 0 | 0 | 0 |
| A+B → B val | 0.9178 | 0.9178 | 0 | 0 | 0 |
| A+B → D*_US | 0.8273 | 0.8273 | 0 | 0 | 0 |
| DTLD → A val | 0.8658 | 0.8639 | −0.0019 [−0.0048, +0.0003] | +0.0033 | 13 |
| DTLD → B val | 0.8829 | 0.8834 | +0.0005 [−0.0017, +0.0030] | 0 | 21 |
| DTLD → D*_US | 0.9602 | 0.9641 | +0.0039 [+0.0010, +0.0071] | −0.0035 | 229 |

**Arrow directions** (exact moves on arrows the run calls relevant; image-bootstrap CI, too narrow, see the deviation above):

| Run | ATLAS test (in-domain for P) J → J+P | LISA test J → J+P |
|---|---|---|
| A | 0.188 → 0.188 (off) | 0.147 → 0.147 (off) |
| B | 0.230 → **0.877** (+0.648 [+0.611, +0.684]) | 0.005 → 0.090 (+0.085 [+0.074, +0.097]) |
| A+B | 0.183 → 0.183 (off) | 0.058 → 0.058 (off) |
| DTLD | 0.433 → **0.849** (+0.417 [+0.388, +0.445]) | 0.338 → 0.395 (+0.057 [+0.047, +0.067]) |

**Decision rule (amendment 1), applied as written:**
1. **Transfer gain: met.** B → D*_US has ΔTSS +0.0315 [+0.0237, +0.0398]. The other transfer cells are 0 (τ off) or have a CI that includes 0.
2. **LISA arrow gain: met** for B and DTLD, with the image-bootstrap caveat.
3. **In-domain non-inferiority: not met.** B → B val has a lower CI bound of −0.0081, below the −0.005 margin (point estimate −0.0019). A → A val and A+B on A and B val are exactly 0.
4. **No ΔFalse-Go above +0.01: met.** The largest is +0.0047.

**Verdict:** criterion 3 fails, so by the pre-registered rule **F is not built**. J+P is the reported test of the pictogram idea. It fails by a narrow margin, on one cell, with a single seed.

Reading:
- **The pictogram classifier switches itself off for the models that matter most.** For A and A+B, the source tune split rejects every override. J+P = J for the best model, so in-domain there is nothing to gain or lose.
- **Where τ is on, rules on arrows help out of domain and are neutral to slightly negative in domain.**
  - Out of domain, B → D*_US gains 3.2 TSS points and cuts False-Go by 3.5 points.
  - Exact arrow moves on ATLAS rise from 0.23 to 0.88 (B) and from 0.43 to 0.85 (DTLD).
  - In domain, the cost is small and its CI includes 0 (−0.2 points on B → A and B → B val).
- **The bottleneck is P's precision on OpenLane** (0.415): US and Singapore circles called arrows. A pictogram stage helps where arrows are common and look like its training crops (Germany). It does not help on OpenLane-V2, where H1 already showed arrows are rare and mostly not readable.

## Exp 5: mixed training A+B+D*_US (2026-11-06, design fixed before training)

**Question.** Does adding German data (DTLD with the OpenLane protocol, D*_US) close the zero-shot gap to Germany without hurting OpenLane-V2? Does the resulting model serve both US and German traffic rules, switched only at decoding time?

**Training data.** The OpenLane-V2 A+B fit split (14,977 kept frames) plus a random sample (seed 0) of 14,977 of the 23,526 kept D*_US fit frames, so both domains carry equal weight. LISA is not used.
- Classes are the union of the training classes: `<moves>|<color>` per light, plus `none|<color>` for non-relevant lights.
- D*_US per-light classes are the DTLD lamps' moves from `serving()`. They don't depend on the US-vs-German direction rule, so the same trained model can be decoded with either rule.
- The recipe is r1's: RF-DETR Medium at 1024, 4 epochs, batch 4 × grad-accum 4, lr 1e-4 / encoder 1.5e-4, single seed. Output: `runs/rfdetr_m_mix`.

**Selection.** Per epoch, the score threshold maximizes the equal-weight mean of TSS on the OpenLane tune split (A_tune + B_tune) and on the D*_US tune split. The reported epoch is the best on that mean. No test set is used to choose anything.

**Reported** (all at that threshold):
- A val and B val (OpenLane rules);
- D*_US test (DTLD test with US rules: right only by a dedicated signal);
- DTLD test with German rules (`dtld_test.jsonl`, `near_turn = signal`: the circle also governs the right turn). This is the "switch rules only" test.

**Compared with** A+B (r1) and the DTLD model, each scored on the same four sets at its own source threshold. Paired segment-cluster bootstrap (`tools/bootstrap_compare.py`). The comparison is descriptive, with a single seed and no decision rule attached.

### Exp 5 results (2026-11-06)

**Selection.** Epoch 1, threshold 0.25 (mean tune TSS 0.9364). By epoch, mean tune TSS was 0.9276 / 0.9364 / 0.9347 / 0.9318. The DTLD sets keep improving after epoch 1 (DTLD test German 0.950 → 0.956 at epoch 3), while A val drops (0.915 → 0.900).

**TSS, each model at its own source-selected epoch and threshold.** Differences are mix minus the other model, with paired segment-cluster bootstrap 95% CIs (10,000 draws).

| Test set | Mix | A+B (r1) | Mix − A+B | DTLD | Mix − DTLD |
|---|---|---|---|---|---|
| A val | 0.9148 | 0.9048 | +0.0100 [−0.0010, +0.0227] | 0.8658 | +0.0490 [+0.0235, +0.0778] |
| B val | 0.9206 | 0.9178 | +0.0028 [−0.0139, +0.0200] | 0.8829 | +0.0377 [+0.0017, +0.0822] |
| A+B pooled | 0.9172 | 0.9102 | +0.0070 [−0.0025, +0.0172] | 0.8729 | +0.0443 [+0.0225, +0.0690] |
| D*_US test (US rules) | 0.9523 | 0.8273 | +0.1250 [+0.1112, +0.1392] | 0.9602 | −0.0079 [−0.0116, −0.0044] |
| DTLD test (German rules) | 0.9497 | 0.5770 | +0.3727 [+0.3596, +0.3857] | 0.9584 | −0.0088 [−0.0127, −0.0049] |

**False-Go** (mix vs A+B vs DTLD):
- A val: 0.146 / 0.145 / 0.341
- B val: 0.092 / 0.098 / 0.166
- D*_US: 0.042 / 0.213 / 0.033
- DTLD German: 0.036 / 0.327 / 0.030

Mix vs A+B on A+B pooled: False-Go −0.002 [−0.029, +0.022], False-Stop −0.007 [−0.013, −0.001].

**Reading.**
- **OpenLane is not hurt by the German data.** Mix vs A+B is +0.007 TSS on A+B pooled (CI includes 0); False-Go is unchanged.
- **The German zero-shot gap closes almost entirely.** On D*_US the mixed model reaches 0.952, against 0.827 for A+B and 0.960 for the DTLD specialist.
- **A small residual cost on German data.** The mixed model is 0.008–0.009 below the DTLD specialist on both DTLD sets, a significant difference. False-Go is +0.006 to +0.008 higher. It saw half as many German frames, and the epoch was chosen on the mean of both domains.
- **Rule switching works for the mixed model.** The same weights score 0.952 with US rules and 0.950 with German rules. Right-turn color accuracy is 0.964 (US) vs 0.942 (German), and right False-Go is 0.057 under German rules.
- **Rule switching fails for the A+B model.** It scores 0.827 with US rules but 0.577 with German rules; right-turn color accuracy drops to 0.272 and right False-Go rises to 0.81. Its per-light moves are learned from OpenLane's lane-based labels, so its round lights rarely carry the moves the German rule needs to give the right turn to the circle.
- **Conclusion on rules.** Switching rules at decoding only works if the model learned per-light moves with the conventions of the region being decoded, which requires training data from that region. The rule switch is a decoding choice, not a substitute for in-region data.
- **Single seed.** The ±0.01 OpenLane differences are within seed noise until seeds are run.

Outputs:
- `runs/rfdetr_m_mix/`: `val_log.jsonl`, `val_log_mix.jsonl`, `per_direction_mix.json`, `bootstrap_vs_r1_m_default.json`, `bootstrap_vs_rfdetr_m_dtld.json`;
- `*_mix.*` files in `runs/tune_rfdetr/r1_m_default/` and `runs/rfdetr_m_dtld/`.

## Audit: junction lights + pictogram + color rule (2026-11-06, `tools/audit_junction_rule.py`)

**Question.** A candidate design detects every vehicle signal of the junction approach (not only the ego lane's), with its pictogram and color. An arrow sets its own maneuvers; every other maneuver takes the circles' color, and the HD map gives the maneuvers the lane allows. How often does a frame make this rule fail? Every frame with at least one colored junction light is counted, including the frames the benchmark drops.

**Failure cases:**
- circles of different colors;
- arrows for the same maneuver with different colors;
- an unreadable pictogram on a light whose color differs from the others;
- on OpenLane, an answer that contradicts the topology label.

**Pictogram sources.**
- OpenLane-V2 has pictograms only for the 1,231 annotated val lights.
- The other OpenLane frames with lights of more than one color fall into two groups:
  - **Benchmark conflict frames** (two equally specific lights governing the same maneuver with different colors). These are counted as failures (lane-specific heads).
  - **The rest.** Their failures are estimated from the failure rate of the labelled val frames in the same region.
- DTLD has its own pictograms. Junction lights there are the relevant lamps plus non-relevant front lamps of at least 0.6× the median relevant height. The relevant-only scope is given as a lower bound.

| Region | Frames with junction lights | One color only | Failing frames (estimate) [bounds] | Drives / sessions with a failing frame |
|---|---|---|---|---|
| OpenLane A (US, Argoverse 2) | 12,468 | 96.3% | 2.0% [1.2%, 3.2%] | 26 / 593 (4%) |
| OpenLane B Boston | 6,052 | 94.4% | 4.9% [4.9%, 5.4%] | 27 / 232 (12%) |
| OpenLane B Singapore | 2,808 | 91.8% | 8.2% [7.3%, 8.2%] | 23 / 123 (19%) |
| DTLD (Germany), junction lamps | 39,234 | 82.4% (arrows resolve another 13.6%) | 4.1% (circles 3.9%, arrows 0.2%) | 305 / 2,089 (15%) |
| DTLD, relevant lamps only | 39,070 | 97.6% (arrows 0.6%) | 1.8% | – |

**How to read it.**
- **US and Singapore.** Almost all multi-color frames are circles disagreeing: lane-specific circle heads, and circle-only turn heads. Arrows separate the colors in only 60 labelled A frames and 13 Boston frames. No labelled Singapore frame is resolved by arrows.
- **Germany (DTLD).** Arrows do the job in 13.6% of frames, as expected. Circle disagreements remain in 3.9%:
  - red+green 1,133;
  - green+yellow 192;
  - red+yellow 180;
  - all three 14.

  Pairs involving yellow are mostly heads changing state a moment apart. The junction scope depends on the size heuristic, so this needs a visual check.
- **Frames vs drives.** Frames are consecutive, so a single junction contributes many failing frames. Per drive, the rule meets at least one failing situation in 4% of Argoverse drives, 12% of Boston drives, 19% of Singapore drives and 15% of DTLD sessions.

Output: `results/junction_rule_audit.json`, which includes example images per failure type.

## Exp 6: junction lights with pictogram, relevance and color on DTLD, joint classes vs factorized head (2026-11-06, design fixed before any code or result)

**Question.** Each model detects every traffic light, with its pictogram, its color and whether it belongs to the junction the ego is approaching. Fixed rules then turn these into per-maneuver signals. Does a factorized head (one light class plus separate pictogram, relevance and color outputs) beat joint `<pictogram>|<relevance>|<color>` classes when everything else is the same?

**Data.** DTLD only (Germany), every frame of the existing session splits:
- fit: 25,635 frames, 1,330 sessions;
- tune: 2,890 frames, 148 sessions;
- test (official): 12,453 frames, 632 sessions.

**Boxes.** Every front-facing DTLD lamp with a visible color (red_yellow counts as red). Off and unknown-state lamps and lamps not facing the camera are left out, as in the existing DTLD baseline.
- **Pictogram:** circle, arrow_left, arrow_right, arrow_straight, arrow_straight_left, unknown (DTLD could not tell), or other (pedestrian, bicycle, pedestrian_bicycle and tram signals).
- **Relevance, junction or not:** the automatic label of `dtsu/labels_dtld.vehicle_group`. A lamp is junction-relevant when it is a route lamp (DTLD relevant), or another front-facing vehicle lamp at least 0.6× the median route-lamp height in that frame. Non-vehicle pictograms are never junction lamps.
  - Known limitation, accepted: this label sometimes includes lamps of the next junction, and DTLD's route flag itself sometimes covers them (manual review of 50 failing frames, see the audit above).
- **Color:** red, yellow or green.

**Rule (same for ground truth and predictions).** The rule uses only the junction lamps.
- An arrow sets the color of its own maneuvers: straight, left, right, or straight and left.
- Every maneuver without an arrow takes the circle color. German convention: the circle also governs the right turn. U-turn follows left.
- unknown pictograms count as circles.
- Predictions: per maneuver, the highest-scoring candidate decides (arrows first, then circles).
- Ground truth: a frame is scored when it has at least one junction lamp and the rule gives one answer. That excludes frames where junction circles disagree, arrows of one maneuver disagree, or an unknown pictogram disagrees (as `tools/audit_junction_rule.py` counts them). Excluded frames are reported.
- The HD map's allowed maneuvers are not available, so all four maneuvers are scored.

**Arms.** Both use RF-DETR Medium at 1024, squeeze resize, the recipe of the DTLD baseline (lr 1e-4, encoder 1.5e-4, batch 4 × grad-accum 4, no flips or crops, photometric jitter), 6 epochs, seed 0. They differ only in the head:
- **J (joint):** one class per `<pictogram>|<relevance>|<color>` seen in fit. Score = the top class probability.
- **F (factorized):** one class "light" (IA-BCE, as RF-DETR's class loss), plus three linear heads on the same decoder features:
  - pictogram (7-way cross-entropy);
  - color (3-way cross-entropy);
  - junction relevance (binary cross-entropy).

  The attribute losses are computed on the Hungarian-matched queries of every decoder layer, weight 1 each. Matching uses the light score and the boxes. Score = light probability, attributes = argmax.

**Selection.** For each arm, the epoch and score threshold (grid 0.05–0.95) with the best TSS on tune. Class-agnostic NMS 0.5. Test is never used to choose anything.

**Metrics on DTLD test.**
- **Primary:** TSS of the rule's per-maneuver colors on the scored frames, with False-Go next to it.
- **Per light**, at the selected threshold, with IoU ≥ 0.5 matching:
  - detection recall and precision;
  - on matched lights: pictogram accuracy (overall, and recall for each arrow class), color accuracy, junction-relevance accuracy and F1, and all three right together.
- **Secondary:** arrow-direction accuracy on ATLAS test and LISA test, the out-of-domain arrow check of Amendment 1.

**Comparison.** F − J, paired on the same frames, with a session-cluster bootstrap 95% CI (10,000 draws, seed 0).
- F is "better" if the TSS CI is above 0 and False-Go does not rise by more than 0.01.
- Per-light differences are reported with their CIs as secondary.

Single seed, so every result is preliminary until seeds are added.

**Implementation note (2026-11-06, before training, no result seen).** F's light logit (decoder and two-stage encoder heads) starts from the pretrained COCO "traffic light" row of RF-DETR's class head. A plain resize to one class keeps row 0, an unused COCO id with a very negative bias. A 20-step smoke run then had no score above 0.002, while the untrained F with the traffic-light row scores DTLD lights at about 0.8. J keeps RF-DETR's own resize: rows 0–29 of the pretrained head, one per joint class. The attribute heads start from PyTorch's default init.


### Exp 6 results (2026-11-08, single seed)

Runs: `runs/exp6_picto_J`, `runs/exp6_picto_F` (`tools/run_exp6.sh`). Scoring: `tools/eval_picto.py`, comparison: `tools/compare_picto.py` → `runs/exp6_picto_F/compare_vs_J.json`.

**Selection on tune.**

| Arm | Tune TSS by epoch (0–5) | Selected epoch / threshold | Tune TSS |
|---|---|---|---|
| J | 0.943, 0.944, **0.945**, 0.941, 0.943, 0.940 | 2 / 0.30 | 0.9450 |
| F | 0.940, **0.946**, 0.941, 0.943, 0.941, 0.941 | 1 / 0.40 | 0.9458 |

Both arms peak in the first three epochs and then drift down slightly, so 6 epochs was enough.

**DTLD test.** 11,494 scored frames of 12,453; 57,093 labelled lights. F − J with the session-bootstrap 95% CI.

| Metric | J | F | F − J |
|---|---|---|---|
| **Rule TSS** | **0.9510** | 0.9463 | **−0.0047 [−0.0083, −0.0013]** |
| False-Go | 0.0331 | 0.0379 | +0.0049 [+0.0001, +0.0099] |
| False-Stop | 0.0219 | 0.0213 | |
| Exact match (all four maneuvers) | 0.9216 | 0.9133 | |
| Light recall (IoU ≥ 0.5) | 0.7346 | 0.6884 | −0.0462 [−0.0496, −0.0427] |
| Light precision | 0.8336 | 0.8715 | +0.0379 [+0.0336, +0.0421] |
| Pictogram accuracy (matched) | 0.9195 | 0.9218 | +0.0023 [−0.0012, +0.0058] |
| Color accuracy (matched) | 0.9777 | 0.9829 | +0.0051 [+0.0034, +0.0070] |
| Junction accuracy (matched) | 0.9606 | 0.9656 | +0.0051 [+0.0021, +0.0080] |
| All three right (matched) | 0.8834 | 0.8890 | +0.0056 [+0.0013, +0.0099] |
| arrow_left recall (n ≈ 4,300) | 0.836 | 0.845 | +0.009 [−0.009, +0.027] |
| arrow_right recall (n ≈ 510) | 0.295 | 0.396 | +0.102 [+0.029, +0.183] |
| arrow_straight recall (n ≈ 2,000) | 0.584 | 0.575 | −0.009 [−0.041, +0.022] |

**Verdict by the pre-registered rule: F is not better.** The TSS CI lies entirely below 0: the factorized head is slightly worse on the rule (−0.5 points), and False-Go rises by 0.5 points (within the 0.01 margin). The head alone brings no improvement on DTLD.

**Post-hoc check of the per-light rows (not pre-registered).** The arms run at different thresholds (0.30 vs 0.40), so their matched lights differ: F matches fewer, easier lights. That inflates F's attribute accuracies and precision. Two checks remove this:
- **Lights matched by both arms** (38,487), paired, session bootstrap:
  - pictogram: J 0.927, F 0.924, F − J −0.0025 [−0.0055, +0.0005];
  - color: 0.983 vs 0.985, +0.0014 [+0.0001, +0.0027];
  - junction: 0.967 vs 0.967, +0.0006 [−0.0015, +0.0029];
  - all three: 0.897 vs 0.893, −0.0034 [−0.0070, +0.0002].
- **Same threshold for both arms:**

  | Threshold | Arm | TSS | False-Go | Recall | Precision | Pictogram | Color | Junction | All three |
  |---|---|---|---|---|---|---|---|---|---|
  | 0.30 | J | 0.9510 | 0.0331 | 0.735 | 0.834 | 0.920 | 0.978 | 0.961 | 0.883 |
  | 0.30 | F | 0.9461 | 0.0365 | 0.717 | 0.800 | 0.917 | 0.979 | 0.963 | 0.880 |
  | 0.40 | J | 0.9518 | 0.0336 | 0.700 | 0.890 | 0.929 | 0.984 | 0.967 | 0.899 |
  | 0.40 | F | 0.9463 | 0.0379 | 0.688 | 0.872 | 0.922 | 0.983 | 0.966 | 0.889 |

So the per-light gains of F in the main table are an operating-point effect. At equal thresholds J detects more lights with higher precision, and the attributes of the lights both arms find are equally good. The one difference that survives is arrow_right (494 lights matched by both: J 0.30, F 0.40), a rare class where sharing the "arrow_right" signal across colors and relevance seems to help. arrow_straight goes the other way (0.61 vs 0.58).

**Reading.**
- With DTLD's amount of data, the 30 joint classes are not starved: most combinations have thousands of boxes. Factorizing the head therefore has little to gain, and the single light class loses some detection quality.
- The rule TSS is about 0.95 for both. On the lights they find, attributes are right ≥ 92% of the time, so detection (recall about 0.70–0.73 at IoU 0.5) is the larger lever. This breakdown of rule errors was not measured.
- Arrow pictograms other than left remain weak for both: right 0.3–0.4, straight about 0.58.
- Single seed: a −0.5-point TSS difference may not survive more seeds.

**Secondary: arrow directions out of domain** (`tools/eval_picto_arrows.py` → `runs/exp6_picto_F/arrows_vs_J.json`). Each arm runs at its selected epoch and threshold. The check covers every annotated arrow with a color that a detection matches under the center criterion. "Exact" means the predicted pictogram's moves equal the arrow's directions; a circle or "other" counts as wrong. The bootstrap is over images, too narrow for the same reason as in amendment 1. Both sets are out of domain for both arms.

| Set | Arrows (recall J / F) | All matched: J → F exact | Called junction: J → F exact |
|---|---|---|---|
| ATLAS test | 2,753 (0.66 / 0.61) | 0.383 → 0.401, +0.017 [−0.000, +0.035] | 0.458 → 0.453, −0.005 [−0.027, +0.016] |
| LISA test | 6,525 (0.75 / 0.75) | 0.153 → **0.259**, **+0.106 [+0.096, +0.116]** | 0.221 → 0.297, +0.075 [+0.063, +0.087] |
| LISA day | | 0.203 → **0.343**, +0.139 [+0.127, +0.152] | |
| LISA night | | 0.029 → 0.047, +0.018 [+0.004, +0.031] | |

- **ATLAS: no difference.** Both read about half of the left arrows they match as arrow_left. The rest are mostly missed or called circles.
- **LISA: F reads arrows better, and its mistakes are safer.** LISA's arrows are almost all left arrows.
  - J scatters them: circle 1,819, other 1,163, arrow_straight 1,030, arrow_left 704.
  - F: arrow_left 1,230, circle 3,023, other 517, arrow_straight 27.
  - J's arrow_straight confusions would put a left-turn color on the straight maneuver. F almost never makes that error and falls back to "circle" instead.
- **Not comparable with the amendment 1 table:** that one scores the moves of arrows a run calls route-relevant, with a different model and threshold.

**Overall.**
- On DTLD, in domain, the factorized head is not better. By the pre-registered primary it is slightly worse: TSS −0.5 points, CI below 0.
- Out of domain on LISA it reads left arrows better (+10 points exact, CI too narrow) and confuses them with straight arrows far less.
- The head alone is therefore not an improvement on the main metric. Its possible benefit is robustness of the pictogram outside DTLD, which needs seeds (and a cluster-level test) before it can be claimed.

**OOD perception of the Exp 6 arms** (`tools/eval_picto_perception.py`, same metrics as the OOD perception table above; each arm at its DTLD-tune epoch and threshold, J e2 / 0.30, F e1 / 0.40). Center AP (IoU ≥ 0.5 AP in brackets), recall / precision at the threshold, color accuracy and yellow recall of matched lights (center matching). Arrows: exact moves on matched arrows the arm assigns to the junction.

| Set | Arm | AP center (IoU) | R / P | Color / yellow R | Arrow exact (n) |
|---|---|---|---|---|---|
| ATLAS test | J | 0.613 (0.525) | 0.69 / 0.55 | 0.953 / 0.87 | 0.458 (1,448) |
| ATLAS test | F | 0.630 (0.531) | 0.64 / 0.64 | 0.957 / 0.95 | 0.453 (1,480) |
| LISA test | J | 0.736 (0.078) | 0.72 / 0.72 | 0.992 / 0.60 | 0.221 (3,060) |
| LISA test | F | 0.747 (0.075) | 0.65 / 0.79 | 0.997 / 0.86 | 0.297 (4,139) |
| LISA day | J / F | 0.836 / 0.841 | 0.90 / 0.75 · 0.90 / 0.79 | 0.996 / 0.999 | 0.306 / 0.395 |
| LISA night | J / F | 0.706 / 0.716 | 0.64 / 0.70 · 0.53 / 0.79 | 0.989 / 0.996 | 0.008 / 0.053 |

- **Detection is close for the two arms** (AP within 0.02). F trades recall for precision at its higher threshold.
- **Against the earlier DTLD maneuver model** (center AP 0.697 ATLAS, 0.750 LISA):
  - ATLAS detection is lower for both Exp 6 arms (0.61–0.63); LISA is the same.
  - Color is better: ATLAS 0.95 vs 0.90; LISA 0.99 vs 0.985.
- **F's yellow recall is higher on both sets** (ATLAS 0.95 vs 0.87, LISA 0.86 vs 0.60), the one clear per-light advantage of the factorized color head out of domain.
- Night arrows on LISA are essentially unreadable for both models.

Review page: `tools/make_picto_gallery.py` → `data/review_picto` (100 images: 34 DTLD test, half right and half wrong by the label, plus 33 ATLAS and 33 LISA, half with an annotated arrow; J model).

**Exp 6 models on OpenLane-V2 A val and B val** (`tools/eval_picto_openlane.py`; trained on DTLD only, each arm at its DTLD-tune epoch and threshold). The junction rule's per-maneuver colors are scored against the benchmark labels two ways:
- **all four directions as predicted:** unfair to the rule, which colors every maneuver when it sees a junction light, while the label says "none" for maneuvers no light governs;
- **map-masked:** directions labelled "none" are set to "none", i.e. the HD map's allowed maneuvers are assumed known, as in the junction-rule audit. This leaks a little label information.

U-turn follows left. The benchmark records mark every B val frame as right-hand traffic, so Singapore's U-turn is also tied to left.

| Model | A val TSS all / masked | A val False-Go | B val TSS all / masked | B val False-Go |
|---|---|---|---|---|
| Exp 6 J | 0.496 / **0.848** | 0.308 | 0.508 / **0.886** | 0.198 |
| Exp 6 F | 0.496 / 0.843 | 0.312 | 0.501 / 0.868 | 0.222 |
| reference: DTLD maneuver model (Exp 2 J, its own decoder) | 0.866 | | 0.883 | |
| reference: A+B (in domain) | 0.905 | | 0.918 | |

- **Even with the map mask, the DTLD-only junction model is below the in-domain OpenLane model.**
  - On A val (Argoverse, US) it is also below the earlier DTLD maneuver model.
  - On B val (nuScenes) it is equal to it.
- **False-Go is high**, mostly because red lights are missed.
  - J on A val: 650 red-labelled directions predicted as go. Of these, 383 had no junction light found ("none"), 134 were predicted green and 133 yellow.
  - B val: 194 / 100 / 102.
  - This is a detection and relevance transfer problem (portrait Argoverse frames, US signal heads), not a rule problem: the rule oracle with ground-truth lights reaches 0.95–0.96 TSS.
- **F is slightly below J on both sets**, consistent with DTLD.

**AWSIM video** (`tools/run_video_picto.py`, J at 0.30): `runs/awsim/AWSIM_only_picto_J.mp4`, with a 4-frame sample sheet next to it.
- Junction lights are found in 2,509 of 7,076 frames; the mixed model found relevant lights in 2,650.
- Straight is red in 223 frames, against 61 for the mixed model.
- Almost every detection is a circle. Some wrong arrows appear: one false "left arrow" on a pole sets straight and right to none for a moment.

## Exp 7: one factorized model on DTLD + ATLAS + OpenLane-V2 with partial labels, against the moves model (2026-11-07, design fixed before any code or result)

**Question.** Exp 6 showed that a factorized head brings nothing when every label is present. Its real advantage is that each attribute loss can be switched off where a dataset lacks that label, so one model can learn from datasets that each label only part of the target. Does a factorized pictogram / junction / color model trained this way beat the moves model on transfer while staying close in domain? If yes, it becomes the thesis's main result.

**Training data (F7).** Every source keeps its own fit split; LISA, S2TLD and BDD100K stay test-only.

| Source | Frames | Pictogram | Junction | Color |
|---|---|---|---|---|
| DTLD fit (Exp 6 export, all frames) | 25,635 | yes | yes (automatic label) | yes |
| ATLAS train, 85% of sessions (seed 0, the same split as the pictogram crops of amendment 1) | about 25,700 | yes | **masked** | yes |
| OpenLane-V2 A fit + B fit (kept frames, as the moves models) | 14,977 | **masked** | yes: the label builder's "relevant" (linked to a lane of the ego approach) | yes |

- **Boxes.** Lamps with a red, yellow or green state (red_yellow → red). ATLAS "off" lamps are left out, as DTLD's are.
  - ATLAS arrow_straight_right becomes an eighth pictogram class: the rule gives it straight and right.
  - Known label conflict, accepted: only DTLD labels pedestrian, bike and tram signals (as "other"). On ATLAS and OpenLane they are unlabelled background.
- **Model.** RF-DETR Medium at 1024, the Exp 6 factorized head with the pictogram head widened to 8, light logit from the COCO traffic-light row.
  - A missing attribute is stored as −1 and drops out of that loss (cross-entropy ignore index, junction BCE masked).
  - Detection and color losses use every box.
  - Recipe as Exp 6 (lr 1e-4 / 1.5e-4, batch 4 × accum 4, seed 0), 5 epochs (Exp 6 peaked at epochs 1–2, and the set is 2.6× larger).

**Comparison model (M).** The Exp 5 mixed moves model `runs/rfdetr_m_mix` (OpenLane A+B fit + a D*_US fit sample; epoch 1, threshold 0.25, as already selected). It is the moves model trained on the closest data. M is not retrained.

**Selection of F7.** Epoch and threshold that maximise the mean of three tune TSS values:
- DTLD tune (junction rule);
- A tune (map-masked);
- B tune (map-masked).

ATLAS's held-out sessions have no target and are not used. Test sets are never used to choose anything.

**Decoding.**
- F7 uses the junction rule: arrows set their own maneuvers, the circle sets the rest including right, and U-turn follows left.
- M uses its own decoder (`assign_scored`): US rule ("dedicated") on OpenLane, and German ("signal") on DTLD test.

**In-domain endpoints**, with paired F7 − M differences and a segment / session cluster bootstrap 95% CI (10,000 draws, seed 0):
1. **A val and B val, map-masked TSS for both models.** Directions labelled "none" are set to "none" in both predictions, i.e. the HD map's allowed maneuvers are assumed known. This puts the two decoders on the same footing; M's unmasked native TSS is reported as a reference.
2. **DTLD test, junction-rule ground truth** (the 11,494 scored frames of Exp 6), TSS for both. M was trained on route relevance (D*_US), a narrower target than the junction label, so this cell is slightly against M; that is recorded here.
3. False-Go on the same cells.

**Transfer endpoints** (out of domain for both models):
4. **LISA test arrow directions.** Exact moves on matched arrows (center criterion) that the model calls relevant or junction: M's moves against F7's pictogram moves. Image bootstrap: LISA has 4 sequences, so the CI is too narrow, as in amendment 1.
5. **Detection, center AP** on LISA test, S2TLD and BDD100K val. Color accuracy of matched lights is secondary.
6. ATLAS test is reported but is not a transfer cell, because ATLAS train is in F7's training data. The AWSIM video is qualitative.

**Decision rule.** F7 is "the stronger thesis result" when all of these hold:
- **(a) In domain is non-inferior:** on each of A val, B val and DTLD test, the lower CI bound of ΔTSS is above −0.01.
- **(b) False-Go does not rise:** no ΔFalse-Go above +0.01 in those cells.
- **(c) Transfer gains:**
  - the LISA arrow-exact ΔCI is above 0;
  - the mean center AP over LISA, S2TLD and BDD100K is not lower than M's by more than 0.02.

Otherwise the moves model stays the main result, and F7 is reported as the generalization study.

Single seed for both, so everything is preliminary until seeds are added.

### Exp 7 results (2026-11-07, single seed, `tools/eval_exp7.py` → `runs/exp7_multi_F/exp7_eval/result.json`)

**Training.** 67,140 images (DTLD 25,635, ATLAS 26,528, OpenLane A 8,956, B 6,021), 217,158 boxes. The pictogram is labelled on 166,263 boxes and the junction on 167,936. 5 epochs ran without error in about 8 h. RF-DETR's own validation mAP stays near zero, as for Exp 6 F; it does not read the factorized head.

**Selection.** The mean tune TSS rose to the last epoch, which was selected: epoch 4, threshold 0.20 (DTLD tune 0.943, A tune 0.951, B tune 0.960, mean 0.951).

| Epoch | Best threshold | DTLD tune | A tune | B tune | Mean |
|---|---|---|---|---|---|
| 0 | 0.25 | 0.941 | 0.916 | 0.937 | 0.931 |
| 1 | 0.40 | 0.941 | 0.938 | 0.960 | 0.946 |
| 2 | 0.30 | 0.940 | 0.942 | 0.952 | 0.945 |
| 3 | 0.25 | 0.944 | 0.948 | 0.951 | 0.948 |
| **4** | **0.20** | 0.943 | 0.951 | 0.960 | **0.951** |

**In domain** (paired F7 − M, cluster bootstrap 95% CI):

| Set | Frames | F7 TSS | M TSS | ΔTSS [CI] | F7 False-Go | M False-Go | ΔFalse-Go [CI] |
|---|---|---|---|---|---|---|---|
| A val (map-masked) | 2,220 | 0.934 | 0.928 | +0.005 [−0.008, +0.018] | 0.138 | 0.146 | −0.008 [−0.053, +0.033] |
| B val (map-masked) | 1,565 | 0.945 | 0.931 | +0.014 [−0.009, +0.041] | 0.072 | 0.092 | −0.019 [−0.053, +0.017] |
| DTLD test (junction GT) | 11,494 | 0.951 | 0.933 | **+0.018 [+0.013, +0.024]** | 0.028 | 0.040 | **−0.011 [−0.017, −0.006]** |

M's unmasked native TSS is 0.915 on A val and 0.921 on B val. F7 on DTLD test (0.951) equals the DTLD-only Exp 6 J model (0.951), so adding ATLAS and OpenLane cost nothing on DTLD. The DTLD cell is slightly against M by design (see the protocol).

**Transfer.**

| Set | F7 center AP | M center AP | F7 color acc. | M color acc. | F7 yellow recall | M yellow recall |
|---|---|---|---|---|---|---|
| LISA test | **0.663** | 0.597 | **0.995** | 0.903 | 0.76 | 0.90 |
| S2TLD | **0.844** | 0.741 | 0.992 | 0.981 | 0.83 | 0.83 |
| BDD100K val | **0.669** | 0.641 | 0.968 | 0.967 | 0.34 | 0.28 |
| *ATLAS test (in domain for F7)* | *0.769* | *0.642* | *0.983* | *0.974* | *0.79* | *0.71* |

Mean center AP over LISA, S2TLD and BDD100K: F7 0.725, M 0.660 (+0.065). M's LISA color accuracy is pulled down by red recall (0.81); F7's red recall is 0.998.

**Arrow directions** (matched arrows the model calls junction/relevant; exact moves; image bootstrap):

| Set | F7 exact (n) | M exact (n) | Δ [CI] |
|---|---|---|---|
| LISA test | 0.234 (3,786) | **0.466** (3,198) | **−0.231 [−0.252, −0.211]** |
| *ATLAS test (in domain for F7)* | ***0.663*** (1,749) | 0.421 (1,486) | +0.243 [+0.211, +0.274] |

Why F7 loses on LISA (exploratory tally of the cached detections). LISA's arrows are almost all left arrows (5,121 of 5,226 matched).
- **F7 reads most as circles:** circle-junction 2,743 (54%), circle-non-junction 492, arrow_left-junction 844, arrow_left-non-junction 493, "other" 420.
- **M answers "left" directly:** left|red 1,052, left|green 393, left+straight 736. It also calls 1,982 of them irrelevant, which leaves them out of its denominator.

This is the Exp 6 F failure (LISA left arrows → circle) again, and ATLAS training did not remove it, although ATLAS arrows are now read well. A likely cause, not tested: LISA's small, dim US left arrows sit in multi-lamp vertical heads, unlike ATLAS's German arrow lamps. M sidesteps the pictogram by predicting the maneuver, which the left-turn bay geometry of US OpenLane data supports.

**Decision (pre-registered rule).**

| Criterion | Result |
|---|---|
| (a) in domain non-inferior (lower CI > −0.01 on A val, B val, DTLD test) | **met**: −0.008, −0.009, +0.013 |
| (b) no False-Go rise above +0.01 | **met**: all three differences negative |
| (c1) LISA arrow-exact CI above 0 | **not met**: −0.231 [−0.252, −0.211] |
| (c2) mean OOD center AP ≥ M − 0.02 | **met**: 0.725 vs 0.660 |

**F7 is not "the stronger thesis result" under the pre-registered rule**, because it fails the LISA arrow criterion, and fails it clearly. The moves model stays the main result. F7 is reported as the generalization study. Its result is strong but split:
- **Where F7 is better:**
  - in domain, it is at least as good as M everywhere and better on DTLD test;
  - it has lower False-Go;
  - it detects lights better out of domain (+0.065 mean center AP) with better color.
- **Where F7 is worse:** reading US left arrows, the one transfer cell that tests the pictogram itself.

The pictogram route generalizes perception, not arrow reading across countries. The LISA-arrow and A/B-val cells are single seed, with narrow or wide CIs respectively, as noted.

**AWSIM video (qualitative).** Produced with `tools/run_video_picto.py --run runs/exp7_multi_F` at 0.20 and saved as `runs/awsim/AWSIM_only_picto_F7.mp4`, with a sample sheet beside it. Compared with Exp 6 J:
- F7 marks junction lights in more frames: 3,370 of 7,076 vs 2,509.
- It reports straight red in fewer frames: 144 vs 223.
- It reads more right arrows: 187 detections vs 46. Some are weak (0.28–0.34) on lamps that look round.

### Exp 7 follow-up checks (2026-11-09, exploratory, no training)

These checks were chosen after the Exp 7 results were known, so they do not change the decision above. F7 is epoch 4 at 0.20, M is epoch 1 at 0.25. Code: `tools/exp7_checks.py`, outputs `runs/exp7_multi_F/exp7_eval/checks_*.json`. VZC: `tools/eval_vzc.py`, output `runs/vzc_eval/result.json`.

**DTLD test on route labels** (the target M trained on, German rule; 11,126 frames in both the route labels and the junction-scored set). TSS: F7 0.9519, M 0.9522, difference −0.0002 [−0.0055, +0.0051]. False-Go 0.030 vs 0.034, difference −0.004 [−0.011, +0.003]. The two models tie, so the Exp 7 DTLD gain of +0.018 came from scoring on F7's own target.

**OpenLane without using the labels.** F7 is decoded with the US rule (right only from a right arrow, left falls back on straight, as M's decoder). The optional mask comes only from lane topology: a maneuver is blanked when no lane of the ego approach at the nearest junction leads to it, keeping the benchmark's inheritance. It uses no traffic light. No junction lies ahead in 476 of the 2,220 A val frames; those are not masked. Threshold 0.20; the tune-selected 0.25 changes these by at most 0.002.

| Set | F7 label mask (Exp 7) | F7 US rule | F7 US rule + topology | M native | M + topology | F7 US rule + topology − M native |
|---|---|---|---|---|---|---|
| A val | 0.934 | 0.918 | 0.926 | 0.915 | 0.921 | +0.011 [−0.004, +0.028] |
| B val | 0.945 | 0.895 | 0.901 | 0.921 | 0.919 | −0.019 [−0.050, +0.007] |

Against the topology mask, the label mask added 0.8 point on A val and 4.4 points on B val. Without it, F7 is not non-inferior on B val by the Exp 7 margin (lower bound −0.050, margin −0.01); A val still passes.

**Center AP, F7 − M** (2,000 draws; LISA in blocks of 100 consecutive frames, S2TLD and BDD100K by image):

| Set | F7 | M | Difference [CI] |
|---|---|---|---|
| LISA test | 0.663 | 0.597 | +0.066 [+0.034, +0.097] |
| S2TLD | 0.844 | 0.741 | +0.103 [+0.096, +0.110] |
| BDD100K val | 0.669 | 0.641 | +0.028 [+0.021, +0.036] |
| Mean of the three | 0.725 | 0.660 | +0.066 [+0.055, +0.076] |

**LISA arrows, F7 − M** (block bootstrap, 10,000 draws). "Exact" requires the light to be called relevant. "Reading" ignores the relevance call.

| Measure | All: F7 / M, difference [CI] | Day: F7 / M, difference [CI] | Night: F7 / M, difference [CI] |
|---|---|---|---|
| Exact, of arrows called relevant (Exp 7 criterion) | 0.234 / 0.466, −0.231 [−0.351, −0.088] | 0.241 / 0.615, −0.373 [−0.487, −0.236] | 0.205 / 0.004, +0.201 [+0.076, +0.393] |
| Exact, of matched arrows | 0.170 / 0.273, −0.104 [−0.186, −0.012] | 0.208 / 0.422, −0.214 [−0.309, −0.103] | 0.089 / 0.002, +0.087 [+0.035, +0.151] |
| Reading, of matched arrows | 0.264 / 0.273, −0.009 [−0.119, +0.112] | 0.256 / 0.422, −0.166 [−0.278, −0.037] | 0.281 / 0.002, +0.279 [+0.145, +0.424] |

The Exp 7 criterion still fails with the wider CI. M's advantage is all in daytime. M reads almost no arrows at night (3 exact of 781 called relevant). F7's pictogram head only saw German arrow labels, while M trained on US scenes, so this comparison mixes architecture with data.

**False-Go causes** (label red, prediction not red, first cause that applies; share of each model's False-Go directions):

| Set | Model | False-Go directions | Red not found | Found, called not relevant | Relevant, wrong color | Relevant and red, decoder chose another light |
|---|---|---|---|---|---|---|
| A val | F7 | 292 | 28% | 18% | 17% | 37% |
| A val | M | 309 | 35% | 21% | 36% | 7% |
| B val | F7 | 145 | 16% | 65% | 13% | 6% |
| B val | M | 184 | 27% | 45% | 10% | 19% |
| DTLD test (junction labels) | F7 | 447 | 21% | 19% | 7% | 52% |
| DTLD test (junction labels) | M | 623 | 27% | 14% | 7% | 52% |

Decoder choice is F7's largest cause on A val (37%, against 7% for M), consistent with the junction scope letting a light of another lane decide. That link was not tested directly. Relevance is the largest cause on B val for both. The DTLD row is on F7's own target.

**Latency** (RTX 5070, bf16, batch 1, median of 300 frames): F7 21.8 ms network / 36.9 ms with JPEG decode and resize, 34.4 M parameters. M 21.5 ms / 36.1 ms, 34.6 M parameters. No practical difference.

**Corrections to the Exp 7 reading above.**
- "It has lower False-Go" holds only on DTLD test (junction labels); on A val and B val the CIs include 0, and on route labels the difference is not significant.
- "In domain, it is at least as good as M everywhere" holds only with the label mask or on F7's own target. The tables here are the fair comparison.
- The Exp 6 results heading says 2026-11-08; those runs finished on 2026-11-07.

## VZC-TLD: all RF-DETR models on a US dataset with native relevance (2026-11-09, exploratory, no training)

VZC-TLD (Trinci et al., local copy at Hugging Face revision `083bc5d`) has 2,988 labelled frames from 2,856 sessions: 2,390 in its train split and 598 in its test split. No model trained on it, so both splits are unseen; the table uses all frames. Each light has a box, a color (62% of boxes are "unknown", treated as ignore boxes for detection and color) and a relevance flag. The image label is RR (relevant red or yellow, 1,681), RG (relevant green, 911) or NoR (no relevant colored light, 396). Always answering RR scores 0.563.

Each model uses its own source-selected epoch and threshold, with nothing tuned on VZC. Relevance definitions differ: lane topology (A, B, A+B, mix), route (DTLD), junction (Exp 6, Exp 7). VZC's "applies to the vehicle's current direction" is closest to route and stricter than junction. Image decision: the highest-scoring light the model calls relevant; red or yellow = stop, green = go, nothing = no relevant light (acts as go). False-Go = P(not stop | RR), False-Stop = P(stop | RG).

| Model | Center AP | R / P | Color / yellow R | Relevant: P / R / F1 | Decision accuracy [CI] | False-Go [CI] | False-Stop |
|---|---|---|---|---|---|---|---|
| A | 0.783 | 0.70 / 0.89 | 0.829 / 0.07 | 0.34 / 0.76 / 0.47 | 0.726 [0.710, 0.742] | 0.170 [0.152, 0.188] | 0.192 |
| B | 0.777 | 0.80 / 0.72 | 0.806 / 0.50 | 0.28 / 0.64 / 0.39 | 0.671 [0.655, 0.688] | 0.187 [0.169, 0.206] | 0.346 |
| A+B | 0.804 | 0.78 / 0.82 | 0.861 / 0.29 | 0.35 / 0.75 / 0.47 | 0.744 [0.728, 0.759] | 0.195 [0.177, 0.214] | 0.144 |
| DTLD | 0.668 | 0.57 / 0.82 | 0.899 / 0.73 | 0.29 / 0.68 / 0.41 | 0.733 [0.717, 0.748] | 0.148 [0.131, 0.165] | 0.210 |
| **Mix (M)** | **0.818** | 0.72 / 0.87 | 0.904 / 0.41 | 0.33 / 0.75 / 0.46 | 0.754 [0.739, 0.769] | 0.189 [0.171, 0.208] | **0.103** |
| Exp 6 J | 0.728 | 0.58 / 0.86 | 0.913 / 0.72 | 0.31 / 0.65 / 0.42 | 0.723 [0.707, 0.739] | 0.161 [0.143, 0.178] | 0.233 |
| Exp 6 F | 0.764 | 0.59 / 0.90 | 0.916 / 0.69 | 0.29 / 0.67 / 0.41 | 0.739 [0.724, 0.755] | 0.167 [0.148, 0.185] | 0.190 |
| **Exp 7 F (F7)** | 0.790 | 0.74 / 0.78 | 0.905 / 0.64 | 0.32 / 0.79 / 0.45 | **0.768** [0.753, 0.783] | **0.111** [0.096, 0.126] | 0.184 |

F7 − M on the image decision (paired session bootstrap, 10,000 draws):
- Accuracy: +0.014, CI [−0.001, +0.029].
- False-Go: −0.078, CI [−0.096, −0.061].
- False-Stop: +0.081, CI [+0.056, +0.106].

Decoding the straight maneuver instead gives the same pattern: accuracy 0.756 vs 0.746, False-Go 0.168 vs 0.233, False-Stop 0.123 vs 0.046. The official test split alone (598 frames) also agrees, with wider CIs: F7 accuracy 0.769, False-Go 0.097, False-Stop 0.216; M 0.746, 0.215, 0.097.

Reading:
- **Detection transfer to VZC favours OpenLane-trained models** (M 0.818, A+B 0.804). F7 (0.790) does not lead here, unlike LISA, S2TLD and BDD100K. No CI was computed for these AP values.
- **Relevance does not transfer.** Precision of "relevant" is 0.28–0.35 for every model, against VZC's narrower definition (12% of annotated lights are relevant). No model here should be claimed to solve relevance in the US.
- **F7 trades stops for goes.** It has the lowest False-Go (0.111, significantly below M), but False-Stop rises by a similar amount. It runs at a lower threshold than M and calls more lights relevant (recall 0.79 vs 0.75), a plausible cause not tested. This is an operating-point difference, not an accuracy win: the accuracy difference is not significant.
- Single seed for every model.

## Exp 8: F7 plus BDD100K train, for night and horizontal lights (2026-11-09, design fixed before any training or result)

**Question.** Does adding BDD100K train to the Exp 7 data improve detection at night and on horizontal lights, without losing the in-domain decisions? Only perception can improve here: BDD100K has no junction or pictogram label, so nothing new teaches relevance or arrows.

**Measured before fixing the design** (exploratory, no training):

| Item | Finding |
|---|---|
| Night in the Exp 7 training set | About 0.3% of frames. ATLAS and OpenLane A 0 of 400 sampled frames, OpenLane B 3.7%, DTLD 0 of 24 checked by eye. A brightness classifier calibrated on BDD100K's time-of-day labels reads 40% on DTLD, but that is DTLD's dark exposure: the frames are daytime. |
| Horizontal lights (w > 1.2 h) in the Exp 7 training set | 8,736 of 217,158 boxes, 95% from OpenLane (A 21%, B 11%, DTLD 0.3%, ATLAS 0.07%). |
| F7 at night, BDD100K val | AP 0.596, recall 0.76, precision 0.40, 3.0 false positives per image (day: 0.743, 0.78, 0.57, 0.7). LISA night: AP 0.531, recall 0.70, precision 0.49. |
| BDD100K annotation quality | A visual review of 96 F7 detections above 0.5 with no matching box, 48 night and 48 day: many are real lights left unlabelled, and some labels cover only the lit bulb, so the measured precision is a lower bound. |
| S2TLD by resolution | 720p: 4,564 frames, 56 of 11,191 lights horizontal (0.5%). 1080p: 1,222 frames, 1,030 of 2,125 horizontal (48%). The two groups share no session. |
| Bosch dev | 5,093 labelled frames, 12 sessions, 54 of 10,029 lights horizontal, 1,270 left arrows. |

**Training data (F8).** The Exp 7 set unchanged, plus one source.

| Source | Frames | Pictogram | Junction | Color |
|---|---|---|---|---|
| Exp 7 set | 67,140 | as Exp 7 | as Exp 7 | yes |
| BDD100K train, clean frames | 18,851 (night 13,477, dawn/dusk 1,134, day 4,222, undefined 18) | **masked** | **masked** | yes |

- **Clean frame:** at least one annotated light, and every annotated light has a color. This removes the frames with color "none" boxes, which are unusable as targets or as background, and needs no ignore mechanism. It leaves 64,096 lights, 6,202 of them horizontal, so the horizontal boxes in training rise by 71%, and night frames reach about 16%.
- **Frames with no annotated light are excluded** (30,626 frames): given the annotation gaps they cannot be trusted as negatives.
- **Boxes stay as annotated.** The rescaling of `box_convention.json` was rejected after visual review on 2026-10-26, and ATLAS was also trained raw in Exp 7. Evaluation uses center matching.
- **Not added, and why:**
  - S2TLD 720p has almost no horizontal lights, and the 1080p group is the only clean horizontal test, so neither is trained on.
  - Bosch brings neither night nor horizontal lights. It addresses US arrows, a different question.
  - VZC, LISA and BDD100K val stay test-only.
- **Accepted label noise:** unlabelled lights in clean BDD frames train as background, and bulb-only boxes conflict with the head boxes of other sources. The guards below would catch the harm.
- **Model and recipe:** exactly F7's (RF-DETR Medium at 1024, factorized head with 8 pictograms, lr 1e-4 / 1.5e-4, batch 4 × accum 4, seed 0, 5 epochs). Expected time about 10 h. `valid/` is unchanged.

**Selection, the same rule for F8 and F7.** Epoch and threshold that maximise the mean of DTLD tune (junction rule), A tune and B tune (US rule + lane topology mask). The label mask used in Exp 7 is dropped because it leaks the benchmark's answer. Nothing is selected on a test set.

**F7 under this rule, computed before F8 was trained.** Epoch 3, threshold 0.25 (DTLD tune 0.944, A tune 0.936, B tune 0.900, mean 0.927). Best mean by epoch: 0.922 / 0.922 / 0.921 / 0.927 / 0.922. Exp 7 selected epoch 4, threshold 0.20 under the label mask, so the F7 comparator here is not the F7 checkpoint and threshold reported in Exp 7, and its numbers will differ from that section.

**Endpoints, F8 − F7, paired.** Center AP; horizontal AP scores only horizontal lights, with the other lights as ignore boxes. Bootstrap 95% CI: blocks of 100 consecutive frames on LISA and S2TLD, sessions on VZC, images on BDD100K and ATLAS (2,000 draws, seed 0).

| Role | Cell | In F8's training domain? |
|---|---|---|
| Night, decision | LISA test night | no |
| Night, reported | BDD100K val night | yes |
| Horizontal, decision | S2TLD 1080p horizontal | no |
| Horizontal, reported | VZC horizontal, BDD100K val horizontal | VZC no, BDD yes |
| Daytime, decision guard | LISA test day | no |
| Reported | BDD100K val day and all, S2TLD 720p, VZC all, ATLAS test | ATLAS and BDD yes |

Recall, precision (lower bound where annotation is incomplete) and color accuracy are reported at each model's own threshold. LISA arrow reading is reported as a check, not a goal.

**In-domain guards** (segment / session bootstrap, 10,000 draws): TSS and False-Go on A val and B val (US rule + topology mask) and DTLD test (junction labels).

**Decision rule.** F8 replaces F7 as the perception model only if all of these hold:
- **(a)** ΔTSS lower CI above −0.01 on A val, B val and DTLD test;
- **(b)** ΔFalse-Go at most +0.01 on those three;
- **(c)** LISA night ΔAP lower CI above 0;
- **(d)** S2TLD 1080p horizontal ΔAP lower CI above 0;
- **(e)** LISA day ΔAP lower CI above −0.02.

BDD100K val is reported but cannot decide: F8 trained on its domain and label style. Single seed for both models, so everything is preliminary until seeds are added.

### Exp 8 results (2026-11-10)

Run `runs/exp8_bdd_F`, result in `runs/exp8_bdd_F/exp8_eval/result.json`. Training took 8 h 42 min, evaluation 1 h 04 min.

**Outcome: F8 does not replace F7 under the rule above.** (c), (d) and (e) hold; (a) and (b) fail, both on B val only.

| Condition | Result | Holds? |
|---|---|---|
| (a) ΔTSS lower CI above −0.01 | A val +0.003, B val **−0.036**, DTLD test −0.002 | no |
| (b) ΔFalse-Go at most +0.01 | A val +0.008, B val **+0.051**, DTLD test +0.001 | no |
| (c) LISA night ΔAP lower CI above 0 | +0.043 | yes |
| (d) S2TLD 1080p horizontal ΔAP lower CI above 0 | +0.083 | yes |
| (e) LISA day ΔAP lower CI above −0.02 | −0.009 | yes |

**Selection.** F8: epoch 3, threshold 0.45 (DTLD tune 0.945, A tune 0.948, B tune 0.913, mean 0.935; best mean by epoch 0.935 / 0.927 / 0.927 / 0.935 / 0.932). F7: epoch 3, threshold 0.25, as fixed above.

**In-domain guards**, F8 vs F7, with paired 95% CIs of the difference.

| Set | Frames | TSS F8 / F7 | ΔTSS (95% CI) | False-Go F8 / F7 | ΔFalse-Go (95% CI) | False-Stop F8 / F7 |
|---|---|---|---|---|---|---|
| A val (US rule + topology) | 2,220 | 0.937 / 0.924 | +0.013 (+0.003, +0.023) | 0.107 / 0.099 | +0.008 (−0.018, +0.034) | 0.014 / 0.026 |
| B val (US rule + topology) | 1,565 | 0.908 / 0.908 | +0.000 (−0.036, +0.027) | 0.138 / 0.087 | +0.051 (−0.010, +0.142) | 0.010 / 0.016 |
| DTLD test (junction labels) | 11,494 | 0.950 / 0.949 | +0.001 (−0.002, +0.004) | 0.033 / 0.031 | +0.001 (−0.003, +0.006) | 0.020 / 0.022 |

**Perception**, center AP, F8 / F7. Recall, precision and color are at each model's own threshold; precision is a lower bound where annotation is incomplete.

| Cell | In F8's domain? | Lights | AP F8 / F7 | ΔAP (95% CI) | Recall F8 / F7 | Precision F8 / F7 | Color F8 / F7 |
|---|---|---|---|---|---|---|---|
| LISA night (decision) | no | 39,629 | 0.714 / 0.635 | +0.079 (+0.043, +0.114) | 0.641 / 0.701 | 0.732 / 0.597 | 0.998 / 0.993 |
| LISA day (decision guard) | no | 18,020 | 0.847 / 0.843 | +0.004 (−0.009, +0.019) | 0.900 / 0.914 | 0.791 / 0.685 | 0.998 / 0.996 |
| S2TLD 1080p horizontal (decision) | no | 1,031 | 0.424 / 0.283 | +0.140 (+0.083, +0.191) | 0.697 / 0.621 | 0.371 / 0.218 | 0.974 / 0.983 |
| S2TLD 1080p | no | 2,126 | 0.706 / 0.591 | +0.114 (+0.047, +0.187) | 0.758 / 0.720 | 0.569 / 0.399 | 0.978 / 0.992 |
| S2TLD 720p | no | 11,191 | 0.895 / 0.890 | +0.005 (−0.002, +0.011) | 0.933 / 0.925 | 0.749 / 0.622 | 0.997 / 0.994 |
| VZC horizontal | no | 637 | 0.631 / 0.414 | +0.217 (+0.181, +0.253) | 0.589 / 0.551 | 0.597 / 0.215 | 0.915 / 0.863 |
| VZC all | no | 9,812 | 0.872 / 0.814 | +0.057 (+0.050, +0.064) | 0.640 / 0.732 | 0.961 / 0.846 | 0.942 / 0.909 |
| BDD val night | yes | 10,553 | 0.843 / 0.608 | +0.235 (+0.223, +0.246) | 0.721 / 0.731 | 0.840 / 0.480 | 0.984 / 0.980 |
| BDD val day | yes | 6,542 | 0.878 / 0.757 | +0.121 (+0.112, +0.130) | 0.770 / 0.775 | 0.859 / 0.640 | 0.971 / 0.959 |
| BDD val all | yes | 18,538 | 0.857 / 0.680 | +0.178 (+0.171, +0.185) | 0.740 / 0.748 | 0.848 / 0.534 | 0.977 / 0.970 |
| BDD val horizontal | yes | 1,680 | 0.325 / 0.079 | +0.246 (+0.223, +0.267) | 0.560 / 0.490 | 0.277 / 0.064 | 0.948 / 0.930 |
| ATLAS test | yes | 6,169 | 0.766 / 0.774 | −0.008 (−0.014, −0.002) | 0.670 / 0.731 | 0.917 / 0.807 | 0.984 / 0.982 |

**LISA arrow reading (check, not a goal).** Exact reading on matched arrows: F8 0.151 (n = 3,920), F7 0.321 (n = 4,583), difference −0.170 (−0.182, −0.157). F8 reads LISA arrows about half as well as F7.

**Reading.** BDD100K train improved the two targets out of domain, night on LISA and horizontal lights on S2TLD and VZC. The gains come with a higher selected threshold, so recall falls at the operating point on LISA night, VZC all and ATLAS while precision rises. On B val, the point ΔFalse-Go fails the +0.01 limit and the ΔTSS interval is too wide to pass; the ΔFalse-Go interval includes 0. Not tested: why False-Go rose on B val. Single seed for both models.

**AWSIM video (qualitative, no labels).** `tools/run_video_picto.py --run runs/exp8_bdd_F` at 0.45, saved as `runs/awsim/AWSIM_only_picto_F8.mp4` with a sample sheet beside it. Compared with the Exp 7 F7 video (epoch 4, threshold 0.20), over 7,076 frames: junction lights in 2,693 frames vs 3,370; straight red in 279 vs 144; straight yellow in 107 vs 39; straight green in 2,305 vs 3,121; no straight color in 4,385 vs 3,772; right-arrow detections 0 vs 187. The straight color differs in 1,622 frames. The two videos use different thresholds, so part of the drop in detections comes from the threshold.

## Exp 9: class imbalance of the pictogram and color heads (2026-11-13, design fixed before any training code or result)

**Question.** F8 reads rare pictograms and yellow badly. Does correcting the class imbalance in the attribute heads raise pictogram and yellow recall without breaking the in-domain per-maneuver decisions? Exp 6 to 8 trained the heads with plain cross-entropy (weight 1, no class weights), with no rebalanced sampling and no flip. Nothing has corrected the imbalance so far.

**Measured before fixing the design** (`tools/exp8_attributes.py`, `tools/exp9_premeasure.py`; F8 epoch 3 at 0.45, no training; outputs in `runs/exp8_bdd_F/exp8_eval/attributes.json` and `exp9_premeasure.json`):

| Item | Finding |
|---|---|
| F8 training pictograms (166,263 labelled boxes; 114,991 boxes have none) | circle 97,601 (58.7%), other 27,932, arrow_left 17,326, arrow_straight 14,989, arrow_right 5,311 (18× rarer than circle), arrow_straight_right 1,693, arrow_straight_left 1,411 (69× rarer) |
| F8 training colors (281,254 boxes) | green 145,797, red 125,882, yellow 9,575 (13× rarer than red) |
| F8 pictogram recall, DTLD test | circle 0.970, arrow_left 0.849, arrow_straight 0.575, **arrow_right 0.298**; macro 0.605 |
| F8 pictogram recall, ATLAS test | circle 0.918, arrow_left 0.669, arrow_right 0.570, arrow_straight 0.760; macro 0.729 |
| F8 pictogram recall, LISA test | circle 0.93 / 0.99 (day / night), **arrow_left 0.221 / 0.036** |
| Size or prior? DTLD lamps at least 30 px tall | arrow_left 0.90, arrow_straight 0.68, **arrow_right 0.33**. Right arrows fail even when large, while left arrows of the same size are read, which points to the class prior. Below 20 px most arrows of any type are read as circles: a resolution limit that no loss weight will fix. |
| LISA left arrows at least 30 px tall | 0.12 (374 of 3,244); 2,662 read as circles. Size is not the cause: LISA's arrows look unlike the training arrows (noted in Exp 7). Imbalance correction is not expected to fix this. |
| Yellow read as | DTLD test: yellow 1,297, red 135, green 64. BDD100K val: **red 185**, yellow 130. VZC: yellow 380, **green 130**, red 49. |
| F8 on the hand-labelled OpenLane-V2 val lights | 74 of 904 detected circles are called arrows (8.2%), 49 of them arrow_straight. Arrow_left: 86 of 112 right. False arrows on US and Singapore circles are what sank J+P (amendment 1), and they are the main risk of up-weighting arrows. |

**Arms.** All use the F8 data, recipe, epochs and seed: RF-DETR Medium at 1024, factorized head, lr 1e-4 / 1.5e-4, batch 4 × accum 4, 5 epochs, seed 0. Each arm changes one thing on top of F8 or on top of the previous arm:

| Arm | Change | Cost |
|---|---|---|
| **LA** (no training) | F8 with post-hoc logit adjustment: pictogram and color logits − τ · log(training prior), τ ∈ {0, 0.25, 0.5, 0.75, 1}. τ = 0 is F8. | inference that keeps the probabilities, about 1.5 h |
| **W** | Class-weighted cross-entropy in the pictogram and color heads: w_c ∝ n_c^−0.5, scaled so the count-weighted mean is 1 (PyTorch's weighted mean then keeps the loss scale). Junction loss unchanged (59 / 41 split). | training, about 9 h |
| **WF** | W plus a horizontal flip (p = 0.5) that swaps the pictogram labels: arrow_left ↔ arrow_right, arrow_straight_left ↔ arrow_straight_right. The weights come from the expected counts after the flip. | training, about 9 h |

**Weights**, fixed now from the F8 training counts. The power 0.5 is not tuned; full inverse frequency would weight arrow_straight_left 69 times a circle.

| | circle | other | arrow_left | arrow_straight | arrow_right | arrow_straight_right | arrow_straight_left | green | red | yellow |
|---|---|---|---|---|---|---|---|---|---|---|
| W | 0.601 | 1.124 | 1.427 | 1.534 | 2.577 | 4.565 | 5.000 | 0.883 | 0.950 | 3.444 |
| WF | 0.596 | 1.113 | 1.749 | 1.520 | 1.749 | 4.723 | 4.723 | 0.883 | 0.950 | 3.444 |

"unknown" has no training box and keeps weight 1.

- **Why a flip now.** It was excluded so far because it swaps arrows. With the labels swapped it is valid for pictograms, and it raises the right-arrow examples from 5,311 to about 11,300 per epoch. It also mirrors the traffic side. The junction flag is geometric (the light governs the ego approach), so it stays valid. OpenLane frames carry no pictogram to swap.
- **Not done, and why.**
  - Oversampling rare frames (LVIS repeat factors) changes the source mix and the detection training. That would confound the B val False-Go guard, which F8 already failed.
  - Focal loss adds a parameter and there is no budget to tune it.
  - A larger model does not address a class prior. Measured 2026-11-13 on the RTX 5070 at 1024: Nano to Large share the DINOv2-S backbone and are within 15% of each other in speed. The DINOv2-B variant is 2.6× slower and does not fit batch 4 in 12 GB.

**Selection.** The Exp 8 rule, unchanged: the epoch and threshold (and τ for LA) that maximise the mean of DTLD tune TSS (junction rule) and A tune and B tune TSS (US rule + lane topology mask). Pictogram or color recall is never used to select. Nothing is selected on a test set.

**Comparisons.**
- Each arm against F8, paired, and WF against W to isolate the flip. F7 is reported beside them.
- Attribute metrics use the lights both models of a pair match above their own thresholds (center matching), so they are not an operating-point effect (as in the Exp 6 check).
- 95% CI by cluster bootstrap, 2,000 draws, seed 0: segments (DTLD, OpenLane), sessions (VZC), blocks of 100 frames (LISA, S2TLD), images (ATLAS, BDD100K).

**Primary endpoints.**
- **P1:** pictogram macro recall on DTLD test, over circle, arrow_left, arrow_right, arrow_straight and other. arrow_straight_left has 1 lamp and is left out.
- **P2:** yellow recall on DTLD test (about 1,500 lamps).

**Guards.**
- **G1:** ΔTSS lower CI above −0.01 on A val, B val (US rule + topology mask) and DTLD test (junction labels), as in Exp 8.
- **G2:** ΔFalse-Go at most +0.01 on the same three sets.
- **G3:** color accuracy on DTLD test, Δ lower CI above −0.005. The yellow weight could turn reds into yellows.
- **G4:** false-arrow rate on the 904 detected hand-labelled OpenLane circles, Δ at most +0.03 (point estimate, because n is small).

**Reported, not deciding.**
- ATLAS test macro recall (in domain), and per-class pictogram recall and precision.
- LISA arrow_left recall (out of domain; not expected to move much).
- Yellow recall and color accuracy on ATLAS, LISA, S2TLD, VZC and BDD100K val.
- AP, recall and precision of every Exp 8 perception cell (detection should not change).
- Pictogram recall by lamp height.

**Decision rule.** An arm succeeds if G1 to G4 hold and P1 or P2 has a lower CI above 0; which one passed is reported. Then:
- If LA succeeds and no trained arm beats it on P1 and P2 (paired CI above 0), LA is adopted, since it needs no training.
- Otherwise, among the trained arms that succeed, WF is preferred over W only if WF − W has a P1 lower CI above 0. If not, W is adopted as the simpler change.

The adopted arm becomes the attribute recipe, not the perception model. F8 still fails the Exp 8 B val guard against F7, and Exp 9 does not change that comparison. Single seed, so everything is preliminary until seeds are added.

**Expected, written down to be checked:**
- arrow_right and arrow_straight recall up on DTLD and ATLAS;
- circle recall down by one or two points;
- more false arrows on OpenLane circles;
- LISA arrow_left nearly unchanged;
- yellow recall up on DTLD, with BDD100K yellow limited by its labels.

**Implementation (after this section is fixed).**
- `dtsu/factorized.py`: optional class weights in the pictogram and color cross-entropy.
- `tools/train_rfdetr.py`: `--class-weights 0.5` (computed from the dataset and saved in `args.json`) and `--flip-swap`. The flip is a transform after the resize that mirrors the image and boxes and permutes the class index. Flipped samples are checked visually before training.
- `tools/eval_exp9.py`: LA inference with the probabilities kept, the selection, the paired endpoints and the decision.
- Order: LA, a smoke test of both trained arms, then W, then WF. About 20 h of GPU in all.

**Implementation notes** (2026-11-13, recorded before any Exp 9 result; `tools/run_exp9.sh`):
- **Mirror classes.** The dataset has no `arrow_straight_right|J|*` class (ATLAS, the only source of straight-right arrows, has no junction label), so a flipped `arrow_straight_left|J|*` has no class index. `dtsu.factorized.mirror_classes` appends the 3 missing classes to the criterion's attribute table only; the detection head has one class, so nothing else reads the index.
- **Checks before training.** Through RF-DETR's own train dataset builder: labels equal the COCO category ids unflipped, and are permuted by the mirror permutation when flipped; flipped images and boxes checked by eye. Weights logged in `args.json` match the table above exactly. Two-epoch smoke runs of W and WF on 160 frames train, save and load.
- **LA reproduces F8 at τ = 0.** F8 is re-run with probabilities kept (batch composition differs from the Exp 8 caches, bf16): 359 of 360 frame × threshold checks give identical kept detections, pictograms and colors; scores differ by at most 0.005. LA's selection runs on these re-run detections; its comparisons with F8 use the Exp 8 caches of F8.
- **"unknown" in LA.** It has no training box, so its prior is set to the circle prior (the rule treats unknown as a circle); a zero prior would make τ favour it without limit.
- **Ties in selection:** lower epoch, then smaller τ, then lower threshold.
- **"Beats it on P1 and P2"** in the decision rule is read literally: a trained arm beats LA only if both paired lower CIs (arm − LA) are above 0.
- **G4** uses the hand-labelled OpenLane lights of A val and B val that both models match (center), as the other attribute metrics.

### Exp 9 results (2026-11-14, single seed)

Runs `runs/exp9_W` and `runs/exp9_WF`; LA caches in `runs/exp9_eval/LA`. Result in `runs/exp9_eval/result.json`, log in `runs/exp9_eval/eval.log`. LA inference took 1 h 04 min, training took 8 h 55 min for each of W and WF, and the evaluation took 1 h 58 min.

**Outcome: no arm is adopted.** None of the three arms passes all four guards.

| Arm | G1 ΔTSS | G2 ΔFalse-Go | G3 color | G4 false arrows | P1 picto macro | P2 yellow | Succeeds? |
|---|---|---|---|---|---|---|---|
| LA | holds | holds | holds | holds | no (Δ 0) | no (Δ 0) | no |
| W | **fails** (A val) | **fails** (A val) | holds | **fails** | no | no | no |
| WF | **fails** (A val) | **fails** (A val) | holds | holds | **yes** | no | no |

**Selection.**
- **LA.** Every τ > 0 lowered the tune TSS mean. The best mean by τ was 0.935, 0.934, 0.931, 0.928 and 0.923 for τ = 0, 0.25, 0.5, 0.75 and 1. LA therefore selected τ = 0 at epoch 3 and threshold 0.45, which is F8 itself, so its endpoints are exactly F8's. The rule selects on TSS only, and on these sets TSS does not reward reading rarer classes.
- **W.** Epoch 1, threshold 0.25. Tune TSS: DTLD 0.944, A 0.936, B 0.918; mean 0.933.
- **WF.** Epoch 4, threshold 0.35. Tune TSS: DTLD 0.943, A 0.947, B 0.905; mean 0.932.
- **F8.** Epoch 3, threshold 0.45; tune mean 0.935.

**Guards G1 and G2**, each arm against F8, with paired 95% CIs.

| Set | Arm | TSS arm / F8 | ΔTSS (95% CI) | False-Go arm / F8 | ΔFalse-Go (95% CI) |
|---|---|---|---|---|---|
| A val | W | 0.917 / 0.937 | −0.020 (**−0.034**, −0.009) | 0.149 / 0.107 | **+0.042** (+0.009, +0.082) |
| A val | WF | 0.927 / 0.937 | −0.010 (**−0.019**, −0.002) | 0.128 / 0.107 | **+0.021** (+0.001, +0.045) |
| B val | W | 0.928 / 0.908 | +0.020 (−0.010, +0.061) | 0.085 / 0.138 | −0.053 (−0.160, +0.023) |
| B val | WF | 0.920 / 0.908 | +0.013 (−0.019, +0.056) | 0.057 / 0.138 | −0.081 (−0.190, −0.007) |
| DTLD test | W | 0.950 / 0.950 | +0.000 (−0.003, +0.003) | 0.029 / 0.033 | −0.003 (−0.008, +0.001) |
| DTLD test | WF | 0.951 / 0.950 | +0.001 (−0.003, +0.004) | 0.035 / 0.033 | +0.002 (−0.004, +0.009) |

**Endpoints and other attribute guards** on DTLD test, measured on lights both models match. P1 here is the 5-class macro on matched lights. The 0.605 in the pre-measurement table is a different average, over all classes at F8's own operating point.

| Comparison | P1 picto macro (95% CI) | P2 yellow recall (95% CI) | G3 color accuracy Δ (95% CI) | G4 OpenLane false arrows |
|---|---|---|---|---|
| W vs F8 | 0.736 / 0.726, +0.009 (−0.005, +0.024) | 0.892 / 0.875, +0.017 (−0.001, +0.036) | −0.001 (−0.003, +0.000) | 0.184 / 0.082, **+0.102** |
| WF vs F8 | 0.776 / 0.729, **+0.048 (+0.032, +0.063)** | 0.889 / 0.880, +0.009 (−0.004, +0.022) | +0.000 (−0.001, +0.002) | 0.052 / 0.081, −0.029 |
| WF vs W | 0.770 / 0.732, **+0.038 (+0.020, +0.059)** | 0.862 / 0.878, −0.017 (−0.033, −0.001) | — | 0.061 / 0.192, −0.131 |
| F7 vs F8 | 0.730 / 0.727, +0.004 (−0.013, +0.020) | 0.858 / 0.874, −0.016 (−0.033, +0.001) | −0.000 (−0.002, +0.001) | — |

**Pictogram recall by class**, arm / F8 on matched lights; Δ with 95% CI.

| Set | Class | W | WF |
|---|---|---|---|
| DTLD test | circle | 0.962 / 0.970, −0.008 (−0.012, −0.004) | 0.968 / 0.971, −0.003 (−0.006, +0.002) |
| DTLD test | arrow_left | 0.858 / 0.851, +0.007 (−0.011, +0.024) | 0.859 / 0.851, +0.008 (−0.015, +0.029) |
| DTLD test | arrow_right (n ≈ 520) | 0.318 / 0.297, +0.021 (−0.044, +0.082) | **0.505 / 0.303, +0.202 (+0.137, +0.273)** |
| DTLD test | arrow_straight | 0.594 / 0.576, +0.018 (−0.013, +0.050) | 0.602 / 0.577, +0.025 (−0.004, +0.054) |
| DTLD test | other | 0.946 / 0.938, +0.008 (+0.003, +0.014) | 0.946 / 0.940, +0.006 (−0.002, +0.013) |
| ATLAS test | macro | 0.730 / 0.730, −0.000 (−0.019, +0.019) | **0.781 / 0.736, +0.045 (+0.022, +0.066)** |
| ATLAS test | circle | 0.897 / 0.921, −0.023 (−0.036, −0.011) | 0.922 / 0.921, +0.001 (−0.011, +0.012) |
| ATLAS test | arrow_right | 0.591 / 0.567, +0.024 (−0.030, +0.077) | **0.752 / 0.584, +0.168 (+0.086, +0.243)** |
| LISA test | arrow_left | 0.196 / 0.175, +0.020 (−0.012, +0.063) | 0.151 / 0.176, −0.025 (−0.050, −0.002) |
| LISA test | arrow_straight (n = 99) | 0.606 / 0.394, +0.212 (+0.062, +0.353) | 0.485 / 0.394, +0.091 (+0.000, +0.188) |

**WF by lamp height**, DTLD test, picto macro Δ against F8:
- 0–15 px: +0.159 (+0.015, +0.244).
- 15–20 px: +0.035 (−0.009, +0.087).
- 20–30 px: +0.038 (+0.003, +0.076).
- At least 30 px: +0.050 (+0.034, +0.068).

**W by lamp height:** at least 30 px, +0.006 (−0.011, +0.023).

**Yellow recall and color accuracy elsewhere**, arm / F8 on matched lights:

| Set | W yellow | WF yellow | W color acc Δ | WF color acc Δ |
|---|---|---|---|---|
| BDD100K val | 0.450 / 0.406, +0.044 (+0.013, +0.076) | 0.451 / 0.386, +0.065 (+0.032, +0.099) | −0.001 | −0.002 |
| VZC | 0.642 / 0.684, −0.042 (−0.080, −0.006) | 0.723 / 0.702, +0.021 (−0.018, +0.061) | −0.008 (−0.014, −0.003) | −0.004 |
| LISA test | 0.839 / 0.909, −0.070 (−0.171, −0.003) | 0.903 / 0.908, −0.005 | −0.002 | −0.000 |
| S2TLD | 0.982 / 0.982, +0.000 | 0.972 / 0.977, −0.005 | −0.003 (−0.004, −0.001) | −0.001 |
| A val | 0.728 / 0.752, −0.024 | 0.716 / 0.751, −0.035 (−0.081, +0.010) | −0.001 | +0.000 |
| B val | 0.880 / 0.870, +0.010 | 0.908 / 0.867, +0.041 (+0.000, +0.141) | +0.005 | +0.007 |

**Hand-labelled OpenLane circles called arrows:**
- A val: W 0.173 vs F8 0.094; WF 0.040 vs F8 0.094.
- B val: W 0.203 vs F8 0.064; WF 0.073 vs F8 0.061.

**Perception**, center AP arm / F8; Δ with 95% CI. Detection was not a target.

| Cell | W | WF |
|---|---|---|
| LISA night | 0.720 / 0.714, +0.007 (−0.011, +0.023) | 0.731 / 0.714, +0.018 (+0.003, +0.035) |
| LISA day | 0.870 / 0.847, +0.023 (+0.015, +0.032) | 0.854 / 0.847, +0.008 (−0.000, +0.016) |
| S2TLD 1080p horizontal | 0.407 / 0.424, −0.017 (−0.089, +0.049) | **0.340 / 0.424, −0.084 (−0.126, −0.017)** |
| S2TLD 1080p | 0.676 / 0.706, −0.030 (−0.070, +0.012) | 0.683 / 0.706, −0.023 (−0.084, +0.030) |
| S2TLD 720p | 0.905 / 0.895, +0.011 (+0.006, +0.015) | 0.922 / 0.895, +0.028 (+0.022, +0.033) |
| VZC horizontal | 0.618 / 0.631, −0.013 (−0.035, +0.008) | 0.641 / 0.631, +0.010 (−0.011, +0.031) |
| VZC all | 0.874 / 0.872, +0.002 (−0.001, +0.006) | 0.865 / 0.872, −0.007 (−0.011, −0.002) |
| BDD val all | 0.863 / 0.857, +0.005 (+0.002, +0.008) | 0.858 / 0.857, +0.001 (−0.002, +0.003) |
| BDD val night | 0.847 / 0.843, +0.004 (−0.000, +0.008) | 0.841 / 0.843, −0.002 (−0.006, +0.002) |
| BDD val horizontal | 0.314 / 0.325, −0.010 (−0.022, +0.002) | 0.329 / 0.325, +0.004 (−0.008, +0.017) |
| ATLAS test | 0.758 / 0.766, −0.008 (−0.014, −0.002) | 0.776 / 0.766, +0.010 (+0.004, +0.016) |

**Expectations, checked:**
- **arrow_right up on DTLD and ATLAS:** yes for WF (+0.20 and +0.17). Not for W, where the change is within noise. arrow_straight rose by 0.02–0.03 in both arms, also within noise.
- **Circle down by one or two points:** W −0.8 on DTLD and −2.3 on ATLAS. WF is unchanged.
- **More false arrows on OpenLane circles:** yes for W (8.2% → 18.4%). The opposite for WF (8.1% → 5.2%).
- **LISA arrow_left nearly unchanged:** yes; W +0.02, WF −0.025.
- **Yellow up on DTLD:** not confirmed; W +0.017 with a lower CI of −0.001, WF +0.009. BDD100K yellow rose in both arms (+0.04 and +0.07).

**Exploratory, not pre-registered:** WF's A val guard failure is not an effect of its threshold. With WF at epoch 4, A val ΔTSS / ΔFalse-Go against F8 were:

| WF threshold | ΔTSS | ΔFalse-Go |
|---|---|---|
| 0.35 | −0.010 | +0.021 |
| 0.40 | −0.012 | +0.030 |
| 0.45 (F8's threshold) | −0.012 | +0.034 |
| 0.50 | −0.015 | +0.044 |

**Reading.**
- **Class weights alone (W) did little for rare pictograms.** P1 rose by +0.009, which is within noise, and right arrows stayed at about 0.32 even on large lamps. W also tripled the false arrows on OpenLane circles and lowered circle recall. The prior was not the only obstacle: there were too few right-arrow examples to learn from.
- **The label-swapping flip (WF) is what helped.** It doubled the right-arrow examples. DTLD right-arrow recall rose from 0.30 to 0.50 and ATLAS from 0.58 to 0.75, and WF beats W on P1. False arrows on OpenLane circles fell instead of rising.
- **Neither arm moved P2.** Yellow recall on DTLD stayed within noise in both, although BDD100K yellow improved.
- **Both trained arms fail on A val, while B val and DTLD test pass.** On A val, False-Go rose by about +0.02 for WF and +0.04 for W. Not tested: which lights cause the extra False-Go on A val. WF's false arrows on A val circles fell (0.094 → 0.040), so the cause is not false arrows on circles.
- **WF also lost horizontal lights on S2TLD 1080p** (−0.084 AP).
- **Caveats.** Single seed. LA was not a real test, because the selection rule never picks τ > 0.

### Exp 9 follow-up: where the extra A val False-Go comes from (2026-11-14, exploratory, no training)

Scripts: `tools/exp9_aval_false_go.py [WF|W]` → `runs/exp9_eval/aval_false_go_{arm}.json`, and `tools/exp9_aval_sheet.py` → `runs/exp9_eval/aval_false_go_{arm}.jpg`, a crop sheet. Both models are decoded as in the guards: US rule plus topology mask, each at its selected epoch and threshold.

**Method.**
- A val has 2,113 (frame, direction) cells whose ground truth is red.
- For each cell where exactly one model predicts not red, I find the light that gave the other model its red. Then I check what the failing model read on that light (IoU ≥ 0.3).
- The left and U-turn directions come from the same light under the US rule, so each such light counts twice.

**Counts.**
- WF goes wrong on 81 cells where F8 is right, and F8 on 37 where WF is right: net +44, which is the +0.021 of G2.
- For W the figures are 139 and 50: net +89, the +0.042.
- The extra cells are concentrated. WF's 81 come from 37 frames in 16 segments, and 50 of them from 5 segments.

**Causes of WF's extra False-Go** (net = WF-only minus F8-only):

| Cause | WF-only | F8-only | Net |
|---|---|---|---|
| Light read, but junction score ≤ 0.5, so the rule ignores it | 26 | 2 | +24 |
| Red light read, but another (green) light sets the direction | 32 | 13 | +19 |
| Light read as not red | 11 | 10 | +1 |
| Light missed | 9 | 11 | −2 |
| No light sets the direction | 3 | 1 | +2 |

**What these lights are.**
- In 52 of WF's 81 cells, F8's red came from a **red left-turn arrow**. On the crop sheet these are mostly US protected-left signals: a separate left-arrow head while the through signals are green. Most sit on the left of the image (centre x 0.16–0.40).
- On those arrows WF:
  - calls them not relevant to the ego approach in 26 cells (junction scores 0.01–0.37, where F8 gave 0.52–0.99);
  - **reads them as right arrows** in 12 cells (segment 10026, small lamps);
  - reads them as circles in 6, so a higher-scored green circle sets the left direction;
  - reads them yellow or green in 8.
- In each of these the left turn turns green: a False-Go on exactly the protected-left phase.

**W vs WF.**
- 48 of WF's 81 cells are also W failures. W shows the same pattern more strongly: 40 cells where a red left arrow is called not relevant, plus 24 cells where a red is read as green.
- So most of the regression is common to both arms and comes with the class-weighted training, not with the flip. The **left → right misreads** are WF-only and fit the flip, which balances left and right arrows: the prior no longer favours "left" on a small, unclear arrow.

**Reading.**
- A val's extra False-Go is a junction-relevance problem on US red left-turn arrows, not a pictogram-on-circle problem.
- The junction head itself is unweighted. Something in weighted training lowers its confidence on these lights; how is not tested. One candidate: arrows weighted higher in the shared decoder, against a junction label that ATLAS, the main arrow source, lacks.
- Not tested: whether a weighted run with a different epoch keeps the pictogram gain without this; whether a separate threshold on the junction score repairs it.
- The pattern sits in a handful of segments, consistent with G2's lower CI of +0.001 for WF.

## Exp 10: WF at 1280 instead of 1024 (2026-11-14, design fixed before any training or result)

**Question.** Does a higher input resolution make WF read pictograms better, above all on small lamps, without breaking the per-maneuver decisions?

**Why resolution.**
- Exp 9 left small lamps as the weakest point. On DTLD test, WF's pictogram macro recall is 0.60 for lamps under 15 px, 0.59 for 15–20 px and 0.82 for lamps at least 30 px tall.
- The pre-measurement for Exp 9 also found that below 20 px most arrows are read as circles: a resolution limit rather than a class prior.
- The training images are larger than 1024 px (DTLD 2048 × 1024, ATLAS 1920 × 1200, OpenLane 1550 × 2048, BDD100K 1280 × 720), so 1280 gives the network more pixels per lamp on all sources except BDD100K.
- Model size is not tested. In rfdetr 1.11.2, RFDETRLarge has the same architecture as RFDETRMedium (DINOv2-S, 4 decoder layers, 34 M parameters); only the pretrained weights differ. The DINOv2-B variant does not fit the 12 GB GPU even at batch 2.

**Arm R.** The WF recipe of Exp 9, unchanged except for `--resolution 1280`:
- RF-DETR Medium, factorized head, class weights at power 0.5 and the label-swapping flip;
- lr 1e-4 / 1.5e-4, batch 4 × accum 4, 5 epochs, 6 workers, seed 0;
- data `data/rfdetr_coco_exp8`.

**Cost**, measured on the RTX 5070 with the factorized head:

| | 1024 (WF) | 1280 (R) |
|---|---|---|
| Train step, batch 4 | 219 ms, 7.1 GB | 398 ms, 10.1 GB |
| Inference, batch 1 | 22 ms | 29 ms |

Expected time: about 16 h of training and 3 h of evaluation.

**Selection.** The Exp 8 rule, unchanged: the epoch and threshold that maximise the mean of DTLD tune TSS (junction rule) and A tune and B tune TSS (US rule plus lane topology mask).

**Comparisons.**
- R against WF, the reference: WF at epoch 4, threshold 0.35, as selected in Exp 9.
- R against F8 is reported beside it.
- As in Exp 9: attribute metrics use the lights both models of a pair match, and every CI is a cluster bootstrap with 2,000 draws.
- G1 and G2 use the segment bootstrap with 10,000 draws.

**Primary endpoints (R − WF):**
- **P1:** pictogram macro recall on DTLD test, over circle, arrow_left, arrow_right, arrow_straight and other. Same definition as in Exp 9.
- **P2:** the same macro recall on DTLD test lamps under 20 px tall, the 0–15 and 15–20 px bins together.

**Guards (R − WF)**, with the thresholds of Exp 9:
- **G1:** ΔTSS lower CI above −0.01 on A val, B val and DTLD test.
- **G2:** ΔFalse-Go at most +0.01 on the same three sets.
- **G3:** DTLD test color accuracy, Δ lower CI above −0.005.
- **G4:** false-arrow rate on the hand-labelled OpenLane circles, Δ at most +0.03.

**Reported, not deciding.**
- The same guards and endpoints against F8, so R can be read against the Exp 9 decision.
- Yellow recall and the other attribute metrics, on every set and against both references.
- AP, recall and precision of every Exp 8 perception cell, against WF and F8. Higher resolution may help small and horizontal lights; that is reported only.
- The A val False-Go trace of the Exp 9 follow-up for R.

**Decision rule.** R replaces WF as the recipe if G1 to G4 hold against WF and P1 or P2 has a lower CI above 0. Otherwise WF stays. R's slower inference (29 vs 22 ms) is reported but does not count against it.
- R does not settle the A val False-Go of WF against F8. If R passes, that comparison is reported as for WF.
- Single seed.

**Expected, written down to be checked:**
- P2 up more than P1;
- arrow_right and arrow_straight recall up;
- small-lamp and horizontal-light AP up (S2TLD 1080p, VZC, BDD100K);
- BDD100K unchanged in color, since its images are 1280 × 720;
- the A val False-Go pattern of WF unchanged, because it is about junction relevance rather than resolution.

**Implementation.**
- `tools/run_exp10.sh`: trains `runs/exp10_R`, then runs the evaluation.
- `tools/eval_exp10.py`: reuses the Exp 9 functions (selection, guards, matched-light attributes, perception cells) with R, WF and F8. Each model is decoded at its own training resolution, read from `args.json`. Writes `runs/exp10_eval/result.json`.

### Exp 10 results (2026-11-15, single seed)

**Run.**
- `runs/exp10_R` trained for 5 epochs: 2.9 h per epoch and 9.1 GB peak, against 1.8 h and 6.6 GB for WF.
- The first epoch took 3.75 h because another GPU job ran alongside it. Training took 15.2 h in total, and evaluation took 1.3 h.
- The library's built-in validation mAP is 0 for every epoch, as for WF, because it does not read the factorized head. All metrics below come from `tools/eval_exp10.py` → `runs/exp10_eval/result.json`.
- An earlier launch on 2026-11-14 was killed with its session before epoch 0 ended and is not used.

**Selection.** Epoch 2 at threshold 0.35, with a tune mean of 0.941 (DTLD 0.950, A 0.951, B 0.921). WF stays at epoch 4, threshold 0.35, and F8 at epoch 3, threshold 0.45.

**Decision: R does not replace WF.** P1 and G3 pass. G1, G2 and G4 fail.

| | R − WF | 95% CI | Rule | Pass |
|---|---|---|---|---|
| P1 DTLD test pictogram macro recall | **+0.049** (0.821 vs 0.772) | [+0.033, +0.066] | lower CI > 0 | ✅ |
| P2 the same, lamps under 20 px | +0.015 (0.609 vs 0.594) | [−0.016, +0.050] | lower CI > 0 | ❌ |
| G1 ΔTSS A val | +0.007 (0.934 vs 0.927) | [−0.005, +0.020] | lower CI > −0.01 | ✅ |
| G1 ΔTSS B val | −0.002 (0.918 vs 0.920) | **[−0.027**, +0.015] | | ❌ |
| G1 ΔTSS DTLD test | +0.004 (0.955 vs 0.951) | [−0.000, +0.008] | | ✅ |
| G2 ΔFalse-Go A val | **−0.036** (0.092 vs 0.128) | [−0.073, −0.005] | ≤ +0.01 | ✅ |
| G2 ΔFalse-Go B val | **+0.035** (0.093 vs 0.057) | [−0.006, +0.097] | | ❌ |
| G2 ΔFalse-Go DTLD test | −0.003 (0.032 vs 0.035) | [−0.009, +0.003] | | ✅ |
| G3 DTLD test color accuracy | +0.003 (0.984 vs 0.981) | [+0.002, +0.005] | lower CI > −0.005 | ✅ |
| G4 false arrows, hand-labelled OpenLane circles | +0.039 (0.097 vs 0.058) | [+0.006, +0.078] | ≤ +0.03 | ❌ |

**Against F8 (reported).**
- ΔTSS: A val −0.003 [−0.012, +0.006], B val +0.010 [−0.018, +0.053], DTLD test +0.0045 [+0.0004, +0.0084].
- ΔFalse-Go: −0.015, −0.045 and −0.001, so lower than F8 on all three sets.
- Pictogram macro recall: P1 +0.097 [+0.081, +0.114] and P2 +0.077 [+0.021, +0.138].
- Hand-labelled false arrows: +0.015.
- Under the Exp 9 guards, R against F8 fails only G1 on A val, and narrowly (lower CI −0.012).

**Pictograms (matched lights, R − WF).**

| Set | Class | R | WF | Δ | 95% CI |
|---|---|---|---|---|---|
| DTLD test | circle | 0.961 | 0.965 | −0.004 | [−0.009, −0.000] |
| | arrow_left | 0.869 | 0.853 | +0.017 | [−0.003, +0.039] |
| | arrow_right | 0.648 | 0.506 | **+0.143** | [+0.076, +0.218] |
| | arrow_straight | 0.679 | 0.597 | **+0.082** | [+0.049, +0.114] |
| | other | 0.947 | 0.938 | +0.009 | [+0.001, +0.017] |
| ATLAS test | macro | 0.826 | 0.777 | +0.049 | [+0.033, +0.066] |
| LISA test | arrow_left | 0.283 | 0.152 | +0.131 | [+0.070, +0.184] |
| B val | circle | 0.810 | 0.872 | −0.061 | [−0.169, +0.003] |
| B val | false arrows on circles | 0.176 | 0.084 | +0.092 | [+0.032, +0.199] |

DTLD test pictogram macro recall by lamp height:

| Height | R | WF | Δ | 95% CI |
|---|---|---|---|---|
| 0–15 px | 0.555 | 0.628 | −0.073 | [−0.219, +0.023] |
| 15–20 px | 0.623 | 0.598 | +0.024 | [+0.000, +0.061] |
| 20–30 px | 0.696 | 0.658 | +0.038 | [+0.002, +0.085] |
| ≥ 30 px | 0.873 | 0.820 | +0.053 | [+0.037, +0.070] |

**Color (R − WF).**
- Yellow recall goes up: DTLD test +0.040 [+0.024, +0.059], LISA night +0.155, BDD val +0.057 [+0.024, +0.092].
- Color accuracy is unchanged on DTLD, LISA, ATLAS and BDD100K (BDD −0.002).
- It drops on S2TLD: −0.011 [−0.021, −0.003] overall and −0.062 [−0.127, −0.011] on its 1920 × 1080 images.

**Perception cells (AP, R − WF).**
- Up: ATLAS test +0.038 [+0.030, +0.045], LISA night +0.036 [+0.014, +0.058], BDD100K +0.008 [+0.005, +0.010].
- Down: LISA day −0.019 [−0.034, −0.006], S2TLD 720p −0.035 [−0.040, −0.030], VZC horizontal −0.090 [−0.119, −0.061].
- No clear change: S2TLD 1080p −0.020, S2TLD 1080p horizontal +0.025 and BDD horizontal +0.004, all with CIs that include 0.

**Expectations check.**
- P2 up more than P1: **no**. The gain grows with lamp size (+0.053 at 30 px and above). Below 15 px the change is −0.073 with a wide CI.
- arrow_right and arrow_straight up: **yes**.
- Small-lamp and horizontal-light AP up: **no**. S2TLD and BDD horizontal are flat, and VZC horizontal and S2TLD 720p go down.
- BDD100K color unchanged: **yes** for accuracy, but its yellow recall rises.
- WF's A val False-Go pattern unchanged: **no**. R removes most of it (see the trace below).

### Exp 10 follow-up: R's False-Go against WF on A val and B val (2026-11-15, exploratory, no training)

**Method.** The Exp 9 trace, generalised: `tools/exp9_aval_false_go.py ARM REF SET` and `tools/exp9_aval_sheet.py ARM REF SET` → `runs/exp10_eval/false_go_R_vs_WF_{A_val,B_val}.{json,jpg}`. The Exp 9 outputs are reproduced unchanged.

**A val: R repairs WF's red left-arrow failures.**
- WF goes wrong on 122 cells where R is right, and R on 47 where WF is right: net −75, the −0.036 of G2.
- In 75 of WF's 122 cells, R's red came from a red left arrow. On those, WF:
  - lets a green light set the direction (37), mostly by reading the arrow as a right arrow (23);
  - calls the arrow not relevant (24);
  - reads it as not red (12);
  - misses it (2).
- By cause, R minus WF is: outvoted −41, junction score too low −24, color −16, missed +6.
- So the Exp 9 regression on US protected-left arrows is mostly a 1024 px effect, not an effect of the weighting.

**B val: two segments.**
- R goes wrong on 92 cells where WF is right, and WF on 21: net +71, the +0.035 of G2.
- **78 of R's 92 cells are in two segments:** 11067 (51 cells, 23 frames) and 11049 (27 cells, 9 frames). Outside them R has 14 and WF 19, so R is slightly ahead there (net −5).
- In every R-only cell the direction ends as "none" (no relevant light read), never green. Causes: missed 55, junction score too low 24, read but routed to another direction 13.
- The hand-labelled false arrows on circles (G4) are not what drives this.
- **Segment 11067** (US city street, a truck partly hiding the signals):
  - A distant through signal: WF reads a red circle at 0.49–0.78. R reads the same lamp either as a red circle at 0.09–0.31, below threshold, or as a red **left** arrow (score 0.58–0.71, junction 0.98), which the rule routes to the left turn.
  - Later frames, a near pole-mounted signal: WF reads a red straight arrow at 0.36–0.46, just over its 0.35 threshold. R gives it 0.05–0.37.
- **Segment 11049** (a single light on a pole and a horizontal light against a bright sky):
  - R reads a red circle but gives it a junction score of 0.18–0.47, where WF gives 0.79–0.93.
  - Where WF scores the horizontal light 0.43–0.56, R misses it.
- **Lower thresholds** (exploratory, `r_thresholds.py` in the session scratchpad). R at 0.30 / 0.25 / 0.20 gives B val ΔFalse-Go +0.030 / +0.026 / +0.022 and A val −0.036 / −0.040 / −0.045. G2 on B val still fails, and the tune mean drops by at most 0.002.

**Reading.**
- 1280 helps pictograms on lamps of 15 px and more, helps yellow, and fixes WF's protected-left arrows on A val.
- It does not help the smallest lamps, so the premise of P2 does not hold: arrows under 15 px are not recovered by 25 % more pixels.
- The guard failures are concentrated:
  - B val False-Go comes from two segments, through lower scores and junction confidence on a few lamps;
  - G4 comes from more arrow calls on hand-labelled circles;
  - G1 on B val is a wide CI around −0.002.
- With one seed and 1,565 B val frames, it cannot be told whether these are a resolution effect or seed noise.
- By the pre-registered rule, **WF stays the recipe**. R is the better model against F8 on every guard set but A val TSS. It costs 30 % more inference time (29 vs 22 ms) and 1.6× the training time.
- Not tested: a second seed of R and WF, which would separate the B val segments from noise; R at a later epoch (epoch 4 has a tune mean of 0.932, below epoch 2).

## Exp 11: R at 1536, first epoch only (2026-11-16, design fixed before any training or result)

**Question.** After one epoch, does 1536 read pictograms better than 1280, and better than 1024, with the same training budget?

**Why one epoch.** Training at 1536 takes about 4.7 h per epoch, about 24 h for all five. The user asked for the first epoch only, then a comparison with WF and R.
- On the selection metric, epoch 0 did **not** already separate R from WF. The tune means at epoch 0 were R 0.9249 and WF 0.9258. R pulled ahead from epoch 1 (0.9338 against 0.9306) and peaked at epoch 2.
- So epoch 0 is compared with epoch 0, and R0 against WF0 on the test sets is reported too. It shows whether R's pictogram gain was already visible after one epoch.
- If it was not, a null result for S says little about a full run.

**Arm S.** The R recipe with these changes:
- `--resolution 1536`;
- `--batch-size 2 --grad-accum 8` instead of 4 × 4, because batch 4 does not fit in 12 GB at 1536. The effective batch (16) and the number of optimizer steps per epoch (5,375) stay the same.
- `--epochs 5 --stop-after 1`, so the learning-rate schedule is the 5-epoch one of R, and the run can be resumed to 5 epochs from `last.ckpt`.

Everything else is unchanged: RF-DETR Medium, factorized head, class weights at power 0.5, `--flip-swap`, lr 1e-4 / 1.5e-4, seed 0, data `data/rfdetr_coco_exp8`.

**Cost**, measured on 2026-11-16:

| | 1280 (R) | 1536 (S) |
|---|---|---|
| Train step | 425 ms at batch 4 (10.1 GB) | 349 ms at batch 2 (7.2 GB); batch 4 does not fit |
| Inference, batch 1 | 29.5 ms | 54 ms |

Expected time: about 4.7 h of training, then 3–4 h of evaluation. The evaluation has to infer three epoch-0 models on every test set.

**Models compared.**
- **S** is the 1536 model at epoch 0.
- **R0** and **WF0** are R and WF at epoch 0.
- **R** and **WF** are the selected models of Exp 10 and Exp 9 (epoch 2, threshold 0.35; epoch 4, threshold 0.35).
- Each epoch-0 model gets its threshold from the Exp 8 rule with the epoch fixed at 0: the threshold that maximises the mean of DTLD tune TSS and A/B tune TSS.

**Primary (S − R0, equal budget):**
- **P1:** DTLD test pictogram macro recall.
- **P2:** the same on lamps under 20 px.

**Guards (S − R0):** G1–G4 with the Exp 9 thresholds.

**Reported, not deciding:**
- the same endpoints and guards for S − WF0, R0 − WF0, S − R and S − WF;
- the other attribute metrics and the Exp 8 perception cells for S − R0;
- inference time.

**Decision rule.** S is resumed to 5 epochs (about 19 h more) if, against R0, P1 or P2 has a lower CI above 0 and G1–G4 hold. Otherwise 1280 stays the highest resolution tried, and S stays at one epoch.
- If R0 − WF0 shows no pictogram gain either, then epoch 0 does not predict the full run at all. In that case the result is reported as inconclusive rather than negative.
- Single seed.

**Expected, written down to be checked:**
- S above R0 on lamps of 15 px and more, as R was above WF;
- P2 flat again, since 1280 did not help lamps under 15 px;
- S2TLD color down again;
- R0 above WF0 on P1, smaller than the full-run gap of +0.049.

**Implementation.**
- `tools/run_exp11.sh` trains `runs/exp11_S` and then runs `tools/eval_exp11.py`.
- `tools/eval_exp11.py` reuses the Exp 9 / Exp 10 functions and writes `runs/exp11_eval/result.json`.

### Exp 11: epoch-0 outcome and continuation (2026-11-17)

**Epoch 0.** Training took 4.7 h (16,949 s including validation, peak memory 6.55 GiB), and the evaluation 2.2 h. Under the rule above, S does not continue:
- **G1 fails on B val:** ΔTSS −0.014, CI [−0.050, +0.014].
- **P1 and P2 are not proven:** P1 +0.018, CI [−0.002, +0.039]; P2 +0.024, CI [−0.039, +0.077].
- **G2–G4 hold.**
- **Tune mean at epoch 0:** S 0.9334, R0 0.9249, WF0 0.9258.

Full numbers are in `runs/exp11_eval/result.json`.

**Continuation, against the pre-registered rule.** The user decided to resume S to the full 5-epoch schedule. This is a deviation from the decision rule and is reported as such: the 5-epoch result is exploratory, not a confirmatory test.
- **Training:** `tools/run_exp11_resume.sh` resumes from the epoch-0 `last.ckpt`. A copy of it, with `metrics.csv` and `args.json`, is kept in `runs/exp11_S/epoch0_backup/`. The schedule, data and settings are unchanged.
- **Evaluation:** `tools/eval_exp11.py --full` selects S over epochs 0–4 and threshold with the Exp 8 rule. It then compares S with R and WF at their selections (guards, attributes, P2, perception cells) and applies the Exp 10 rule to S − R. It writes `runs/exp11_eval_full/result.json`.
- **Expected time:** about 19 h of training (4 epochs × 4.7 h), then about 3 h of evaluation.

### Exp 11 results (2026-11-19, single seed)

**Epoch 0, against the expectations written before training** (S − R0 unless stated):

| Expectation | Result | Held? |
|---|---|---|
| S above R0 on lamps of 15 px and more | 15–20 px +0.032 [−0.029, +0.084]; 20–30 px +0.055 [+0.028, +0.085]; 30 px and over +0.011 [−0.010, +0.034] | partly: only 20–30 px is clear |
| P2 flat | +0.024 [−0.039, +0.077] | yes |
| S2TLD color down | +0.007 [+0.002, +0.014] | no, it went up |
| R0 above WF0 on P1, below the full-run gap of +0.049 | +0.028 [+0.013, +0.044] | yes |

Because R0 − WF0 already showed the pictogram gain, the epoch-0 test was conclusive, not inconclusive. The pre-registered outcome stands: S stays at one epoch.

**5-epoch continuation (exploratory, at the user's request; see the note above).**

*Run.*
- **Resume attempt 1:** stopped at 16:36 on 2026-11-17, 7 minutes in. There was no traceback, and the launcher shell was gone too.
- **Resume attempt 2** (from 17:03 on 2026-11-17):
  - It completed epochs 1–3 in 30,111 s, 19,347 s and 19,655 s. Another program shared the GPU for long stretches, so these are slower than epoch 0 (16,949 s).
  - It stopped the same way at about 17:52 on 2026-11-18, at step 25,449 of epoch 4.
- **Epoch 4** was rerun from the end-of-epoch-3 `last.ckpt` (backup in `runs/exp11_S/epoch3_backup/`). It finished at 00:20 on 2026-11-19, in 20,929 s.
- **Evaluation:** 00:20–02:39.
- **Logs:** the rows of the killed attempts are kept in `metrics_attempt*_killed*.csv`, and `metrics.csv` holds one clean row set per epoch.
- **Settings:** schedule, data and every training setting unchanged. Peak memory 6.56 GiB.

*Selection (Exp 8 rule):* **epoch 2, threshold 0.35**, the same epoch as R.

| Model | Tune mean by epoch, 0 → 4 | DTLD tune | A tune | B tune | Mean at selection |
|---|---|---|---|---|---|
| S (1536) | 0.933, 0.936, **0.942**, 0.940, 0.938 | 0.955 | 0.946 | 0.924 | 0.9417 |
| R (1280) | 0.925, 0.934, **0.941**, 0.935, 0.932 | 0.950 | 0.951 | 0.921 | 0.9407 |
| WF (1024) | 0.926, 0.931, 0.924, 0.928, **0.932** | 0.943 | 0.947 | 0.905 | 0.9318 |

S follows R's curve: it peaks at epoch 2, then declines slightly.

*Decision, S − R, with the Exp 10 rule:* **S does not replace R.**

| Criterion | S − R | Result |
|---|---|---|
| P1 DTLD pictogram macro recall | +0.010 [−0.002, +0.022] | not shown |
| P2 under 20 px | −0.011 [−0.030, +0.006] | not shown |
| G1 TSS A val | −0.003 [−0.011, +0.005] | **fails** (lower bound below −0.01) |
| G1 TSS B val | +0.019 [+0.008, +0.031] | pass |
| G1 TSS DTLD test | +0.007 [+0.003, +0.011] | pass |
| G2 False-Go A val | +0.017 (0.109 against 0.092) [−0.004, +0.040] | **fails** (limit +0.01) |
| G2 False-Go B val / DTLD test | −0.027 / −0.004 | pass |
| G3 DTLD color accuracy | +0.002 [+0.000, +0.003] | pass |
| G4 hand-labelled false arrows | −0.020 | pass |

*Where S differs from R* (95 % CI excludes 0 unless stated):
- **OpenLane B val**, R's weak set in Exp 10:
  - TSS +0.019, False-Go 0.066 against 0.093.
  - False arrows on round lights −0.092.
  - Yellow recall −0.049.
- **DTLD test:**
  - TSS +0.007, junction accuracy +0.004, false arrows on round lights −0.008.
  - Pictogram recall: circle +0.010, straight arrow +0.030, lamps of 30 px and over +0.014.
  - Lamps under 30 px: no clear change (under 15 px −0.019 [−0.055, +0.024]).
- **ATLAS test:**
  - Pictogram macro recall about level, +0.001 [−0.016, +0.019].
  - False arrows −0.059, color accuracy +0.006, detection AP +0.020.
- **LISA (OOD):**
  - Pictogram macro recall +0.074 [−0.012, +0.088], not clear.
  - Yellow recall −0.060.
  - Detection AP: day +0.018, night −0.020.
- **S2TLD (OOD):**
  - Detection AP: 1080p +0.084, 1080p horizontal lights +0.107, 720p +0.022.
  - Color accuracy +0.011.
- **VZC-TLD (OOD):** horizontal-light AP +0.076, all lights +0.010, junction accuracy +0.012.
- **BDD100K val:**
  - Yellow recall −0.055.
  - Detection AP −0.003, horizontal lights −0.035.
- **Cost:** inference 54 ms per frame, against 29.5 ms for R.

*S − WF, for reference:*
- **Pictogram macro recall:** DTLD +0.058 [+0.043, +0.074], ATLAS +0.052, LISA +0.127.
- **Arrow classes on DTLD:** right +0.129, straight +0.110, left +0.035.
- **By lamp height on DTLD:** lamps 20 px and over +0.048 to +0.065. Under 15 px is worse, −0.089 [−0.227, −0.006].
- **TSS:** A val +0.003, B val +0.017, DTLD +0.011.
- **False-Go:** A val 0.109 against 0.128, B val 0.066 against 0.057.

**Reading.**
- Going from 1280 to 1536 does not give the clear pictogram gain the rule asks for.
- It is a trade:
  - **Better:** B val decisions and the detection of small or distant lights (S2TLD 1080p, VZC horizontal lights).
  - **Worse:** A val False-Go, yellow recall on B val, BDD100K and LISA, and twice the inference time.
- The gain from resolution concentrates in lamps of 20 px and over, and in detection. Lamps under 15 px do not improve at 1280 or at 1536.
- R stays the selected model, and 1536 is the highest resolution tried.
- Single seed, and the continuation was not pre-registered. So the B val gain and the A val False-Go loss are both candidates to recheck, not established effects.

Results: `runs/exp11_eval/result.json` (epoch 0) and `runs/exp11_eval_full/result.json` (5 epochs). A comparison page of F8, WF, R and S on every set was published as an artifact on 2026-11-19.

**AWSIM video, R and S (2026-11-19, qualitative, no labels).** The source of the earlier AWSIM videos (`AWSIM_only.mp4`, 924×1054, 60 fps, 7,076 frames) is no longer on disk. R and S were therefore run on a different AWSIM clip, `run_full_clean.mp4` (854×468, 30 fps, 5,560 frames). The counts below are not comparable with the J, F7 and F8 videos.

- Run with `tools/run_video_picto.py --arm F --run runs/exp10_R` and `--run runs/exp11_S`, each at its selected epoch 2 and threshold 0.35.
- Outputs are in `runs/awsim/run_full_clean_picto_{R,S}.mp4`, each with a `.jsonl` log.
- `tools/awsim_rs_compare.py` writes the counts to `run_full_clean_picto_RS_counts.json` and a side-by-side sheet to `run_full_clean_picto_RS_sample.jpg`.

| | R | S |
|---|---|---|
| Frames with any detection | 4,048 | 4,087 |
| Frames with a junction light | 3,920 | 3,013 |
| Junction detections | 6,771 | 3,970 |
| Of which red | 3,778 | 1,191 |
| Of which yellow | 168 | 53 |
| Straight red (frames) | 1,932 | 1,075 |
| Straight yellow (frames) | 148 | 47 |
| Straight green (frames) | 1,840 | 1,773 |
| No straight color (frames) | 1,640 | 2,665 |
| Arrow detections (right / left) | 0 / 0 | 112 / 7 |

- The straight color differs in 1,147 frames. In 849 of them, R reports red and S reports none.
- In 818 of those 849 frames, S still detects a red light but marks it as not part of the junction. S has 4,766 non-junction red detections, against 2,205 for R.
- In this simulator clip, the drop in red reports comes from S's junction head, not from its detector.
- On the sample frames, S misses the red stop light directly ahead, and R misreads a green light as yellow.
- This matches the A val False-Go rise of S on real data, but it is one clip with no labels.

**AWSIM camera view, R and S (2026-11-19, qualitative, no labels).** The source `AWSIM_ Autoware autonomous driving simulator_1080p60.mp4` (1920×1080, 60 fps, 7,076 frames) is available again. The camera view was cropped to x 70–990, y 60–960. This leaves out the window title and the bottom HUD, so the crop is tighter than the old `AWSIM_only.mp4` (x 72–996, y 24–1078).
- The new crop is saved as `data/awsim/AWSIM_cam.mp4`, 920×900.
- R and S were run on it at epoch 2 and threshold 0.35. Outputs are `runs/awsim/AWSIM_cam_picto_{R,S}.mp4` and `.jsonl`.
- `tools/awsim_rs_compare.py AWSIM_cam` writes `AWSIM_cam_picto_RS_counts.json` and `_RS_sample.jpg`.

| | R | S |
|---|---|---|
| Frames with any detection | 3,800 | 3,059 |
| Frames with a junction light | 3,568 | 2,808 |
| Junction detections | 5,245 | 3,953 |
| Straight green (frames) | 3,028 | 2,496 |
| Straight red (frames) | 340 | 171 |
| Straight yellow (frames) | 171 | 138 |
| No straight color (frames) | 3,537 | 4,271 |
| Arrow detections (straight / right / left) | 109 / 65 / 37 | 24 / 2 / 1 |
| Junction lamp height, median | 16.7 px | 20.6 px |
| Junction lamps under 15 px | 40 % | 18 % |

**Where R and S disagree**
- The straight color differs in 992 frames. Most are R green → S none (637) or R red → S none (171).
- In 854 frames R reports a color and S reports none. There, R's largest junction lamp has a median height of 12.9 px, and 63 % are under 15 px. R's median score on them is 0.43.
- In only 48 of the 171 red → none frames does S detect a red light that it marks as not in the junction.

**Reading**
- On this tighter crop, S mostly does not detect small, distant lamps that R reports. This differs from the 854px clip, where S detected the light but marked it as not in the junction.
- The sample sheet shows R reporting green on a far lamp of a few pixels, where S reports nothing. The images can't tell which model is right.
- A possible cause, not tested: the video is only 920×900. Upscaling it to S's 1536 input adds no detail, so small lamps look blurrier than S's training data at the same size.

## Main model and code reorganisation (2026-11-20)

**Main model.** R (Exp 10) is the model presented in the paper and the thesis: RF-DETR Medium with the factorized head,
class weights at power 0.5 and the label-swapping flip, input 1280, trained on DTLD + ATLAS + OpenLane-V2 A/B +
BDD100K with partial labels, epoch 2 at threshold 0.35. WF (1024) and S (1536), which differ from R only in resolution
(S also in batch 2 × accumulation 8, the same effective batch), are its resolution ablation.
- This choice was made after the Exp 10 and Exp 11 results were known. It departs from the Exp 10 decision rule,
  under which WF stays the recipe (R failed G1 and G2 on B val, and G4). It is a post-hoc choice and is reported as one.
- On the tune splits (three maneuvers), S has the highest selection mean: S 0.9399, R 0.9388, WF 0.9299. On the test
  sets S is above R on B val (ΔTSS R − S −0.018 [−0.029, −0.008]) and DTLD test (−0.007 [−0.011, −0.003]). R is
  the highest resolution with a latency of about 30 ms (29.5 ms against 54 ms for S).

**Three maneuvers, not four.** The Exp 6 to Exp 11 results above were computed while U-turn was still an output
maneuver (it copies left, so left counted twice). `tools/rescore_three_dirs.sh` re-scored only the joint-class
models, so those sections keep four-maneuver numbers. The reorganised evaluation scores straight, left and right.
Under three maneuvers the selection rule still picks epoch 2 at threshold 0.35 for R. Its scores are TSS 0.937 on A val
(US rule + topology mask), 0.918 on B val and 0.956 on DTLD test, against 0.934, 0.918 and 0.955 with four.

**Camera-only OpenLane-V2 scores.** The headline OpenLane-V2 numbers use the lane-topology mask, which needs the
frame's HD-map lanes. The evaluation now also reports the US rule without the mask: R scores TSS 0.927 on A val and
0.913 on B val camera only.

**Code.** The repository now holds only what R and its ablations need: `dtsu/` (labels, rules, dataset readers,
factorized head, model, evaluation, statistics), `scripts/` (labels, export, training, evaluation, video), `splits/`
(the fixed tune / held-out splits, written out from the earlier label files), `annotations/` (the OpenLane-V2
pictogram labels with their box keys). Removed from the tree and kept at tag `pre-cleanup`: the joint-class
("moves") models and their evaluations, the two-stage J+P arm, the audits, galleries and annotation tools, the
result snapshots of Exp 2 to Exp 7, and the thesis draft. The readers of ATLAS, BDD100K and VZC-TLD now read the
original dataset files, instead of the intermediate files of a separate project.

**Checks that the new code reproduces the old one.**
- Labels: the rebuilt OpenLane-V2 label files equal the earlier ones apart from the dropped U-turn entries (no frame
  changes its kept flag). The DTLD fit, tune and test records are identical to the Exp 6 export.
- Training set: `scripts/export_training_set.py` writes the same 57 classes, 85,991 images (same files and sizes) and
  281,254 annotations as `data/rfdetr_coco_exp8`, the set R was trained on.
- Model loading: on 16 A val frames and 8 ATLAS images, the new loader gives the cached detections of R exactly
  (score difference 0).
- Evaluation: R against WF, scored from the same cached detections by the code of tag `pre-cleanup` and by
  `scripts/evaluate.py`, over three maneuvers: all 803 values agree exactly. These include the selection grid
  optimum, TSS and False-Go with their CIs, every attribute metric and every detection cell.

## Publication protocol: seeds of R and M, R against M, camera-only headline (2026-11-22, written before any seed run or R-against-M evaluation)

**Purpose.** This section fixes what the paper reports and how. It was written before any of the runs or
evaluations below existed. R is fixed as the main model (see the section above); nothing below can change that
choice. Every result is reported whatever its direction.

**Decisions taken for the paper (2026-11-22).**
- The paper is a method paper on R. Its baseline is **M**, the mixed moves model of Exp 5 (`runs/rfdetr_m_mix`: joint
  `<moves>|<color>` classes, OpenLane-V2 A + B fit plus an equal-size D*_US fit sample, 1024 px, 4 epochs).
- Zou et al. enter only through the numbers in their paper. Their benchmark is not public, has about 1.4× our
  frames and scores four maneuvers, so that comparison is indicative and untested.
- **Headline OpenLane-V2 decoding is camera only**: the US rule without the lane-topology mask, for R and for M.
  The topology-masked scores are reported as a secondary "with map prior" variant.
- The resolution ablation (1024 / 1280 / 1536) and the recipe steps before it (F7, F8, W) stay single seed and
  are labelled as such.

**Seeds.**
- **R:** two more runs, seeds 1 and 2, with exactly R's command (`scripts/reproduce_R.sh`: `scripts/train.py`,
  `data/train_full`, 1280, 5 epochs, batch 4 × accumulation 4, lr 1e-4 / 1.5e-4, class weights 0.5,
  `--flip-swap`). R itself is seed 0. Outputs `runs/R_seed1`, `runs/R_seed2`.
- **M:** two more runs, seeds 1 and 2, with M's command and data (`tools/run_rfdetr_mix.sh` at tag `pre-cleanup`,
  `data/rfdetr_coco_fit_mix`). The original M was trained without a seed (`seed: null`); it counts as the first of
  the three M runs. Outputs `runs/M_seed1`, `runs/M_seed2`.
- **Selection, per run, never on a test set.** Each R run: the rule that selected R (the Exp 8 rule over three
  maneuvers: the epoch and threshold that maximise the mean of DTLD tune TSS with the junction rule and A / B tune
  TSS with the US rule and the topology mask). The mask is used here only on the tune splits, which are part of the
  training data. Each M run: the Exp 5 rule over three maneuvers (the mean of OpenLane tune TSS and D*_US tune TSS).
- If a run crashes it is resumed from its `last.ckpt` with the same settings, and the restart is logged. A run is
  not replaced by another seed because of its result.

**Primary comparison: R against M, in domain.** Per frame, each model's scores are averaged over its three runs.
Differences R − M are paired on the same frames, with a cluster bootstrap 95% CI (10,000 draws, seed 0):
segments on OpenLane-V2, sessions on DTLD.

| Cell | Frames | R decoding | M decoding |
|---|---|---|---|
| A val | 2,220 | US rule, camera only | its own decoder (`assign_scored`, US rule) |
| B val | 1,565 | US rule, camera only | the same |
| DTLD test, route labels, German rule | the frames in both the route labels and R's junction-scored set (11,126 in Exp 7) | junction rule | its own decoder, German rule |

- Metrics: TSS (primary) and False-Go, with Acc_color, Acc_pass, EM and False-Stop beside them. Three maneuvers.
- DTLD route labels are the target M was trained on. This cell was chosen because R's own junction labels favour
  R (Exp 7 checks); the junction-label cell is reported as secondary.
- The topology-masked OpenLane-V2 cells are reported for both models as secondary.
- Each model is also reported as the mean and standard deviation over its three runs.

**Secondary comparison: perception out of domain.** Center AP (as in the Exp 8 cells), recall and precision at each
run's threshold, and color accuracy on matched lights.
- Out of domain for both models: LISA test (day, night), S2TLD (720p, 1080p, 1080p horizontal), VZC-TLD (all,
  horizontal).
- In R's training domain, reported and marked as such, never used for a claim: BDD100K val (R trained on BDD100K
  train) and ATLAS test (R trained on ATLAS train).
- AP of a model = the mean of its three runs' APs. The CI of the R − M difference comes from a cluster bootstrap
  (2,000 draws) that recomputes every run's AP in each draw: blocks of 100 consecutive frames on LISA and S2TLD,
  sessions on VZC-TLD.
- VZC-TLD image decision (accuracy, False-Go, False-Stop), as in the VZC section, per model averaged over runs.

**Claims the paper may make**, each only if its condition holds:
- **"R is non-inferior to M in domain":** the lower CI bound of ΔTSS is above −0.01 on A val, B val and the DTLD
  route-label cell. **"R is better in domain"** on a cell: the lower bound is above 0 there.
- **"R has lower False-Go"** on a cell: the upper CI bound of ΔFalse-Go is below 0 there.
- **"R detects lights better out of domain":** the lower CI bound of the R − M difference of the mean center AP
  over LISA test, S2TLD and VZC-TLD (all lights) is above 0.
- Anything else is described as a point estimate with its CI, without a claim.

**Comparison with Zou et al.** R and M (run-averaged) are also scored over four maneuvers, with U-turn taking left as
in the original labels, on A val and on A + B val. These rows sit next to the paper's reported numbers, with no
test.

**Paper ablation table (single seed, existing runs, re-scored with the code of the reorganised repository).**
F7 (`runs/exp7_multi_F`), F8 (`runs/exp8_bdd_F`), W (`runs/exp9_W`), WF (`runs/exp9_WF`), R (`runs/exp10_R`),
S (`runs/exp11_S`), each at its selection under the Exp 8 rule over three maneuvers: camera-only and topology-masked
TSS and False-Go on A val and B val, DTLD test TSS, DTLD test pictogram macro recall, mean out-of-domain center AP,
latency. Each step is compared with the one before it (paired CIs as above), as a description, not a test.

**Error analysis** (exploratory): the False-Go cause breakdown of the Exp 7 checks (light missed, called not
relevant, wrong color, outvoted by another light) for R seed 0 and M on A val, B val and the DTLD route-label cell.

**Order.** Train R seeds 1 and 2, then M seeds 1 and 2 (about 45 h in all, in one queue). The new runs are
evaluated as they finish, but their scores are not read until all six runs exist. Meanwhile, the code for the
multi-run comparison and for M's evaluation in the reorganised repository is written and checked against the
`pre-cleanup` code on M's existing run (both must give the same numbers, as for R). The re-scored ablation rows
use only existing runs and can be read at once.

**Launch notes (2026-11-22).**
- The first launch (10:34) failed before the first step: `data/train_full` had been exported while `DATASETS` was a
  relative path, so its 18,851 BDD100K file names were relative and did not resolve from RF-DETR's image root. The
  queue then started seed 2, which failed the same way. Neither run trained; their logs are in
  `runs/failed_launch_2026-11-22/`.
- `scripts/export_training_set.py` was re-run (the current `dtsu/paths.py` makes `DATASETS` absolute). The new export
  has the same annotations, categories and image sizes, and every file name resolves to the same file as before. The
  old export is kept as `data/train_full_relpaths`. Seed 1 started at 10:38 from this export, and the queue
  (`runs/seeds_queue.sh`) now stops when a training run fails.

**Ablation rows re-scored (2026-11-22).** `scripts/evaluate.py` from the existing detection caches, with no new
inference: F8 against F7, W against F8, WF against W (`results/ablation_recipe/`). R against WF and S against R were
already in `results/ablation_resolution/`. The selection under the Exp 8 rule over three maneuvers gives the same
epochs and thresholds as before: F7 epoch 3 at 0.25, F8 epoch 3 at 0.45, W epoch 1 at 0.25.

## The baseline M back in the repository (2026-11-22)

M, the protocol's baseline, is again part of the code, so every number of the R-against-M comparison comes from the
public repository: `dtsu/moves.py` (joint classes, the decoder `assign_scored`, model loading and detection),
`dtsu/labels_dtld.py` (DTLD route labels under German rules and D*_US), `dtsu/moves_evaluation.py` and
`scripts/evaluate_moves.py` (the Exp 5 selection, M's maneuver cells, the perception cells of R, and the paired
R-against-M cells of the protocol), `scripts/export_moves_set.py` (M's training set), `scripts/train.py --moves`
(RF-DETR's own head) and `tests/test_moves.py`.
- **No private file is needed any more.** The route labels used `route_frames.json` of a separate project only for its
  `rule_ok` flag. The 701 frames where it is false (484 train, 217 test), the frames the manual route review rejects,
  are now `splits/dtld_route_review.json`.
- **Checks against the `pre-cleanup` code:**
  - labels: DTLD route labels and D*_US are identical frame by frame on train and test (kept flag, colors, boxes,
    session), as are the fit and tune splits;
  - training set: `scripts/export_moves_set.py` writes the same 46 classes, 29,954 / 4,687 images, annotations, files
    and image sizes as `data/rfdetr_coco_fit_mix`, the set M was trained on; the oracle TSS is 1 on both folders;
  - decoding: on M's cached detections of Exp 7 (epoch 1, threshold 0.25), the per-maneuver colors of every A val and
    B val frame equal those of the old decoder;
  - detection cells: from the same caches, center AP 0.641 on BDD100K val, 0.818 on VZC-TLD and 0.642 on ATLAS test,
    with recall, precision and color accuracy as in the Exp 7 and VZC sections.
- The queue of the protocol still trains M's seeds with the `pre-cleanup` code (as the protocol says); the new
  `scripts/train.py --moves` makes the same RF-DETR call and is the one documented in the README.
- `dtsu.model.load_model` takes a `device` (default `cuda`, as before), for CPU inference while the GPU trains.
- `scripts/make_examples.py` draws R and M side by side on frames drawn at random (seed 0) in and out of
  distribution (`docs/examples/`). It is qualitative and was not used for any decision.

### M on the AWSIM clip (2026-11-22)

M (`runs/rfdetr_m_mix`, epoch 1, threshold 0.25, German right-turn convention to match R's junction rule) was run
on the AWSIM camera-view crop (7,076 frames, CPU) with `scripts/run_video.py`. It was compared with R's existing
output (`AWSIM_cam_picto_R`, epoch 2, threshold 0.35) by `scripts/compare_videos.py` (outputs `runs/awsim/AWSIM_cam_R_vs_M*`).
Qualitative only: the clip has no labels.

| | frames with a used light | straight green | straight red | straight yellow | straight none |
|---|---|---|---|---|---|
| R | 3,568 | 3,028 | 340 | 171 | 3,537 |
| M | 2,798 | 2,578 | 105 | 86 | 4,307 |

The straight decision differs on 1,388 frames. Most of them are M missing a light R finds (green->none 723,
red->none 164, yellow->none 137): M finds fewer distant lights, which matches its lower input resolution (1024 vs 1280).
The safety-relevant case is red->green on 58 frames, in two episodes (around t = 23-25 s and t = 96 s). In both, the
crop shows a lit red circle and M gives the light a green "L+R+S" class (score 0.27-0.48), so the straight maneuver
reads green: a False-Go that R does not make. The opposite case (R green, M red) occurs on 1 frame.

**Seed queue paused (2026-11-22).** R seed 1 finished epoch 0 (10,751 s including validation, peak 9.09 GiB) and
stopped at step 6,799 of epoch 1 at about 14:24, with no traceback in its log; the queue's shell was gone too, as in
the Exp 11 resume attempts. The user then postponed the seed runs, so that the AWSIM/Autoware work and the other
thesis activities come first. Nothing was relaunched. When the seeds resume, `bash runs/seeds_queue.sh` continues
R seed 1 from the end-of-epoch-0 `last.ckpt` with unchanged settings. The protocol above is unchanged; no seed
result exists yet.

## ONNX export of R for Autoware / AWSIM (2026-11-22, no training)

`scripts/export_onnx.py` → `runs/onnx_R/` (not tracked; 150 MB): R at epoch 2 as a float32 ONNX graph, opset 17,
static input 1x3x1280x1280, outputs per query (300) of box, light score, pictogram, color and junction
probabilities, with no threshold and no NMS. It also writes the I/O contract (`model_config.json`), the torch-free
reference pipeline (`deploy/reference_infer.py`: preprocessing, threshold 0.35, NMS 0.5, junction > 0.5, junction
rule), 7 golden frames (5 AWSIM, 1 DTLD test, 1 OpenLane-V2 A val) with their input tensors and raw outputs,
`parity.json` and `SHA256SUMS.txt`. The ROS 2 node is not part of it.
- **Graph.** On the same input tensor, ONNX (onnxruntime, CPU) matches float32 PyTorch on the CPU within 4e-6 in
  score, box and junction on every golden frame. PyTorch on the GPU differs by up to 0.08 in score, because of TF32
  convolutions, so the reference is taken on the CPU.
- **Whole pipeline against the evaluated model** (PyTorch bf16 with the repository decoder): agreement of the
  per-maneuver colors, by the resize used before the ONNX graph.

  | Resize | AWSIM_cam.mp4, every 16th frame (443) | DTLD test, every 24th image (519) |
  |---|---|---|
  | Pillow bilinear (antialias) | 99.3-99.5%, 3 frames differ | 99.8-100%, 1 frame |
  | `cv2.INTER_LINEAR` | 98.6-98.9%, 6 frames | 99.6-99.8%, 2 frames |
  | `cv2.INTER_AREA` | 97.5-97.7%, 13 frames | 99.0-99.2%, 6 frames |

- `tests/test_deploy.py` checks that the reference decoder's rule and NMS equal `dtsu.rules.junction_rule` and
  torchvision's NMS.
- The package's own check (`reference_infer.py --check golden`, run in isolated mode outside the repository) passes on
  every golden frame.
- Not measured: TensorRT (fp32 / fp16) and latency on the Autoware machine.

## Multi-run comparison code, checked on M's existing run (2026-11-23, no training)

The code of the protocol's primary comparison is written, and its M side is checked against the `pre-cleanup` code
on M's existing run (`runs/rfdetr_m_mix`), as the protocol requires before any seed result is read.

**Code.**
- `dtsu/moves_evaluation.py`: `factorized_cells` and `moves_cells` give each model's per-frame scores on the cells
  of the protocol (A val and B val camera only, DTLD test against the route labels on the common frames, and the
  topology-masked A val and B val as secondary). `average_runs` averages a model's runs per frame (TSS contribution
  and False-Go numerator; the denominator is the ground truth's). `compare_runs` pairs the two run-averaged models
  with the existing cluster bootstrap (`stats.paired_frames`: 10,000 draws, seed 0, segments on OpenLane-V2,
  sessions on DTLD) and adds each model's mean and sample standard deviation over its runs. `claims` applies the
  protocol's three bounds. `paired_cells` (used by `evaluate_moves.py --reference`) is now these two functions.
- `scripts/compare_runs.py` reads the result file of every run (run folder, cache, selection). It refuses fewer
  than three runs per model unless `--min-runs` is lowered for a check, so no R-against-M score is computed by
  accident before all six runs exist.
- `evaluate.py` and `evaluate_moves.py` also write the run's detection cache to their result file.
  `evaluate_moves.py` decodes the DTLD route cell from the same all-frames cache as the paired cells, so the route
  test is inferred once.
- `tests/test_runs.py` checks the algebra: one run per model gives exactly the single-run comparison; the averaged
  model's metrics, ΔTSS and ΔFalse-Go are the means over runs; runs on different frames are refused; and the claims
  follow the protocol's bounds.

**M, evaluated with the new code** (`results/M.json`, `--no-detection`). The Exp 5 rule over three maneuvers
selects **epoch 1 at 0.25**, the selection M was published with (tune mean 0.9356; epochs 0, 2 and 3: 0.9271, 0.9335,
0.9308). These are M's own scores, known since Exp 5 and Exp 7. No R-against-M cell was computed.

| Cell (three maneuvers) | Frames | TSS | False-Go |
|---|---|---|---|
| A val, camera only | 2,220 | 0.920 | 0.137 |
| A val, topology mask | 2,220 | 0.927 | 0.139 |
| B val, camera only | 1,565 | 0.921 | 0.104 |
| B val, topology mask | 1,565 | 0.920 | 0.104 |
| D*_US test | 11,501 | 0.953 | 0.050 |
| DTLD test, route labels | 11,347 | 0.949 | 0.040 |

**Check against the `pre-cleanup` code** (`runs/M_eval/check/`, untracked). The tree of tag `pre-cleanup`
(`runs/pre_cleanup_code`) scored the detections cached by the new code, with the label files of that time
(`data/labels_pre_cleanup`), matched frame by frame by image path:
- M's selection with `tools/eval_rfdetr.py` (preset `mixed`), M's metrics on every cell above, and the Exp 7
  paired cells of `tools/eval_exp7.py` / `tools/exp7_checks.py`. To pair them with a factorized model without
  reading R against M, the reference is F7 (epoch 4, threshold 0.20, its Exp 7 caches), with the US rule camera
  only and with the topology mask on A val and B val, and the junction rule against the route labels on DTLD test
  (11,126 common frames).
- New: `scripts/compare_runs.py --min-runs 1` with F7 as "R" and M.
- **All 229 values agree exactly**: the whole selection grid (4 epochs × 19 thresholds), the selected epoch and
  threshold, the six metrics of M on the six cells, and per paired cell both models' metrics, ΔTSS and ΔFalse-Go
  with their CIs and the frame counts.
- Inference: the new loader's detections of M at epoch 1 equal the Exp 7 caches of M exactly on A val and B val
  (every frame, score and box difference 0). On the 11,494 DTLD test frames of the Exp 7 cache, 11,488 are
  bit-identical. The other 6 differ by at most 0.0023 in score, because bf16 inference ran in different batches
  (the Exp 7 cache was inferred on the junction-scored frames only). No maneuver color changes on any frame at 0.25.

**Not covered.** The R side of the comparison uses the decoders checked on R in the reorganisation entry, but R and
M were not paired here. The out-of-domain perception comparison over runs (AP recomputed per run in each bootstrap
draw) and the four-maneuver rows for Zou et al. are separate steps and not written yet.
