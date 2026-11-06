# Replication plan — "One Perception, All Maneuvers" (Zou et al., arXiv 2609.32316)

> **Final write-up: [REPORT.md](REPORT.md)** (results vs the paper, per-direction breakdown, all deviations).

> **Update 2026-10-26: data scope is now OpenLane-V2 subset_A + subset_B** (front cameras
> `ring_front_center` / `CAM_FRONT`). With US driving rules applied to both subsets, including the
> Singapore scenes of subset_B, the label distribution reproduces Fig. 3 closely: straight red/green
> 37.3/53.9 vs 40.7/49.7, right none 96.7 vs 96.9, U-turn = left 99.9%. Mirroring the rules for
> left-hand traffic does not match (right none 85.7%). So the paper's benchmark is very likely A+B
> with US rules.
>
> Kept frames:
>
> | | train | val | total |
> |---|---|---|---|
> | subset_A | 10,141 | 2,220 | |
> | subset_B | 6,805 | 1,564 | |
> | A+B | 16,946 | 3,784 | 20,730 |
> | paper | 23,904 | 5,147 | 29,051 |
>
> We keep about 71% as many frames as the paper. The remaining difference is unexplained.
> The left-hand-traffic variant of the labels is in `data/labels/variants/`.
>
> **Protocol.**
> - Train one model on A+B train combined.
> - Report val metrics on subset_A val (main, compared to the paper's tables), on subset_B val, and on A+B combined.
> - Val is the evaluation split, so the last epoch is reported; no checkpoint is selected on val.
>
> **Images.** Each image is resized so its long side is 1024 px (aspect kept), then padded to a multiple of 32:
> - A becomes a 1024×800 portrait canvas.
> - B becomes a 576×1024 landscape canvas.
> - Each batch holds a single shape.
>
> **Label spot-check (2026-10-26).** I looked at 30 val frames, 6 per tricky case: turn arrows, straight=none, dedicated right, crossing the junction, conflicts.
> - It found one bug: when the ego stands at the stop line, the junction lane starts slightly behind it, and the builder skipped to the next junction.
> - The fix (`ENTERING_X = -3 m`) changed about 140 subset_B frames and 2 subset_A frames.
> - Kept frames are now 16,947 train / 3,785 val.
> - The rest looked right or plausible.
> - Conflicts are mostly a near light plus a farther intersection's light, and stay dropped.
> - In Singapore a green right arrow gives right = none. That follows from the US-rule convention that matches Fig. 3.
>
> **Tuning split.** `data/labels/tune/` holds 10% of train scenes (70 per subset): 1,185 A + 785 B kept frames. Hyperparameters are tuned on these and never on val.
>
> **Tuning (2026-10-28).** Three 4-epoch runs, trained on the fit split and scored on the tune split. Last-epoch combined scores:
>
> | run | λ (C/P/D) | clip | TSS | EM | False-Go | False-Stop |
> |---|---|---|---|---|---|---|
> | t1_default | 1/1/1 | 0.1 | crashed (NaN, epoch 2); epoch 1 = 0.563, always "go" | | | |
> | t2_clip1 | 1/1/1 | 1.0 | 0.642 | 0.379 | 0.256 | 0.155 |
> | **t3_det0.2** | **1/1/0.2** | **0.1** | **0.841** | **0.756** | **0.195** | **0.065** |
>
> - t3 is chosen. With full detection weight, the detection loss (~20, summed over 6 decoder layers) swamps the color/pass gradients.
> - t3's scores swing between epochs: 0.874 → 0.707 → 0.818 → 0.841. They recover as the cosine lr decays.
> - t1's NaN was an fp16 overflow in the ROI/direction branch; its weights were still finite. That branch now runs in fp32. Non-finite batches are skipped, and training stops after 20 in a row.
> - Full run: `bash tools/run_full.sh` → `runs/swin_t_AB/` (resumable).
>
> **Full run v1 failed (kept in `runs/swin_t_AB_v1_nan/`).**
> - Epochs 0–13 had no bad batches. Then fp16 overflow in the detector decoder's cross-attention skipped 1,328 / 2,591 batches in epochs 14 / 15. The run stopped in epoch 16 after 20 bad batches in a row.
> - Val TSS on A swung between 0.62 and 0.86 from epoch to epoch (best 0.863 at epoch 11; paper 0.880). Epoch 7 collapsed to "go" everywhere (False-Go 0.92).
> - Root cause: timm Swin `features_only` outputs have no LayerNorm, so feature scale drifted. Activations reached ~1e5 in the decoder, ~6e5 at the ROI projection and ~3e6 in the direction attention.
> - Fix:
>   - one LayerNorm per Swin output stage (standard in Swin detection codebases);
>   - the detector runs in fp32 too, about 25% slower.
> - Being checked with tune run t4 (t3 settings + fix) before restarting the full run.
>
> **Pipeline check (2026-10-28, `check_pipeline.py`).**
> - All 169 Swin tensors equal timm's ImageNet weights.
> - With dynamic shifted-window masks, outputs at our canvases equal timm models built for exactly 1024×800 / 576×1024 (max diff 0).
> - Stage shapes: 256×200 → 32×25 (A) and 144×256 → 18×32 (B).
> - The ImageNet classifier on frames gives road-vehicle classes.
> - Dataset boxes drawn on the resized, padded tensors sit on the lights in A and B.
> - With the fix, t4 activations peak at ~11 (FPN), ~2.5 (decoder) and ~200 (ROI/direction tokens, fp32).
> - t4 tune TSS: 0.881 → 0.894 at epochs 0–1, with no bad batches (t3: 0.874 → 0.707).
> - The full run starts automatically after t4 if t4's final TSS ≥ 0.841 and it has no bad batches.
>
> **t4 result.**
> - Combined tune TSS by epoch: 0.881 / 0.894 / 0.895 / **0.901**. EM 0.843, False-Go 0.116, False-Stop 0.034.
> - A_tune 0.893, B_tune 0.913. No bad batches; the curve is steady (t3 final: 0.841).
>
> **Speed.**
> - Under AMP the detector now runs in bf16 (fp32's range, so no overflow). The ROI/direction branch stays fp32, with TF32 on.
> - `runs/bench_bf16.log` times the change and checks bf16-vs-fp32 prediction agreement on t4 before the full run starts.
> - Result: A 0.460 → 0.350 s/step (1.31×), B 0.334 → 0.251 s/step (1.33×). On the t4 model, 99.84% of direction predictions are equal to all-fp32, and tune TSS is 0.9000 vs 0.9008.
> - Full run v2 started 2026-10-28 10:38 at ~3.1 it/s (~25 min/epoch incl. eval), so it should end around 20:40.
>
> **Final result (run v2, `runs/swin_t_AB/`).**
> - Stopped by request after epoch 14 of 24. The learning rate was still ~31% of peak (cosine not finished).
> - The color/pass training loss was ~0.001, while val had been flat since epoch 4.
> - The reported model is the last finished epoch (14), not chosen on val. Its metrics are in `final_metrics.json` and the checkpoint in `last.pt`.
>
> | | Acc_color | Acc_pass | EM | TSS | False-Go | False-Stop |
> |---|---|---|---|---|---|---|
> | paper (Subset-A) | 0.9019 | 0.9356 | 0.8031 | 0.8802 | 0.1372 | 0.0416 |
> | **ours, A val** | **0.9098** | **0.9341** | **0.8221** | **0.8887** | **0.1642** | **0.0352** |
> | ours, B val | 0.9334 | 0.9468 | 0.8518 | 0.9106 | 0.1208 | 0.0214 |
> | ours, A+B val | 0.9196 | 0.9394 | 0.8343 | 0.8978 | 0.1431 | 0.0298 |
>
> - A val TSS over epochs 4–14: mean 0.898, range 0.887–0.908.
> - A val False-Go over epochs 4–14: mean 0.146, range 0.105–0.212.
>
> Training command, once a CUDA build of PyTorch is installed:
>
> ```
> python -m dtsu.train \
>   --train-set data/labels/subset_A_train.jsonl data/OpenLane-V2/subset_A \
>   --train-set data/labels/subset_B_train.jsonl data/OpenLane-V2/subset_B \
>   --val-set A data/labels/subset_A_val.jsonl data/OpenLane-V2/subset_A \
>   --val-set B data/labels/subset_B_val.jsonl data/OpenLane-V2/subset_B \
>   --out-dir runs/swin_t_AB --amp
> ```

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

**Scope.** Replicate only the paper's best setting: the direction-aware model ("Ours") with a Swin-Tiny
backbone, trained and evaluated on Subset-A. Targets from Tables 1–3:

| Acc_color | Acc_pass | EM | TSS | False-Go | False-Stop |
|---|---|---|---|---|---|
| 0.9019 | 0.9356 | 0.8031 | 0.8802 | 0.1372 | 0.0416 |

Out of scope: the other backbones, the MLLM baselines, the structured pipelines and Subset-B
(whose results are only in a supplement we don't have).

## Task and metrics

From one front-view image, predict a color c ∈ {none, red, yellow, green} and a passability
p ∈ {stop, go} for each maneuver d ∈ {straight, left, right, U-turn}.

- Acc_color and Acc_pass are averaged over all frame × direction states.
- EM: a frame counts only if all 4 colors and all 4 passabilities are correct.
- TSS = (Acc_color + Acc_pass + EM) / 3.
- False-Go = #(gt stop, pred go) / #(gt stop), pooled over directions.
- False-Stop = #(gt go, pred stop) / #(gt go), pooled over directions.

## Model (paper, 50.1M params)

1. Swin-T + 256-d FPN → F. Global token g = pooled F.
2. A lightweight DETR-style traffic-light detector predicts boxes b̂_j.
3. ROI token t_j = φ_r(RoIAlign(F, b̂_j)) + φ_b(b̂_j), with φ linear.
4. T = [g, t_1..t_K]. Four learnable direction queries: h_d = MHA(q_d, T, T).
5. Heads: color (4-way) and pass (2-way) on each h_d.
6. Loss: L = λc·CE_color + λp·CE_pass + λd·L_DETR (cls + L1 + GIoU, Hungarian matching).
7. Training: AdamW with a cosine schedule, fixed resize that keeps the aspect ratio. Hyperparameters are not given.

## Data: OpenLane-V2 subset_A, front camera only

- Source: Argoverse 2, with 7 ring cameras at 2 Hz.
  - Front camera = `ring_front_center`, 1550 × 2048 (portrait).
  - Traffic elements are annotated only on the front view, as 2D boxes.
- Splits from the official `data_dict_subset_A.json`:
  - train: 700 scenes / 22,477 frames
  - val: 150 scenes / 4,806 frames, used as the test set (same as the paper)
  - test: annotations not usable
- Downloads needed:
  - `OpenLane-V2 info` (~8.8 GB)
  - image train `image_0..6` (~100 GB)
  - image val `image_7` (~21 GB)
- The image tars bundle all 7 cameras, so every tar has to be downloaded. Only
  `*/image/ring_front_center/*` is extracted (~1/7 of the volume). Each tar can be deleted after extraction.
- Develop the label builder first on the ~300 MB `sample` package.

## Gaps and ambiguities (my interpretation)

| # | Gap | Interpretation |
|---|-----|----------------|
| 1 | No code or labels released | Rebuild the labels from OpenLane-V2 annotations. Exact numbers won't match. |
| 2 | Frame counts | The paper says 23,904 train / 5,147 val, but subset_A has only 22,477 / 4,806 *before* filtering. So these counts can't be subset_A alone. They fit subset_B after filtering (27,968 / 6,019) or A+B combined. We use subset_A and report our filtered counts. |
| 2b | Measured (label builder v1, 2026-10-26) | subset_A train+val keeps 12,361 / 27,283 frames (train 10,141, val 2,220). 13,258 frames have no colored light at all. The color mix is greener than Fig. 3 (straight: 28% red / 64% green, vs. 41% / 50% in the paper). The raw subset_A light annotations are also green-heavy (59% green). This again suggests the paper's benchmark mixes in other data. |
| 3 | Passability | Per the Fig. 2 legend: red→stop; green, yellow and none→go. |
| 4 | U-turn | Inherits the left-turn signal unless a dedicated light exists. In Fig. 3 the U-turn distribution ≈ left. |
| 5 | Light-to-maneuver association | Use `topology_lcte`. Classify each centerline S/L/R/U by heading change. Keep only centerlines on the ego approach: start ahead of ego, aligned heading, laterally close. Lights linked only to other lanes are ignored. |
| 6 | Frame filtering | Keep frames with ≥1 light linked to an ego-approach lane. Drop conflicting-color cases. Manual QC cannot be replicated. |
| 7 | Hyperparameters | Defaults: λ = 1/1/1, LR 1e-4 (backbone 1e-5), short side ~640, light DETR decoder with ~50 queries, K = 16 ROI tokens. Tune on a held-out train split. |

## Phases

0. **Environment.** venv with CPU PyTorch, torchvision, timm, scipy and Pillow. Switch to a CUDA 12.8+ build when the GPU is free (the RTX 5070 needs it).
1. **Sample data.** Download the `sample` package and inspect the JSON schema.
2. **Label builder** (`dtsu/labels.py`). Per-frame record: 4 × (color, pass), front image path, GT light boxes. Checks: Fig. 3-style stats and visual spot-checks like Fig. 2.
3. **Metrics** (`dtsu/metrics.py`). All six metrics, unit-tested.
4. **Model** (`dtsu/model.py`). Check that the parameter count is ~50M.
5. **CPU smoke test.** Forward/backward passes and overfit ~50 frames.
6. **Full data.** Download subset_A info + images and extract the front camera only.
7. **Training and evaluation on val.** Needs the GPU.

Phases 0–6 don't need the GPU.

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
