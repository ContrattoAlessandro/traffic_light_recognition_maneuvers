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
