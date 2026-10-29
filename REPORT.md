# Replication report: "One Perception, All Maneuvers" (direction-aware model, Swin-Tiny)

**Paper:** Zou et al., *One Perception, All Maneuvers: Directional Traffic Signal Understanding for
Maneuver-Level Signal Intent Prediction* (arXiv 2609.32316).
**Scope:** the paper's best setting only, the direction-aware model ("Ours") with a Swin-Tiny
backbone, using the front camera of OpenLane-V2. Not replicated: the paper's other backbones, MLLM
baselines, structured baselines (TLD-READY / TopoPoint / One-Stage). Added: a backbone ablation with
ConvNeXtV2-Tiny (§6).
**Date:** 2026-10-28 (backbone ablation: 2026-10-29).

## 1. Summary

| | Acc_color ↑ | Acc_pass ↑ | EM ↑ | TSS ↑ | False-Go ↓ | False-Stop ↓ |
|---|---|---|---|---|---|---|
| Paper, Subset-A (Tables 1–3) | 0.9019 | 0.9356 | 0.8031 | 0.8802 | 0.1372 | 0.0416 |
| **Ours, subset_A val** | **0.9098** | **0.9341** | **0.8221** | **0.8887** | **0.1642** | **0.0352** |
| Δ (ours − paper) | +0.0079 | −0.0015 | +0.0190 | **+0.0085** | +0.0270 | −0.0064 |
| Ours, subset_B val | 0.9334 | 0.9468 | 0.8518 | 0.9106 | 0.1208 | 0.0214 |
| Ours, A+B val | 0.9196 | 0.9394 | 0.8343 | 0.8978 | 0.1431 | 0.0298 |

- **The main result reproduces.** On subset_A val we get TSS 0.889 vs the paper's 0.880, with higher color accuracy and EM and the same go/stop accuracy.
- **Weak point, False-Go:** 0.164 vs 0.137, i.e. predicting "go" for a stop direction more often. It fluctuates strongly between epochs (0.105–0.212 over epochs 4–14, mean 0.146), so the gap is partly where training stopped.
- **The numbers aren't strictly comparable.** The paper's labels are not public; ours are rebuilt from the OpenLane-V2 annotations, and the kept-frame counts differ (§5).
- **Swapping the backbone for ConvNeXtV2-Tiny does not help** (§6). With everything else fixed, mean TSS over epochs 4–14 is 0.891 vs 0.898 on subset_A val, and False-Go is higher (0.191 vs 0.146).

## 2. Task and metrics

From one front-camera image, the model predicts for each maneuver d ∈ {straight, left, right, U-turn}:
- a color c ∈ {none, red, yellow, green};
- a passability p ∈ {stop, go}, with red → stop and everything else → go (Fig. 2 legend).

The metrics follow the paper's Eqs. 4–7:
- Acc_color and Acc_pass are pooled over frame × direction.
- EM counts a frame only if all 4 colors and all 4 passabilities are right.
- TSS = (Acc_color + Acc_pass + EM) / 3.
- False-Go = P(pred go | gt stop) and False-Stop = P(pred stop | gt go), pooled over directions.

The implementation is in `dtsu/metrics.py`, with unit tests.

## 3. Setup

### Data
- OpenLane-V2 subset_A (Argoverse 2, `ring_front_center`, 2048×1550 portrait) and subset_B (nuScenes, `CAM_FRONT`, 900×1600). Front camera only.
- The official val split is the evaluation set, as in the paper; the test split has no usable annotations.
- Labels come from our label builder (`dtsu/labels.py`, §5.1).

| Kept frames | train | val |
|---|---|---|
| subset_A | 10,141 / 22,477 | 2,220 / 4,806 |
| subset_B | 6,806 / 27,968 | 1,565 / 6,019 |
| A+B | 16,947 | 3,785 |
| paper | 23,904 | 5,147 |

**Protocol.**
- One model is trained on A+B train combined.
- It is evaluated on subset_A val (main, compared with the paper), on subset_B val, and on A+B together.

### Model (`dtsu/model.py`, `dtsu/detr.py`), 52.0M parameters (paper: 50.1M)
1. **Backbone.** Swin-Tiny (`swin_tiny_patch4_window7_224`, ImageNet-1k weights, timm) with a LayerNorm on each output stage. Its features feed a 256-d FPN (P2–P5).
2. **Global token g.** The mean of the pooled FPN levels, followed by a linear layer.
3. **Detector.** A lightweight DETR (6-layer encoder on P5; 6-layer decoder over P3 + P4 + encoded P5; 50 queries; 3 light classes red/green/yellow).
4. **ROI tokens.** Each of the top-16 detected boxes gives an ROI token t_j = φ_r(RoIAlign(P2, b_j)) + φ_b(b_j), with linear φ (Eq. 9).
5. **Direction attention.** Four learnable direction queries attend to T = [g, t_1..t_16] through multi-head attention (8 heads), followed by LayerNorm and an FFN (Eq. 11).
6. **Heads.** A color head (4-way) and a pass head (2-way) for each direction (Eq. 12).

### Training (`dtsu/train.py`, `tools/run_full.sh`)

| Setting | Value |
|---|---|
| Loss | λc·CE_color + λp·CE_pass + λd·L_DETR (Eq. 15), λ = 1 / 1 / 0.2. DETR loss: Hungarian matching, class + 5·L1 + 2·GIoU, auxiliary losses on all decoder layers |
| Optimizer | AdamW, lr 1e-4 (backbone 1e-5), weight decay 1e-4, cosine schedule with 500-iteration warmup, gradient clip 0.1 |
| Input | Long side resized to 1024 (aspect kept), padded to a multiple of 32: A → 1024×800, B → 576×1024. Each batch holds one canvas shape |
| Augmentation | Brightness/contrast only. No flips (they swap left/right) and no hue shifts (they change light colors) |
| Batch | 4 |
| Precision | Mixed: fp16 backbone/FPN, bf16 detector, fp32 ROI/direction branch, TF32 |
| Schedule | 24 epochs planned; **stopped after epoch 14** (§5.3). The reported model is the last finished epoch, not selected on val |
| Hardware | RTX 5070 12 GB, PyTorch 2.14.1 + CUDA 13.0, timm 1.0.30. About 24 min per epoch |

### Hyperparameter tuning
- Tuning used a held-out 10% of train scenes (70 per subset; 1,185 A + 785 B kept frames); val was never used.
- Each run lasted 4 epochs. The table gives combined tune-split TSS at the last epoch.

| Run | λ (C/P/D) | Clip | Other | Tune TSS |
|---|---|---|---|---|
| t1 | 1/1/1 | 0.1 | | crashed with NaN in epoch 2; collapsed to "always go" |
| t2 | 1/1/1 | 1.0 | | 0.642 |
| t3 | 1/1/0.2 | 0.1 | | 0.841 (unstable: 0.874 → 0.707 → 0.818 → 0.841) |
| **t4** | **1/1/0.2** | **0.1** | **+ per-stage LayerNorm** | **0.901** (steady: 0.881 → 0.894 → 0.895 → 0.901) |

## 4. Results

### 4.1 Overall, per epoch (subset_A val)
- TSS from epoch 4 to 14 stayed between 0.887 and 0.908 (mean 0.898). It was above the paper's 0.880 in every epoch from epoch 0 on.
- The training color/pass loss reached ~0.001, while val stayed flat, so further epochs mostly fit the training set.
- Full per-epoch metrics: `runs/swin_t_AB/val_log.jsonl`.

### 4.2 Per direction
- The paper reports no per-direction results, so this section is ours only. Source: `tools/eval_directions.py` → `runs/swin_t_AB/per_direction.json`.
- "Majority" is the accuracy of always predicting that direction's most frequent color, given as a reference point.

**subset_A val (2,220 frames)**

| Direction | GT none/red/yellow/green % | Acc_color | Acc_pass | False-Go | False-Stop | #stop | Majority Acc_color |
|---|---|---|---|---|---|---|---|
| straight | 3.9 / 28.6 / 5.0 / 62.5 | 0.865 | 0.921 | 0.132 | 0.058 | 636 | 0.625 |
| left | 0.9 / 32.8 / 4.1 / 62.2 | 0.900 | 0.910 | 0.172 | 0.050 | 729 | 0.622 |
| right | 97.7 / 0.6 / 0.1 / 1.5 | 0.978 | 0.996 | 0.643 | 0.000 | 14 | 0.977 |
| U-turn | 0.9 / 33.1 / 4.1 / 61.9 | 0.896 | 0.910 | 0.176 | 0.048 | 734 | 0.619 |

**subset_B val (1,565 frames)**

| Direction | GT none/red/yellow/green % | Acc_color | Acc_pass | False-Go | False-Stop | #stop | Majority Acc_color |
|---|---|---|---|---|---|---|---|
| straight | 3.1 / 41.2 / 2.2 / 53.6 | 0.911 | 0.928 | 0.120 | 0.038 | 644 | 0.536 |
| left | 0.4 / 42.2 / 2.9 / 54.4 | 0.931 | 0.939 | 0.104 | 0.030 | 661 | 0.544 |
| right | 95.8 / 2.4 / 0.3 / 1.5 | 0.953 | 0.976 | 0.892 | 0.003 | 37 | 0.959 |
| U-turn | 0.4 / 42.2 / 2.9 / 54.4 | 0.939 | 0.944 | 0.095 | 0.027 | 661 | 0.544 |

**Per-class color recall, straight direction**

| Recall | none | red | yellow | green |
|---|---|---|---|---|
| A val | 0.20 (87) | 0.87 (636) | 0.42 (110) | 0.94 (1,387) |
| B val | 0.19 (48) | 0.89 (644) | 0.85 (34) | 0.97 (839) |

**Observations**
- **Straight, left and U-turn are learned well**, 25–40 points above the majority baseline.
  - Left and U-turn behave almost identically, as expected: U-turn inherits left in 99.9% of labels.
  - Straight is the hardest of the three on A (0.865 color). Its "none" class (no light governs straight, e.g. T-junctions or turn-only lanes) is rarely recognized: recall 0.19–0.20.
- **The right turn is not learned.** Its color accuracy equals the majority baseline (A: 0.978 vs 0.977; B: 0.953 vs 0.959). The model predicts "none" almost always. Right-turn red is rare (14 frames in A val, 37 in B), so it gets False-Go 0.64 (A) and 0.89 (B).
  - These cases are few, so they barely affect pooled False-Go, but they are the most safety-relevant errors.
  - The paper's Fig. 3 has the same imbalance (right "none" 96.9%). The paper doesn't say how its model handles it.
- **Yellow is weak on A.** Straight yellow recall is 0.42 on A vs 0.85 on B; yellow is mostly confused with red or green. A val has 110 yellow straight frames, more than B, so this isn't only a sample-size effect. A likely contributor is Argoverse yellow lights looking like red/amber.
- **Passability is almost exactly a function of the predicted color.** Pass = rule(color) in 99.4–99.9% of predictions, even though the two heads are separate. So go/stop errors come from color errors, mainly red ↔ green/yellow.
- **subset_B is easier than subset_A** across all directions (TSS 0.911 vs 0.889).

### 4.3 Label distribution vs the paper's Fig. 3 (A+B, train + val, kept frames)

| Direction | none | red | green | yellow |
|---|---|---|---|---|
| straight (ours / paper %) | 6.5 / 7.0 | 36.9 / 40.7 | 53.9 / 49.7 | 2.7 / 2.5 |
| left | 0.9 / 7.2 | 40.8 / 41.3 | 55.5 / 48.8 | 2.9 / 2.8 |
| right | 96.6 / 96.9 | 1.8 / 0.3 | 1.5 / 2.7 | 0.1 / 0.2 |
| U-turn | 0.8 / 7.2 | 40.9 / 41.3 | 55.4 / 48.8 | 2.9 / 2.8 |

- Most of the distribution matches. That supports our reading that the paper's benchmark is A+B with US rules (§5.1).
- The main mismatch is left/U-turn "none": 0.9% for us vs 7.2% in the paper. In the paper, left "none" ≈ straight "none", which suggests its left labels inherit straight's "none" more often than ours do.

## 5. Assumptions and deviations

### 5.1 Labels (largest source of difference)
- The paper's labels and code are not released. We rebuilt per-frame, per-direction labels from OpenLane-V2:
  - lane ↔ traffic-element links (`topology_lcte`);
  - lane ↔ lane links (`topology_lclc`);
  - lane centerline geometry;
  - traffic-element colors (the `attribute` field: 1 red, 2 green, 3 yellow, 4–12 arrows).
- **Our rules:**
  1. **Maneuver per lane.** Follow linked lanes to the first junction lane being entered (it starts beyond 3 m behind the ego). Classify its heading change: within 30° is straight, beyond 135° is U-turn, otherwise left or right.
  2. **Relevant lights.** Keep lights linked to lanes that are aligned with the ego (within 60°) and at the nearest junction (within 10 m). If the ego has passed the stop line, use approach lanes ending within 30 m behind it.
  3. **Lights to directions.**
     - Straight takes straight/generic lights.
     - Left takes left lights, otherwise inherits straight.
     - U-turn takes U-turn lights, otherwise inherits left.
     - Right takes only dedicated right-only lights, otherwise "none".
     - The most specific light wins. Ties with different colors mark a conflict, and the frame is dropped (596 frames).
  4. **Frame filter.** A frame is kept if at least one relevant colored light exists and there is no conflict.
- **Traffic side.** US (right-hand) rules everywhere, including the Singapore (left-hand traffic) scenes in subset_B. This reproduces Fig. 3; mirrored rules give right "none" 85.7% instead of the paper's 96.9%. A side effect: in Singapore, a green right-turn arrow yields right = none.
- **Frame count.** We keep 20,732 frames vs the paper's 29,051 (71%). The reason is unknown: the paper's filtering rules aren't specified, and it also did manual QC we can't reproduce.
- **Spot-check.** We checked 30 val frames by eye across 5 tricky cases. It found one bug (the stop-line case), which we fixed. Everything else looked correct or plausible; no full manual QC.

### 5.2 Data scope
- The paper says Subset-A, but its frame counts (23,904 + 5,147) exceed subset_A's total before filtering (22,477 + 4,806). So the paper cannot be using subset_A alone, and A+B fits its numbers and Fig. 3.
- We therefore train on A+B and report subset_A val as the main comparison.

### 5.3 Training
- **Hyperparameters.** The paper gives none beyond AdamW + cosine + fixed aspect-preserving resize. Everything in the training table (§3) is our choice, tuned only on the held-out train split.
- **Detection-loss weight 0.2.** With weight 1, the DETR loss (~20 summed over 6 decoder layers) swamped the color/pass gradients, and training collapsed (t1, t2).
- **Per-stage LayerNorm on Swin outputs.** This is not in the paper, but it is standard in Swin detection setups. Without it, activations drifted to ~10^5–10^6, fp16 overflowed and val swung between 0.62 and 0.86. The first full run failed this way (kept in `runs/swin_t_AB_v1_nan/`).
- **Precision.** bf16 for the detector and fp32 for the ROI/direction branch, for numerical stability. On the t4 model, 99.84% of direction predictions match all-fp32.
- **Stopped early.** The run stopped at epoch 14 of 24, at the user's request. The learning rate was still ~31% of peak, so the cosine schedule hadn't finished; the last epochs might have reduced the epoch-to-epoch fluctuation. The reported checkpoint is the last finished epoch, not selected on val.
- **Detector quality.** The auxiliary detector learns slowly (DETR loss went from ~57 to ~11 over 14 epochs). So the ROI tokens likely carry only part of the localized evidence the paper intends.

### 5.4 Model details we had to choose
- Global token pooling (mean over levels), ROI level (P2, 7×7), K = 16, 50 detector queries.
- Decoder memory: P3 + P4 + encoded P5.
- One attention layer (+ FFN) for the direction queries; MLP heads.

Our model has 52.0M parameters vs the paper's 50.1M, so the paper's design differs slightly.

## 6. Backbone ablation: ConvNeXtV2-Tiny

**Question.** Does replacing Swin-Tiny with ConvNeXtV2-Tiny improve the model, all else equal?

### Setup (`tools/run_convnextv2.sh` → `runs/convnextv2_t_AB/`)
- **Backbone:** `convnextv2_tiny.fcmae_ft_in1k` (timm): FCMAE self-supervised pretraining, then ImageNet-1k fine-tuning (83.0% top-1).
  - It matches Swin-T `ms_in1k` in size (27.9M vs 27.5M backbone parameters; model total 52.3M vs 52.0M).
  - It matches the channels (96/192/384/768) and strides (4–32).
  - Its pretraining data is ImageNet-1k only, like Swin-T's. The in22k variants were avoided so that pretraining data is not a second variable.
- **Kept identical to §3:** labels, data, input size and augmentation, FPN, detector, ROI/direction branch, loss weights, optimizer, learning rates, clip, batch, precision, seed.
  - Also identical: the 24-epoch cosine schedule, stopped after epoch 14 like the Swin run (`--last-epoch 14`), so both are compared at the same learning rate.
- **Per-stage LayerNorm**, as for Swin: a channels-first LayerNorm (eps 1e-6) on each backbone output.
  - The ConvNeXt-V2 repo has no detection code. This norm matches ConvNeXt V1's official detection backbone.
- **Drop-path 0.1**, which is timm's Swin default, set explicitly for both backbones.
- **Deliberately not adopted from the official ConvNeXt recipes:**
  - layer-wise learning-rate decay;
  - their drop-path and weight-decay values;
  - excluding norms and GRN parameters from weight decay.
  - Adopting them would change more than the backbone. At learning rate 1e-5 × weight decay 1e-4, the weight-decay difference is negligible.

### Implementation check against the official ConvNeXt-V2
We compared our backbone with the official `facebookresearch/ConvNeXt-V2` model code and its `convnextv2_tiny_1k_224_ema.pt` checkpoint.

| Check | Result |
|---|---|
| Architecture | Same 18 blocks (3/3/9/3), 7×7 depthwise conv, GRN, no layer scale, LayerNorm eps 1e-6 |
| Drop-path schedule | Identical (linear 0 → 0.1 over 18 blocks) |
| Weights | 196 / 196 backbone tensors bit-identical to the official checkpoint (only the classifier head and final norm unused) |
| Logits at 224 | Max \|diff\| 2.7e-6 vs official, same top-1 |
| Stage features at 1024×800 and 576×1024 | Relative error 4e-7 to 4e-6 vs official |
| Per-stage LayerNorm | Matches the official ConvNeXt detection LayerNorm (max \|diff\| 2e-6) |
| fp16 autocast | Finite; 0.1–0.8% relative error vs fp32 |
| Preprocessing | RGB, ImageNet mean/std, as in the official code |

### Results

**subset_A val**

| | Swin-T | ConvNeXtV2-T | Δ |
|---|---|---|---|
| TSS, epoch 14 | 0.889 | 0.894 | +0.006 |
| Acc_color, mean ep 4–14 | 0.917 | 0.911 | −0.006 |
| Acc_pass, mean ep 4–14 | 0.938 | 0.932 | −0.006 |
| EM, mean ep 4–14 | 0.841 | 0.830 | −0.010 |
| **TSS, mean ep 4–14** | **0.898 ± 0.006** | **0.891 ± 0.007** | **−0.007** |
| False-Go ↓, mean ep 4–14 | 0.146 | 0.191 | +0.046 |
| False-Stop ↓, mean ep 4–14 | 0.036 | 0.029 | −0.007 |
| Epochs with higher TSS | 12 / 15 | 3 / 15 | |

**subset_B val and A+B (TSS)**

| | Swin-T | ConvNeXtV2-T | Δ |
|---|---|---|---|
| B, epoch 14 | 0.911 | 0.919 | +0.009 |
| B, mean ep 4–14 | 0.921 ± 0.013 | 0.915 ± 0.009 | −0.005 |
| A+B, epoch 14 | 0.898 | 0.905 | +0.007 |
| A+B, mean ep 4–14 | 0.908 ± 0.009 | 0.901 ± 0.007 | −0.006 |

**Observations**
- **No improvement from ConvNeXtV2.** It is ahead at epoch 14 only because epoch 14 was one of Swin's weakest (subset_A range 0.887–0.908). Averaged over epochs 4–14 it is slightly behind on A, B and A+B, and it has the higher subset_A TSS in only 3 of 15 epochs.
- **The gap is small.** It is about one standard deviation of the epoch-to-epoch swings, from one seed per backbone. The supported conclusion is "no gain, possibly slightly worse", not a clear loss.
- **ConvNeXtV2 leans towards "go".** It makes more False-Go errors (A: 0.191 vs 0.146) and fewer False-Stop errors (A: 0.029 vs 0.036). This is the more safety-relevant direction of error.
- **It still exceeds the paper's Swin-T result:** subset_A TSS 0.894 at epoch 14 vs 0.880.
- **Cost:** about 1.5× the training time per epoch (≈30 vs ≈20 min, excluding validation) and about 30% more peak GPU memory (9.5 vs 7.3 GB at batch 4, 1024×800).

## 7. Reproducing

```bash
# labels (per subset and split)
python -m dtsu.labels --root data/OpenLane-V2/subset_A --split train --data-dict data/OpenLane-V2/subset_A/data_dict_subset_A.json --out data/labels/subset_A_train.jsonl
# training (resumable)
bash tools/run_full.sh
# per-direction evaluation of runs/swin_t_AB/last.pt
python tools/eval_directions.py
# backbone ablation (§6), resumable; outputs in runs/convnextv2_t_AB/
bash tools/run_convnextv2.sh
```

Outputs in `runs/swin_t_AB/`:

| File | Contents |
|---|---|
| `last.pt` | epoch-14 checkpoint |
| `final_metrics.json` | final metrics |
| `val_log.jsonl` | per-epoch metrics |
| `per_direction.json` | per-direction metrics |
| `train.log` | training log |

The working notes and history are in `REPLICATION_PLAN.md`.
