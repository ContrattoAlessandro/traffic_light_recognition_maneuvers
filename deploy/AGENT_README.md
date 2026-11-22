# R, the traffic-light model: how it works (read this first)

This file is for the engineer or coding agent who integrates R into Autoware and AWSIM on the Linux machine. It
explains what the model computes, how it was trained, what its outputs mean, how they become a stop / go decision
per maneuver, and where it fails. Read it before writing any code. Then read:

1. `README.md`: the package, the first checks, the exact pipeline to reproduce, and what the ROS 2 side must do;
2. `model_config.json`: every constant (input size, mean / std, class orders, thresholds);
3. `reference_infer.py`: the executable specification of pre-processing, post-processing and the rule.

Source repository: <https://github.com/ContrattoAlessandro/traffic_light_recognition_maneuvers> (code in `dtsu/`,
research log in `docs/EXPERIMENT_LOG.md`).

## 1. The task

A vehicle approaching a signalized junction does not need the color of every light in view. It needs to know
whether the maneuver it is about to make (**straight, left or right**) is allowed. R answers this from **one front
camera image**, with no HD map. For each maneuver it returns `red`, `yellow`, `green` or `none` (no light of this
junction governs it). Stop / go follows the color: only red means stop.

The camera, not the map, decides which lights matter. That is the point of the work. In the Autoware integration,
the map is used for one thing only: translating "the planned maneuver at the next junction" into the
`traffic_light_group_id` and `turn_direction` that Autoware's planner expects (`README.md`, section 5).

## 2. What the network is

RF-DETR Medium, a real-time DETR detector (34 M parameters):
- **Backbone:** DINOv2-S vision transformer with windowed attention, followed by a multi-scale projector.
- **Decoder:** a two-stage deformable-attention decoder with 4 layers and **300 object queries**. Each query either
  finds one traffic light or reports background. DETR-style detectors need no anchors.
- **Head (the part specific to this work, "factorized"):** the usual class layer is replaced by one
  "light vs background" logit, plus three small linear heads on the same query features:

| Output (per query) | Meaning | Values |
|---|---|---|
| `score` | probability that the query is a traffic light | sigmoid |
| `boxes` | box center x, center y, width, height, normalized to the input image (0-1) | 4 floats |
| `picto` | what the lit lamp shows | softmax over `circle, arrow_left, arrow_right, arrow_straight, arrow_straight_left, unknown, other, arrow_straight_right` |
| `color` | the color of the lit lamp | softmax over `red, yellow, green` |
| `junction` | probability that the light belongs to the junction the ego vehicle is approaching | sigmoid |

`unknown` = a vehicle lamp whose pictogram cannot be read (treated as a circle). `other` = a pedestrian, bicycle or
tram signal (never used for driving). There is no "off" class: a dark lamp should simply not be detected.

The ONNX graph returns these five tensors for all 300 queries, without thresholding or NMS.

## 3. What "junction" means

Junction does **not** mean "the light of my lane". It means "a vehicle signal of the junction I am approaching, facing
me", including lights for neighbouring lanes. It was learned from two sources:
- **DTLD (Germany):** the lamps DTLD marks relevant to the driven route, plus the other front-facing vehicle lamps
  at least 0.6 x the median height of those route lamps (lamps of the same junction, seen from the same distance).
- **OpenLane-V2 (US, Singapore):** lights that the HD-map topology links to a lane of the ego approach at the
  nearest junction.

Lights of the next junction further down the road, side streets and pedestrian signals should get a low junction
probability.

## 4. From lights to maneuvers (the rule)

After post-processing (score > 0.35, class-agnostic NMS at IoU 0.5), the junction lights (junction > 0.5) of the
frame go through a fixed rule, the **junction rule**:

- For each maneuver, the **highest-scoring arrow pointing that way** sets its color. Arrows map as:
  `arrow_left` → left, `arrow_right` → right, `arrow_straight` → straight, `arrow_straight_left` → straight and left,
  `arrow_straight_right` → straight and right.
- A maneuver with no such arrow takes the color of the **highest-scoring round light** (`circle` or `unknown`).
- With neither, the maneuver is `none`.

Examples:

| Junction lights seen | straight | left | right |
|---|---|---|---|
| red circle | red | red | red |
| red circle + green left arrow | red | green | red |
| green circle + red left arrow | green | red | green |
| red circle + green straight arrow + green right arrow (a common Japanese phase) | green | red | green |
| nothing above the thresholds | none | none | none |

The rule is the German convention: the circle governs every maneuver without its own arrow, the right turn included.
It also fits Japanese signals, where green arrows give extra permission under a red circle. The repository also
has a US rule (right is `none` unless a right arrow exists; left falls back on straight), used only to score
against US benchmark labels. Use the junction rule in AWSIM.

R is **stateless**: every frame is decoded on its own. There is no tracking, no smoothing and no flashing
detection.

## 5. How it was trained

| Source | Images | Labels used | Why |
|---|---|---|---|
| DTLD v2.0 (Germany) | 25,635 | pictogram, junction, color | the only source with all three |
| ATLAS (Germany) | 26,528 | pictogram, color | more arrows, three camera focal lengths |
| OpenLane-V2 A + B (US, Singapore) | 14,977 | junction, color | US signal heads, junction from the HD-map topology |
| BDD100K (US) | 18,851 | color | night scenes and horizontal lights |

- **Partial labels:** each attribute loss is switched off where a dataset does not label that attribute. This is
  why one model can learn from four differently labelled datasets.
- **Input:** every image is squeezed to 1280 x 1280; aspect ratio is not kept, the same in training and inference.
- **Training recipe:** 5 epochs, pictogram and color losses weighted against class imbalance, a horizontal flip
  that swaps left / right arrow labels, brightness / contrast jitter, no random crop.
- **Selection:** epoch 2 and the 0.35 threshold were chosen on held-out tune splits, never on a test set.
- **Never trained on:** simulation images, Japanese signals.

## 6. How well it works (single training run, test sets never used for selection)

Per-maneuver decisions. TSS is the mean of color accuracy, stop / go accuracy and whole-frame exact match.
False-Go is the share of red maneuvers predicted as go.

| Test set | TSS | Color acc. | False-Go | False-Stop |
|---|---|---|---|---|
| DTLD test (Germany, 11,494 frames) | 0.955 | 0.962 | 0.031 | 0.017 |
| OpenLane-V2 A val (US), camera only | 0.927 | 0.937 | 0.090 | 0.030 |
| OpenLane-V2 B val (US + Singapore), camera only | 0.913 | 0.930 | 0.105 | 0.014 |

Per detected light, DTLD test:
- color accuracy 0.979, yellow recall 0.89, junction accuracy 0.96;
- pictogram recall: circle 0.96, left arrow 0.86, straight arrow 0.67, right arrow 0.63.

Pictogram reading by lamp height:

| Lamp height | Pictogram macro recall |
|---|---|
| under 15 px | 0.54 |
| 15-20 px | 0.62 |
| 20-30 px | 0.69 |
| 30 px and over | 0.87 |

Detection on datasets never seen in training (center AP):

| Dataset | Center AP |
|---|---|
| LISA (US), day | 0.84 |
| LISA (US), night | 0.77 |
| S2TLD (China), 720p | 0.89 |
| VZC-TLD (US) | 0.87 |
| Horizontal lights: S2TLD 1080p | 0.37 |
| Horizontal lights: VZC-TLD | 0.55 |
| Horizontal lights: BDD100K | 0.33 |

**Latency:** 29.5 ms for the network on an RTX 5070 (PyTorch, bf16, batch 1). TensorRT on the target is not
measured yet.

## 7. Weak points that matter in AWSIM

- **Horizontal lights.** AWSIM's default map (Nishi-Shinjuku, Tokyo) uses Japanese signals, which are mostly
  horizontal. R saw few horizontal lights in training, and its detection AP on them is 0.33-0.55 (section 6).
  Expect missed lights.
- **Small, distant lamps.** Pictograms under 15 px are read correctly about half the time. On the AWSIM clip used for
  `golden/` (a 920 x 900 crop of the simulator window), the median junction lamp was 16.7 px tall, and 40% were under
  15 px. A higher-resolution camera image helps.
- **Simulation domain.** R has never seen rendered images. The AWSIM frames in `golden/` are decoded sensibly, but
  no AWSIM accuracy has been measured.
- **Lane-specific lights.** The rule takes the highest-scoring round light of the whole junction. Where different
  lanes have different circle colors, it can take the wrong one. In the US data this "another light decided" error
  is a large share of False-Go.
- **Junction scope.** The light of the next junction can occasionally be marked as part of the junction.
- **US left-turn arrows** at night or far away are often read as circles.
- **No temporal logic.** Single-frame errors flicker. A yellow phase seen for one frame is indistinguishable from a
  misread.

## 8. What the integration should add

All of this is on the Autoware side; `README.md` section 5 lists the ROS 2 interfaces.
- **Temporal filtering:** for example a short majority vote, or hysteresis that switches to red quickly and back to
  green only after several consistent frames.
- **A stated policy for `none`:** publish UNKNOWN (Autoware stops) or omit the group (it may be treated as
  passable). This is a safety decision and must be written down.
- **Logging** of R's per-frame maneuver colors next to Autoware's ground-truth signal state, for the KPI comparison
  of baseline Autoware, R, and an oracle.

## 9. How to check an integration

- `python3 reference_infer.py --check golden` must print `ALL OK`.
- A TensorRT engine is checked with `--check-outputs` against `golden/<name>.outputs.npz` (`README.md`, section 3).
- Image resizing changes results: Pillow bilinear (the reference) agrees with the evaluated model on 99.3-100% of
  maneuver colors. Without Pillow, use `cv2.INTER_LINEAR` (98.6-99.8%); `cv2.INTER_AREA` agrees less (97.5-99.2%).
  The numbers are in `parity.json`.
