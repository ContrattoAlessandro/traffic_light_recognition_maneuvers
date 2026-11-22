# R for Autoware / AWSIM: ONNX hand-off package

This package holds the traffic-light model R as ONNX, the exact pre- and post-processing it needs, and test data to
check an integration against. It is built by `scripts/export_onnx.py` in the
[repository](https://github.com/ContrattoAlessandro/traffic_light_recognition_maneuvers). The ROS 2 node and the
Autoware wiring are not part of it (section 5).

## 1. What R does

**Input:** one front-camera RGB image. **Output:** for every traffic light it finds, a box, a pictogram, a color and
whether the light belongs to the junction the ego vehicle approaches. A fixed rule turns the junction lights into one
color per maneuver:

| Maneuver | Value |
|---|---|
| straight, left, right | `red`, `yellow`, `green` or `none` (no junction light governs it) |

Stop / go follows the color: red means stop, everything else means go. R reads relevance from the image; it does not
use the HD map to find or select lights.

## 2. Contents

| File | What it is |
|---|---|
| `AGENT_README.md` | how the model works: outputs, the junction concept, the rule, training, accuracy, weak points |
| `r_1280_opset17.onnx` | R in float32, static input `image` [1, 3, 1280, 1280], opset 17 (TensorRT 8.6 and 10.x) |
| `model_config.json` | the I/O contract: every constant below, class orders, the checkpoint's sha256 |
| `reference_infer.py` | the reference pipeline in numpy + onnxruntime (no torch): preprocessing, post-processing, rule |
| `golden/` | 7 frames (5 AWSIM, 1 DTLD, 1 OpenLane-V2) with their exact input tensor, raw outputs and expected maneuvers |
| `parity.json` | ONNX + this pipeline against the evaluated PyTorch model on AWSIM and DTLD frames |
| `SHA256SUMS.txt` | checksums of every file |

## 3. First check on the Autoware machine

```bash
sha256sum -c SHA256SUMS.txt
pip install onnxruntime numpy pillow          # onnxruntime-gpu works too
python3 reference_infer.py --check golden     # must print ALL OK
python3 reference_infer.py some_frame.png     # detections and maneuver colors as JSON
```

TensorRT engine (build it on the target machine; an engine is tied to its GPU and TensorRT version):

```bash
trtexec --onnx=r_1280_opset17.onnx --saveEngine=r_1280_fp32.engine
trtexec --onnx=r_1280_opset17.onnx --saveEngine=r_1280_fp16.engine --fp16
```

Check an engine by running it on `golden/<name>.input.npz` (key `image`), saving its five outputs to an npz with the
keys `boxes score picto color junction`, then:

```bash
python3 reference_infer.py --check-outputs mine.npz golden/<name>.outputs.npz
```

Float32 should match to about 1e-5. For fp16, check that the maneuvers of every golden frame are unchanged before
using it.

## 4. The pipeline an integration must reproduce

All constants are in `model_config.json`; `reference_infer.py` is the executable specification.

1. **Preprocess.** Convert to RGB. Squeeze the whole image to 1280 x 1280 (no crop, no letterbox, aspect ratio not
   kept) with bilinear interpolation **with antialiasing**. Scale to 0-1, subtract mean (0.485, 0.456, 0.406),
   divide by std (0.229, 0.224, 0.225), NCHW float32. Pillow's `resize(..., BILINEAR)` is the reference. Without
   Pillow, `cv2.INTER_LINEAR` is the closer substitute; `cv2.INTER_AREA` is worse, so don't use it. Agreement of
   the maneuver colors with the evaluated PyTorch model (`parity.json`):

   | Resize | AWSIM, 443 frames | DTLD test, 519 frames |
   |---|---|---|
   | Pillow bilinear | 99.3-99.5% (3 frames differ) | 99.8-100% (1 frame) |
   | `cv2.INTER_LINEAR` | 98.6-98.9% (6 frames) | 99.6-99.8% (2 frames) |
   | `cv2.INTER_AREA` | 97.5-97.7% (13 frames) | 99.0-99.2% (6 frames) |

   The remaining differences with Pillow come from float32 here against bf16 in the evaluated model. On the same
   input tensor, the ONNX graph matches float32 PyTorch within 4e-6 on every golden frame.
2. **Network.** 300 queries. `score` = light probability; `boxes` = cx, cy, w, h normalized to the image;
   `picto` = probabilities over `circle, arrow_left, arrow_right, arrow_straight, arrow_straight_left, unknown,
   other, arrow_straight_right`; `color` = probabilities over `red, yellow, green`; `junction` = probability.
3. **Post-process.** Keep `score > 0.35`. Class-agnostic NMS at IoU 0.5, highest score first. Pictogram and color
   are the argmax. A light is a junction light when `junction > 0.5`.
4. **Rule** (German convention, the one R was selected and evaluated with). Per maneuver, the highest-scoring
   junction arrow that points that way sets the color (`arrow_straight_left` counts for straight and left, and
   so on). Otherwise the highest-scoring junction round light (`circle` or `unknown`) sets it. Otherwise the
   maneuver is `none`. `other` (pedestrian, bicycle, tram signals) never sets a maneuver.

R is stateless: every frame is decoded on its own. Any temporal smoothing is part of the integration.

## 5. What this package does not do (the Autoware side)

- **The ROS 2 node.** It subscribes to the traffic-light camera
  (`/sensing/camera/traffic_light/image_raw` in the AWSIM sample setup) and runs sections 4.1-4.4. It then publishes
  `autoware_perception_msgs/msg/TrafficLightGroupArray` on `/perception/traffic_light_recognition/traffic_signals`.
- **Maneuver → `traffic_light_group_id`.** Autoware's planner looks signals up by the regulatory element of the
  lanelet on the route. The node takes the next signalized lanelet of the planned route and its `turn_direction`
  (straight / left / right) from the Lanelet2 map. It then publishes R's color for that maneuver under that group
  id. The map is used only for this lookup, never to find or select lights.
- **U-turn.** R has no U-turn output: in the training labels U-turn followed left in 99.9% of frames. Use the left
  color if a route needs one.
- **`none`.** Decide whether a maneuver with no junction light is published as `UNKNOWN` (Autoware stops) or left
  out of the message. A light that is never received may be treated as passable, so this is a safety choice that
  must be stated.
- **Native recognition.** Switch off Autoware's map-based traffic-light pipeline so only this node publishes on
  the topic.

The architecture, topics and the four test phases are described in `autoware_awsim_integration_guide.md` (written
before U-turn was dropped: use three maneuvers).

## 6. Facts to keep in mind

- Training images: DTLD (2048x1024), ATLAS (1920x1200), OpenLane-V2 (1550x2048 and 1600x900), BDD100K (1280x720),
  all squeezed to 1280. R has never seen AWSIM. The AWSIM clip in `golden/` is a 920x900 crop of the simulator
  window, not the ROS camera topic.
- Latency measured in PyTorch (bf16, batch 1, RTX 5070): 29.5 ms for the network. TensorRT on the target is not
  measured yet.
- Selection: epoch 2, threshold 0.35, chosen on the tune splits only (`results/R.json` in the repository).
