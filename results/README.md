# Results

Outputs of `scripts/evaluate.py`; the tables in the main README are generated from them by `scripts/make_tables.py`.

| File | Model(s) | Run folder used |
|---|---|---|
| `R.json` | R, input 1280 (main model) | `runs/exp10_R`, epoch 2, threshold 0.35 |
| `M.json` | the baseline M (`scripts/evaluate_moves.py`, no detection cells) | `runs/rfdetr_m_mix`, epoch 1, threshold 0.25 |
| `ablation_resolution/1024.json` | the same recipe at 1024 | `runs/exp9_WF`, epoch 4, threshold 0.35 |
| `ablation_resolution/1536.json` | the same recipe at 1536 | `runs/exp11_S`, epoch 2, threshold 0.35 |
| `ablation_resolution/R_vs_1024.json`, `R_vs_1536.json` | R ("a") paired with the other resolution ("b") | as above |

The run folder names are those of the research log (`docs/EXPERIMENT_LOG.md`: Exp 9 WF, Exp 10 R, Exp 11 S).

Layout of each file:
- `selection`: epoch and threshold, with the tune scores and the whole selection grid;
- `maneuver_metrics`: Acc_color, Acc_pass, EM, TSS, False-Go, False-Stop per test set (`*_camera_only`: US rule
  without the lane-topology prior);
- `maneuvers`: TSS and False-Go of "a" and "b" and their difference with a segment-bootstrap 95% CI (10,000 draws);
- `attributes`: per test set, rates on the labelled lights both models detect (pictogram, color, yellow, junction,
  false arrows on round lights), each with "a", "b", "d" and a cluster-bootstrap 95% CI of "d" (2,000 draws);
- `detection`: per cell, center AP of both models with the CI of the difference, and recall, precision and color
  accuracy at each model's threshold.
In a single-model file ("a" = "b") read the "a" values; the differences are zero by construction.

Recipe ablation (`ablation_recipe/`, single seed, re-scored from the existing detection caches):

| File | "a" | "b" |
|---|---|---|
| `F8_vs_F7.json` | F8, + BDD100K train (`runs/exp8_bdd_F`, epoch 3, threshold 0.45) | F7 (`runs/exp7_multi_F`, epoch 3, threshold 0.25) |
| `W_vs_F8.json` | W, + class weights (`runs/exp9_W`, epoch 1, threshold 0.25) | F8 |
| `WF_vs_W.json` | WF, + label-swapping flip (`runs/exp9_WF`, epoch 4, threshold 0.35) | W |

Seed runs of the publication protocol (`docs/EXPERIMENT_LOG.md`, 2026-11-22) will be written to `seeds/`, one result
file per run, and `seeds/R_vs_M.json` by `scripts/compare_runs.py` once the three runs of each model exist: per cell,
`paired` (R as "a", M as "b", the run-averaged models, 10,000-draw cluster bootstrap), `a_metrics` / `b_metrics`,
`a_runs` / `b_runs` (each run's value, mean and standard deviation), and the protocol's `claims`.
