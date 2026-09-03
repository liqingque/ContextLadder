# E3: Constrained Reliability-Based Adaptive Fusion — Report

ContextLadder_复赛综合审查与提升实验计划.md, chapter 8 (lines 887-927). Exploratory experiment,
run after E1 and E2 as the plan directs. **Verdict: negative result — do not adopt.**

## What was built

- `scripts/e3_reliability_gate.py` (new, read-only w.r.t. `scripts/run_hcce.py`,
  `scripts/a2b_train_variants.py`, `scripts/grouped_ood_common.py`,
  `scripts/evaluate_grouped_ood.py`, `scripts/e1_neutral_unknown.py`).
- Stage `oof`: reused E2's 32-fold four-axis manifest
  (`outputs/grouped_ood/split_manifest.json`) to generate cross-fitted OOF
  `film`/`legacy` predictions (one training run per fold gives both experts).
  23,680 OOF rows (5,920 train rows × 4 axes) written to
  `outputs/e3_reliability_fusion/oof_gate_training_table.csv`, with raw
  per-fold prediction arrays under `outputs/e3_reliability_fusion/oof/`
  (`<axis>_fold<k>_{film,legacy}.npy`, ~804 MB total). Took 763s (~13 min).
- Stage `gate`: fit a tiny MLP gate (11 reliability features → sigmoid → 
  `[0.35, 0.65]`) directly on the exact fused-blend MSE loss
  `L = E[(g·film+(1-g)·legacy-true)^2] + λ_g·E[(g-0.5)^2]`, computed in closed
  form per row (`A + 2Bg + Cg²`) so no label ever needs to leave the OOF
  stage. λ_g selected by **leave-one-axis-out inner CV strictly inside the
  OOF table** (never touches validation) — grid `[0, 0.01, 0.03, 0.1, 0.3, 1, 3]`.
  Frozen gate: `outputs/e3_reliability_fusion/gate_model.pt`.
- Stage `confirm`: retrained both experts on the full 5,920-row train split at
  3 pre-committed seeds (20260810 / 3407 / 42, matching `configs/final.yaml`'s
  ensemble seeds), applied the **frozen** gate once, and compared against the
  fixed 50/50 blend on the official validation split, broken out by the five
  required subsets. Single confirmation pass, no re-tuning.
  `outputs/e3_reliability_fusion/validation_confirmation.json`.

Reliability features used (no labels): `compound_seen`, `strain_seen`,
`time_seen`, `plate_seen`, `source_seen`, `instrument_seen`, `time_distance`,
`expert_disagreement` (from `scripts/run_hcce.py::feature_rows`, imported
read-only) plus `compound_log1p_count`, `strain_log1p_count`,
`plate_log1p_count` (new, this file's `freq_log1p_features`).

## 1. Does the gate learn anything on OOF (train-internal)?

Leave-one-axis-out inner CV, mean per-row loss (lower is better; the analytic
optimum of `L_row(g)` at fixed g=0.5 is the "fixed 50/50" baseline):

| λ_g | CV loss (gate) | CV loss (fixed 0.5) | improvement |
|---|---|---|---|
| 0.0  | 0.214392 | 0.215151 | **+0.000760** |
| 0.01 | 0.214352 | 0.215151 | **+0.000800** (selected) |
| 0.03 | 0.216303 | 0.215151 | -0.001152 |
| 0.1  | 0.216130 | 0.215151 | -0.000979 |
| 0.3  | 0.216210 | 0.215151 | -0.001059 |
| 1.0  | 0.215327 | 0.215151 | -0.000176 |
| 3.0  | 0.215785 | 0.215151 | -0.000634 |

Only the two weakest-regularization settings beat the fixed blend at all, and
by a **0.37% relative** margin. Every stronger regularization setting (which
is exactly what the spec's risk section (8.2) calls for to avoid over-routing)
makes the gate *worse* than fixed 50/50 in cross-validation. This is already
a weak signal before validation is even touched.

## 2. Gate value distribution — is it flat, or clipped?

**OOF (`gate_value_distribution_oof.json`, n=23,680):**
mean 0.527, std 0.113, only **2.4%** of rows within 0.01 of 0.5 (not flat),
but **25.0%** pinned at the 0.35 floor and **11.5%** pinned at the 0.65
ceiling — over a third of all rows are clip-saturated.

**Validation (`gate_value_distribution_validation.json`, n=3,038):**
mean 0.426, and **59.8%** of all validation rows sit at the 0.35 floor
(`< lo+0.001`). Broken out by subset:

| subset | g mean | g std |
|---|---|---|
| strain_unseen (n=1547) | 0.350000 | 1.1e-6 |
| both_unseen (n=269) | 0.350003 | 3.2e-6 |
| compound_unseen (n=1065) | 0.5387 | 0.0223 |
| both_seen_time_shift (n=157) | 0.5429 | 0.0915 |

**Every single row with an unseen strain gets g pinned at exactly the floor**
(std ~1e-6, i.e. effectively a constant, not a per-row estimate). Since the
official validation split has exactly one unseen strain (the "S2, n=1"
scenario the spec explicitly warns about), the gate has not learned a
continuous reliability function for strain at all — it learned a step
function on the single binary feature `strain_seen`, and the `[0.35, 0.65]`
clip is doing 100% of the work rather than shaping a nuanced score. This is
precisely the failure mode the task asked me to check for ("是不是...全贴在
边界（说明被 clip 救了）") — and for the strain axis, yes, exactly that.

Compound-unseen and time-shift rows show genuine per-row spread (std 0.02-0.09),
so the gate is not a pure constant everywhere — but the axis that matters most
for this competition's OOD story (strain generalization) collapsed to a
constant.

## 3. Seen vs. unseen entities — is there an interpretable difference?

OOF means: `strain_seen`=0.586 vs `not strain_seen`=0.350 (floor);
`plate_seen`=0.456 vs `not plate_seen`=0.582; `time_seen`=0.501 vs
`not time_seen`=0.605. So plate/time unseen → gate leans *toward* HCCE/FiLM
(higher g), consistent with `HCCEModel`'s built-in "shrunk residual" fallback
for unseen plates giving FiLM a real generalization advantage there — a
plausible, interpretable mechanism. But strain unseen → gate leans *away*
from HCCE (floor, 0.35), the opposite direction, and it is fully saturated.
The two axes disagree in sign, which is at least *consistent* with a real
architectural asymmetry (plate has an explicit hierarchical fallback in the
model; strain does not) rather than pure noise — but the strain result in
particular is built from only 4 train-internal strain entities and confirmed
by only 1 unseen validation strain, so this is not a statistically resolvable
claim either way; see §7.4 of the E2 report's own low-power-axis caveat.

## 4. One-time validation confirmation — deltas vs. fixed 50/50

3 seeds (20260810, 3407, 42), full-train retrain, frozen gate, single pass:

| subset | n | mean Δ abs_pcc | Δ pcc sign consistent across 3 seeds? | mean Δ rmse |
|---|---|---|---|---|
| all | 3038 | +0.0000065 | **no** | -0.00057 |
| strain_unseen | 1547 | +0.0000224 | **no** | -0.00113 |
| compound_unseen | 1065 | +0.0000020 | yes (positive, ~0) | -0.0000047 |
| both_unseen | 269 | -0.0000556 | **no** | +0.00064 |
| both_seen_time_shift | 157 | -0.0000133 | yes (negative) | +0.00051 |

Per-seed detail (abs_pcc, fixed vs. gate):

- seed 20260810: all −0.00008, strain_unseen −0.00013, compound_unseen
  +0.00000, both_unseen −0.00021, time_shift −0.00001
- seed 3407: all +0.00005, strain_unseen +0.00010, compound_unseen +0.00000,
  both_unseen +0.00001, time_shift −0.00000
- seed 42: all +0.00005, strain_unseen +0.00009, compound_unseen +0.00000,
  both_unseen +0.00004, time_shift −0.00002

## 5. Does the improvement exceed noise?

**No.** For 3 of the 5 subsets — `all`, `strain_unseen`, and `both_unseen`
(the two axes that matter most for OOD) — the sign of the PCC delta flips
across the 3 pre-committed seeds. A metric whose sign is not stable across
seeds cannot be claimed as a real effect; it is inside the noise floor. The
two subsets with a consistent sign (`compound_unseen`: +2.0e-6,
`both_seen_time_shift`: −1.3e-5) are consistent about being **essentially
zero or very slightly negative** — not evidence of a real gain either.
Compare to the metric's own scale: abs_pcc sits around 0.984-0.993 across
subsets, so these deltas are 3-4 orders of magnitude smaller than the metric
itself, and RMSE deltas (±0.0005-0.002) are similarly negligible next to the
model's baseline RMSE.

## 6. Judgment: should this be adopted?

**No.** Three independent pieces of evidence all point the same way:

1. Cross-validated (train-internal, leakage-safe) improvement over fixed
   50/50 is tiny (0.37% relative) and disappears entirely once any
   meaningful regularization toward 0.5 is applied — the spec's own
   anti-overfitting safeguard (§8.2/8.3 point 4) is exactly what erases the
   gate's advantage.
2. The gate did not learn a continuous reliability function for the
   axis that matters most (strain): it collapsed to a constant at the clip
   boundary, driven by a single low-cardinality binary feature. The `[0.35,
   0.65]` clip is not lightly bounding an otherwise-smooth estimator here —
   it is doing all the work for 60% of validation rows.
3. The one-time validation confirmation shows deltas that are both
   minuscule (parts in 10^-5 to 10^-4 of abs_pcc) and sign-inconsistent
   across the three pre-committed seeds for the subsets that matter, i.e.
   indistinguishable from seed noise.

Fixed 50/50 fusion is already at least as good as the constrained reliability
gate, and simpler, more robust, and free of the tail risk of a badly-clipped
degenerate gate on a differently-distributed test population. **This result
does not replace, and was never used to replace, any frozen artifact under
`runs/final/`, `configs/final.yaml`, or `prediction.csv`.**

## Artifacts

- `scripts/e3_reliability_gate.py` — the new (only) script for this experiment.
- `outputs/e3_reliability_fusion/oof_gate_training_table.csv` — 23,680-row
  OOF feature/A-B-C table (compact; used to train and CV the gate).
- `outputs/e3_reliability_fusion/oof/` — 96 files, raw per-fold `film`/`legacy`
  OOF prediction arrays + sample-id lists (~804 MB).
- `outputs/e3_reliability_fusion/oof_manifest_used.json` — OOF-stage config record.
- `outputs/e3_reliability_fusion/lambda_selection.json` — leave-one-axis-out
  CV grid and selection.
- `outputs/e3_reliability_fusion/gate_model.pt` — frozen gate (state dict +
  feature normalization stats + λ_g).
- `outputs/e3_reliability_fusion/gate_value_distribution_oof.json` /
  `gate_value_distribution_validation.json` — gate-value distributions.
- `outputs/e3_reliability_fusion/validation_confirmation.json` — the one-time,
  3-seed, 5-subset confirmation run.
