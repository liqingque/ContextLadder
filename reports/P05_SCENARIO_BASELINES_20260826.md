# P0.5 Scenario Baselines — 2026-08-26

## Frozen protocol

The gate was written before any train/validation proteome or metadata was
opened. `configs/p05_gates.json` SHA-256 is
`c02e413cd766fb1303ab6544f81796b59691bc0ef341cd95a8d638602fc215a7`.
It freezes the official Gate-W weights, seven held-out-entity clusters, paired
2,000-draw cluster bootstrap, ΔW ≥ 0.010, no significant negative module,
mandatory ΔW_noS2, and 200 compound permutations. The candidate map was then
frozen in `configs/p05_candidates.json` before official evaluation.

No test metadata, test proteome, or test labels were read. The only target
matrix materialized by `scripts/p05_scenario_baselines.py` is the 5,920-row
`split_final=train` matrix. The protein contract is obtained through
`run_hcce.apply_official_protein_filter`; 4,422 proteins are retained.

## Reproduction commands

```bash
source /home/lxm/anaconda3/etc/profile.d/conda.sh && conda activate tl
python scripts/p05_scenario_baselines.py --only b1 --max-iter 25
python scripts/p05_scenario_baselines.py --only b2
python scripts/p05_scenario_baselines.py --only b3 --max-iter 20
python scripts/evaluate_official_modules.py \
  --candidates configs/p05_candidates.json \
  --baseline mask_compound_3seed_base \
  --out outputs/p05_scenario_baselines --workers 4 --bootstrap 2000
python scripts/p05_compound_nulls.py
```

B1 is global train-only PCA rank 64/128 plus HistGradientBoosting latent
heads, each as an equal 3-seed ensemble. B2 is a train-only scenario-routed
low-rank statistic. B3 is four scenario-routed GBDT heads, equal-averaged over
seeds 20260810, 3407, and 42. Category maps and numeric moments are fit on
train only and unknown categories map to token 0. Individual B3 seed files are
also retained for audit.

## Official validation results

The official evaluator found 2,806 matched validation rows with scenario
counts seen=139, S1=1,065, S2=1,333, S3=269. Conclusions use
`official_plate`; `bio` numbers are not used.

| candidate | abs PCC | RMSE | FC all | S1 official_plate | S2 residual | S3 FC | DEP direction |
|---|---:|---:|---:|---:|---:|---:|---:|
| mask_compound_3seed_base | 0.988598 | 0.420259 | 0.491355 | 0.203387 | 0.407705 | 0.344605 | 0.891936 |
| B1 GBDT PCA64 | 0.976734 | 0.606582 | 0.324686 | 0.081769 | 0.294807 | 0.258669 | 0.808653 |
| B1 GBDT PCA128 | 0.976863 | 0.604997 | 0.326265 | 0.081493 | 0.295145 | 0.258311 | 0.809963 |
| B2 scenario stats | 0.981071 | 0.549638 | 0.348879 | 0.073488 | 0.370610 | 0.335239 | 0.826608 |
| B3 scenario GBDT | 0.981191 | 0.559951 | 0.352098 | 0.076750 | 0.353607 | 0.312719 | 0.816213 |

All four candidates have a negative S1 Δ versus the frozen baseline. The
2,000-draw seven-entity paired cluster-bootstrap results at official_plate
are:

| candidate | ΔS1 | 95% CI | Gate-W consequence |
|---|---:|---:|---|
| B1 PCA64 | -0.121618 | [-0.162040, -0.087022] | significant negative module |
| B1 PCA128 | -0.121894 | [-0.162437, -0.087369] | significant negative module |
| B2 stats | -0.129898 | [-0.158646, -0.089646] | significant negative module |
| B3 scenario GBDT | -0.126637 | [-0.167666, -0.093140] | significant negative module |

Therefore Gate-W fails for every candidate, independently of the weighted
score threshold: the “no significant negative module” condition is violated
by S1. ΔW_noS2 consequently cannot rescue any candidate and none is eligible
for adoption. Classification: **DOMINATED** — every weighted module is worse
and the S1 deterioration is significant on every candidate. The frozen plan
(`refine-logs/EXECUTION_PLAN_P05_P2_20260826.md`) defines only ADOPT,
INFORMATIVE NEGATIVE ("failed Gate-W but a CI-excludes-zero *improvement* on
some module"), and FLAT ("all module Δ CIs contain zero"); a uniform
significant *deterioration* falls under none of the three, so the label used
here is outside the plan's taxonomy — that is a gap in the plan, not a
deviation by the executor. The current `mask_compound_3seed_base` line remains
unchanged.

The evaluator's `module_metrics.json` does not emit the separate time-FC
module, so no invented time-FC number is used here; this does not affect the
negative Gate-W decision because the S1 condition already fails. S2 is also
reported as single-cluster/undecidable per the frozen protocol.

### Gate-W weighted score

`scripts/gate_w_summary.py` assembles the official weighted score
W = 0.20·absPCC + 0.25·FC_all + 0.20·S1 + 0.20·S2 + 0.10·T + 0.05·DEPacc
from `module_metrics.json`
(`outputs/p05_scenario_baselines/gate_w_summary.json`,
`outputs/p05_scenario_baselines/gate_w_table.md`):

| candidate | abs PCC | RMSE | FC all | S1 | S2 | S3 FC | DEP | W | ΔW | ΔW_noS2 | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| mask_compound_3seed_base | 0.988598 | 0.420259 | 0.491355 | 0.203387 | 0.407705 | 0.344605 | 0.891936 | 0.5218 | — | — | BASELINE |
| B1_gbdt_global_pca64 | 0.976734 | 0.606582 | 0.324686 | 0.081769 | 0.294807 | 0.258669 | 0.808653 | 0.4181 | -0.1037 | -0.1014 | DOMINATED |
| B1_gbdt_global_pca128 | 0.976863 | 0.604997 | 0.326265 | 0.081493 | 0.295145 | 0.258311 | 0.809963 | 0.4186 | -0.1032 | -0.1009 | DOMINATED |
| B2_scenario_stats | 0.981071 | 0.549638 | 0.348879 | 0.073488 | 0.370610 | 0.335239 | 0.826608 | 0.4471 | -0.0747 | -0.0841 | DOMINATED |
| B3_scenario_gbdt | 0.981191 | 0.559951 | 0.352098 | 0.076750 | 0.353607 | 0.312719 | 0.816213 | 0.4424 | -0.0794 | -0.0857 | DOMINATED |

Reading of this table:

- **The adoption floor is ΔW ≥ +0.010.** It is not a new number: it is the
  frozen S1 effect floor 0.053 converted through that module's 0.20 official
  weight (0.053 × 0.20 = 0.0106, floored to 0.010 in
  `configs/p05_gates.json`). The 0.053 floor is a detection threshold set by
  S1 having only six clusters and a cluster-bootstrap SD of 0.0265, and it
  does not move.
- **Observed ΔW is several times the threshold in the wrong direction.**
  |ΔW| ranges from 0.0747 (B2) to 0.1037 (B1 PCA64), i.e. roughly 7.5× to
  10.4× the +0.010 adoption floor, with the sign reversed. All six weighted
  modules — absPCC, FC_all, S1, S2, S3 FC, DEPacc — are worse than the
  baseline for all four candidates, with no exception; RMSE is also higher
  (worse) for all four. ΔW_noS2 is likewise negative throughout, so dropping
  the single-cluster S2 term rescues nothing.
- **The 10% term uses S3 FC alone.** The official evaluator emits no separate
  `val_time` module — `val_time` rows have both entities present in train and
  are stratified as `seen` — so the 10% slot is filled with the S3 FC value
  and this substitution is recorded in `gate_w_summary.json` rather than
  hidden. No time-FC number is invented here or anywhere in this report.
- **No confidence interval is attached to ΔW.** The gate decides on the
  per-module paired cluster bootstrap already tabulated above; a CI for the
  weighted sum is not derivable from module point estimates and is not
  manufactured.

### Scope limits of this comparison

1. **PCA bottleneck confound.** B1 and B3 both place a rank-64/128 train-only
   PCA between the features and the protein output. P6
   (`reports/P6_PROTEIN_DECODER_RESULTS.md`) already measured that a fixed
   rank-128 decoder underperforms the direct output head (Δabs-PCC =
   −0.000611, ΔFC-PCC = −0.009372). So their deficit here confounds "GBDT vs
   neural" with "PCA bottleneck vs direct head", even though P6's measured
   penalty is far smaller than the gap observed in this table. B2 uses no GBDT
   at all and also loses, so the loss cannot be attributed to gradient
   boosting either. The defensible claim is that **these three specific
   constructions are dominated**, not that gradient boosting is dominated.
2. **Effort asymmetry.** These baselines were built in a single day with
   `HistGradientBoostingRegressor` at `--max-iter 25` (B1) and `--max-iter 20`
   (B3), against a main line that has had weeks of tuning. The comparison is
   not effort-matched and should not be read as a method-class verdict.
3. **B3's S1 route is self-fulfilling.** B3 zeroes the compound feature column
   on the `S1_chem_only` route (`scripts/p05_scenario_baselines.py` line 262),
   so its S1 = 0.076750 is partly a construction artifact of that routing
   choice. It is not evidence about scenario-routed methods in general.

## Compound invariance (identity check, not a statistical test)

An earlier draft of this report presented the 200-draw compound-permutation
null as a test and read its p = 1.0 as evidence that the classical baselines
carry no compound information. That reading was wrong and is retracted here.

**The permutation null has zero variance, so the test has no power.**
`outputs/p05_scenario_baselines/compound_nulls_200.json` records `null_sd`
= 0.0 for `mask_compound_3seed_base`, `B1_gbdt_global_pca64`, and
`B1_gbdt_global_pca128`, and 1.3877787807814457e-17 / 2.7755575615628914e-17
for `B2_scenario_stats` / `B3_scenario_gbdt` — i.e. floating-point zero. A
permutation test whose null distribution is a point mass cannot reject
anything; p = 1.0 is an arithmetic identity, not a result.

**The identity is structural, not empirical.** In
`scripts/p05_scenario_baselines.py` the train-only category maps are built in
`TrainOnlyMeta.fit` as

```python
self.maps[key] = {v: i + 1 for i, v in enumerate(sorted(set(vals)))}   # line 62
```

so index 0 is reserved as the shared unknown token and every compound not seen
in train maps to it. B3 goes further: on the `S1_chem_only` and `S3_both`
routes it zeroes the compound feature column outright,

```python
if route in ("S1_chem_only", "S3_both"): xt[:, 1] = 0.0; xval[:, 1] = 0.0   # line 262
```

so on S1 the compound feature is removed by construction. Permuting compound
labels among rows that all already carry the same token changes no input, so
it changes no prediction.

**This table therefore adds no independent evidence for the zero-compound-
information thesis.** It is a consistency check on the implementation, and it
is reported as such.

The direct measurement of that identity is
`outputs/p05_scenario_baselines/compound_invariance.json`
(`scripts/compound_invariance_check.py`, granularity `official_plate`). Inside
each context block it takes every pair of validation rows carrying *different*
unseen compounds and reports the maximum absolute difference between their
predicted 4,422-protein vectors.

S1_chem_only:

| candidate | rows | contexts with >1 compound | cross-compound pairs | pairs bitwise identical | max\|Δ\| |
|---|---:|---:|---:|---:|---:|
| mask_compound_3seed_base | 1065 | 379 | 991 | 991 | 0.000e+00 |
| B1_gbdt_global_pca64 | 1065 | 379 | 991 | 991 | 0.000e+00 |
| B1_gbdt_global_pca128 | 1065 | 379 | 991 | 991 | 0.000e+00 |
| B2_scenario_stats | 1065 | 379 | 991 | 991 | 0.000e+00 |
| B3_scenario_gbdt | 1065 | 379 | 991 | 991 | 0.000e+00 |

S3_both:

| candidate | rows | contexts with >1 compound | cross-compound pairs | pairs bitwise identical | max\|Δ\| |
|---|---:|---:|---:|---:|---:|
| mask_compound_3seed_base | 269 | 96 | 250 | 250 | 0.000e+00 |
| B1_gbdt_global_pca64 | 269 | 96 | 250 | 250 | 0.000e+00 |
| B1_gbdt_global_pca128 | 269 | 96 | 250 | 250 | 0.000e+00 |
| B2_scenario_stats | 269 | 96 | 250 | 250 | 0.000e+00 |
| B3_scenario_gbdt | 269 | 96 | 250 | 250 | 0.000e+00 |

Every cross-compound pair is bitwise identical on both scenarios and every
candidate, including the frozen baseline. The models emit one vector per
context. That is the whole content of the permutation result.

**Where the actual empirical evidence lives.** The claim that no constructible
compound representation currently beats noise on this task is supported by the
representation-channel experiments, not by this table. DCB-40
(`reports/DCB40_EXECUTION_RESULTS_20260816.md`) measured every constructible
compound-representation channel under a fixed decoder and none beat the
negative control: φ_TXT (main channel) ρ̄ = 0.0268, CI [−0.0117, +0.0664];
φ_SSPS (C2 ablation) ρ̄ = −0.0325, CI [−0.0727, +0.0126]; Tanimoto
(construction control) −0.0148, CI [−0.0589, +0.0392]; random (negative
control) −0.0250, CI [−0.0643, +0.0183]. The maxT 95% critical value is
0.0638 against an observed |mean| of 0.0346, so φ_TXT is not significant after
multiplicity correction either. The sparse-support sign-optimal ceiling is
0.0491 and the SSPS prior's own Gate-0 score is ρ̄ = 0.0059 with a CI
containing zero — both from `reports/M0_MODULE_EVAL_RESULTS_20260816.md`, not
from DCB-40. The oracle full-span capacity at k = 36 is 0.2671, so the ceiling
is a content limit, not a capacity limit.

The claim that survives is therefore narrower and better sourced: routing
unseen compounds to a shared token is not a modelling shortcut, it is
currently the only choice with evidence behind it.

`compound_nulls_200.json` is retained for traceability of the pre-registered
protocol, but it is superseded by the invariance check above and must not be
cited as a statistical test.

## Artifacts and audit

- Code: `scripts/p05_scenario_baselines.py`, `scripts/p05_compound_nulls.py`,
  `scripts/compound_invariance_check.py`, `scripts/gate_w_summary.py`
- Frozen configs: `configs/p05_gates.json`, `configs/p05_candidates.json`
- Evaluation: `outputs/p05_scenario_baselines/module_metrics.json`
- Gate-W: `outputs/p05_scenario_baselines/gate_w_summary.json`,
  `outputs/p05_scenario_baselines/gate_w_table.md`
- Invariance: `outputs/p05_scenario_baselines/compound_invariance.json`
- Nulls (superseded, retained for traceability):
  `outputs/p05_scenario_baselines/compound_nulls_200.json`
- Predictions: B1 PCA64/PCA128, B2, B3 ensembles; B3 seed-level parquet files
- Leakage proofs: `outputs/p05_scenario_baselines/leak_check_*.json`
- Logs: `outputs/p05_b1.log`, `outputs/p05_b2_smoke.log`, `outputs/p05_b3_rerun.log`, `outputs/p05_official_eval.log`, `outputs/p05_compound_nulls.log`, `outputs/p05_invariance.log`, `outputs/p05_gate_w.log`

All prediction contract checks passed: 3,038 unique validation `sample_ID`
rows, 4,422 finite protein columns, no non-finite values, and no test truth.
