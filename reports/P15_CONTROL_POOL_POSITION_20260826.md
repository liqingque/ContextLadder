# P1.5 Control-pool position and R1 zero-information control

**Date:** 2026-08-26  
**Scope:** R15-001/R15-002 only. This is a diagnostic experiment, not a submission-line change.

## Decision summary

The official `train|val` control pool covers all 2,806 matched validation treatments and gives the frozen submission line FC-PCC **0.491354711**. The legal train-only pool matches 1,198/2,806 treatments, coverage **0.4269422666**, and gives FC-PCC **0.595047947** on that selected subset. The two numbers are not directly comparable without their coverage labels; the train-only number must not be presented as the official validation FC score.

For train-only method development, the train-only pool remains the primary leakage-safe sensitivity: every control anchor is available before validation labels are considered, so it answers what can be computed from the fit split alone. The official `train|val` pool is reported in parallel—and is the primary full-coverage/public-caliber score—because the competition metric supplies validation-side controls. Keeping both labels visible prevents the fit-only diagnostic (coverage 42.7%) from being mistaken for the full official metric.

Using `official_plate` for S1, the frozen line is **0.203386812** (1,059 S1 rows with a train-estimated context mean). R1 convex mixing increases this diagnostic S1 score to 0.215531727, 0.225294428, and 0.232308063 for `w=0.05, 0.10, 0.15`, respectively. These changes are not evidence of compound information: the 200-draw same-context compound permutation null is equal to the true score to floating-point precision (absolute gaps ≤ 8.33e-17 for all official-pool candidates). Therefore **R1 is a reject/diagnostic positive control and is NOT ADOPTED**.

## Protocol and data boundary

- Evaluation data read: `configs/data_paths.yaml` entries `input/WAYB_WAYC_metadata_train_val(1).csv` and `input/WAYB_WAYC_proteome_raw_train_val.csv` only.
- Prediction source: `outputs/r1_convex_mix_3seed/prediction_val_mean50_3seed.parquet` (the frozen current submission line).
- No separate held-out data path was opened; every result JSON carries `"no_test_truth": true`.
- Protein filter: reused `run_hcce.apply_official_protein_filter`, yielding 4,422 proteins under the frozen train-only filter.
- FC matching key: the shared `match_controls` implementation (source/data source, strain, medium, temperature, time, time unit, instrument, plate), with DMSO preferred over Water.
- S1 interpretation: `official_plate` context key (strain, medium, temperature, time, time unit, plate), with context means estimated from train matched rows.
- R1 construction: for each validation treatment matched to a train control, `y_mix=(1-w)y_model+w y_control_train`; unmatched rows are unchanged. Weights were exactly `{0.05, 0.10, 0.15}`.
- Permutation null: 200 deterministic draws, seed `20260816`; within each official-plate S1 context, each model prediction is replaced by a prediction from a different unseen compound in that same context when available. The target and control anchor are held fixed.
- S1 paired uncertainty: 2,000 compound-cluster bootstrap draws, seed `20260816`, six unseen-compound clusters.

## R15-001: control-pool sensitivity

The shared official evaluator was run on the frozen line and all three diagnostic variants. The complete machine-readable result is [module_metrics.json](/data/LXM/VC/outputs/p15_control_pool/official_evaluator/module_metrics.json). The independent P1.5 evaluator, including train-only pool and 200 permutations, is [p15_results.json](/data/LXM/VC/outputs/p15_control_pool/p15_results.json).

| Pool / candidate | matched / total | coverage | absolute PCC | RMSE | FC-PCC | S1 official_plate | S2 residual | S3 FC | DEP direction |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| train-only / baseline | 1198 / 2806 | 0.426942 | 0.988598 | 0.420259 | 0.595048 | 0.195663 | n/a | n/a | 0.901227 |
| train-only / R1 w=.05 | 1198 / 2806 | 0.426942 | 0.988617 | 0.419917 | 0.595048 | 0.207890 | n/a | n/a | 0.901227 |
| train-only / R1 w=.10 | 1198 / 2806 | 0.426942 | 0.988627 | 0.419734 | 0.595048 | 0.217722 | n/a | n/a | 0.901227 |
| train-only / R1 w=.15 | 1198 / 2806 | 0.426942 | 0.988628 | 0.419709 | 0.595048 | 0.224788 | n/a | n/a | 0.901227 |
| official train|val / baseline | 2806 / 2806 | 1.000000 | 0.988598 | 0.420259 | 0.491355 | 0.203387 | 0.407705 | 0.891936 |
| official train|val / R1 w=.05 | 2806 / 2806 | 1.000000 | 0.988617 | 0.419917 | 0.491378 | 0.215532 | 0.407705 | 0.891955 |
| official train|val / R1 w=.10 | 2806 / 2806 | 1.000000 | 0.988627 | 0.419734 | 0.491394 | 0.225294 | 0.407705 | 0.891970 |
| official train|val / R1 w=.15 | 2806 / 2806 | 1.000000 | 0.988628 | 0.419709 | 0.491403 | 0.232308 | 0.407705 | 0.891985 |

Official-pool scenario FC-PCC for the baseline is `seen=0.671725`, `S1=0.586075`, `S2=0.426484`, `S3=0.344605`. R1 changes these only at the fourth-to-sixth decimal in `seen`/`S1`; S2 and S3 are unchanged because the R1 operation is applied only to train-pool matched rows. The weighted 10% time/S3 slot is therefore reported through the official S3 sensitivity here; no new time-specific model is introduced by P1.5.

## R15-002: R1 positive control and null

| Candidate | official S1 | ΔS1 vs baseline | 95% cluster-bootstrap CI | 200-null mean | true − null |
|---|---:|---:|---:|---:|---:|
| baseline | 0.203386812 | — | — | 0.203386812 | 0.00e+00 |
| R1 w=.05 | 0.215531727 | +0.012144916 | [+0.005160723, +0.020515111] | 0.215531727 | +5.55e-17 |
| R1 w=.10 | 0.225294428 | +0.021907616 | [+0.007973458, +0.038527590] | 0.225294428 | +2.78e-17 |
| R1 w=.15 | 0.232308063 | +0.028921252 | [+0.008677482, +0.053010378] | 0.232308063 | −2.78e-17 |

The train-only-pool sensitivity gives the same null conclusion: true-minus-null gaps for `w=.05/.10/.15` are `−8.33e-17/+2.78e-17/+5.55e-17`. This is the expected algebraic signature of reweighting the shared train control anchor, not adding compound biology.

## Reproducibility ledger

Commands (all run from `/data/LXM/VC`):

```bash
/home/lxm/anaconda3/envs/tl/bin/python scripts/p15_control_pool.py \
  --baseline outputs/r1_convex_mix_3seed/prediction_val_mean50_3seed.parquet \
  --out outputs/p15_control_pool

/home/lxm/anaconda3/envs/tl/bin/python scripts/evaluate_official_modules.py \
  --candidates outputs/p15_control_pool/candidates_official.json \
  --baseline baseline --out outputs/p15_control_pool/official_evaluator \
  --workers 1 --bootstrap 2000
```

Source and artifact SHA256:

| Artifact | SHA256 |
|---|---|
| `scripts/p15_control_pool.py` | `57f51ca21a695f467ba88574cc9d9f6a8ab2c02c76575e25df1bbaf922997122` |
| `scripts/evaluate_official_modules.py` (unchanged) | `6eb1e41ce0f9028714ee00bd73d837c07ed05ed473cfa545f804acaa70854068` |
| frozen baseline parquet | `b076ddea10236b8e0b0eb4613b5ed2d565cae9e9934634a985978faa2222e31d` |
| existing R1 w=.05 parquet | `e9c68dec6263f2cdaf962a11c54d404020386b6635c6a35411c5203bc7252361` |
| existing R1 w=.10 parquet | `2f96ecf37c4779f61e8b72abb5178c83f3ad53f998d5aac6526e3e0710b409a6` |
| P1.5 R1 w=.15 parquet | `dc7aa4a7dc07dccaa1156e72aefc6cdb18478077de35499a29bd0b5c426c7ec4` |
| `outputs/p15_control_pool/p15_results.json` | `84584098c6e57b164ef61b65021b0797f07302457424e4966d4df914d4da3865` |
| official evaluator `module_metrics.json` | `964563fd092f4c022a608fb6d104c4f1b1293127a92e1fda1ed1566e9ff43bc5` |

The first attempted invocation with the system Python failed before reading data because its NumPy 2.2.6 was ABI-incompatible with the installed pandas/pyarrow. The successful runs used the frozen project `tl` environment (Python 3.8.18, NumPy 1.24.3, pandas 2.0.3, pyarrow 17.0.0). This is an environment blocker for the default interpreter only; it does not affect the successful experiment.

## Position

R15 establishes a protocol warning and a clean negative result: train-only control pools silently discard 1,608 matched validation treatments, while official-pool FC is the only full-coverage score. R1 can move S1 and RMSE, but its compound permutation null moves identically. **Do not adopt R1, do not change the submission line, and do not treat the train-only FC number as the official score.**
