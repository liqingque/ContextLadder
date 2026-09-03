# E2: Train-Internal Four-Axis Grouped Pseudo-OOD Evaluation Protocol

Spec: `ContextLadder_复赛综合审查与提升实验计划.md` §7 (L846-886), design notes in §6.6.2 (L732-758).

## 1. Why

Official validation has exactly one unseen strain and six unseen compounds — not
enough to select between strain-side model variants, and not enough to see
whether "unseen X performs worse" generalizes beyond that one X. This protocol
builds four independent leave-one-group-out sweeps entirely inside the 5,920
`split_final == "train"` rows, so any candidate can be scored on plate,
strain, compound, and time generalization without ever touching validation or
test truth. It is intended as a reusable, standing evaluation harness, not a
one-off number.

**Hard constraints honored throughout:**
- `scripts/run_hcce.py` and `scripts/a2b_train_variants.py` are imported only,
  never edited (verified: `git status` shows no changes to either file).
- Every fold refits categorical vocabularies and normalisation statistics
  from scratch on fold-train rows only, via the untouched
  `HCCEMetaEncoder.fit()` / `fit_model_variant()`. The held-out entity's rows
  are transformed through that fold-train-only vocabulary, so they map to the
  reserved unknown token (index 0) for whichever field was left out.
- No official-validation or test truth is read anywhere in
  `scripts/build_grouped_ood_folds.py`, `scripts/evaluate_grouped_ood.py`, or
  `scripts/grouped_ood_common.py`.
- No plain random K-fold: every split is entity-grouped (whole
  plate/strain/compound/time-value on one side of the split only).

## 2. The four axes

| Axis | Field | Entities in train | Folds | Mode |
|---|---|---:|---:|---|
| plate | `Yeast_cell_plate` | 144 (22-71 rows each) | 12 | grouped (LPT bin-packing) |
| strain | `Strains` | 4 (BAH, CEK, CGD, DHY210) | 4 | complete leave-one-out |
| compound | `perturbation_no_concentration` | 40 (64-543 rows each) | 10 | grouped (LPT bin-packing) |
| time | `pert_time` | 6 (15/30/60/90/120/240) | 6 | complete leave-one-out |

**Grouping algorithm (plate, compound):** longest-processing-time (LPT)
greedy bin packing — entities sorted by descending row count, each assigned
to the currently smallest-total bin. This balances fold-test row counts
(plate folds: 486-495 rows; compound folds: 545-630 rows) without ever
splitting an entity's rows across folds. 144-way LOPO was explicitly rejected
per the task spec (would cost well over an hour per variant); 12 grouped
folds keep full-4422-protein HCCE training tractable (~30-35s/fold observed).

**Time axis / "blocked, not random":** with only 6 distinct `pert_time`
values in train, each fold holds out exactly one value — trivially a
contiguous block since it is a single point, so the full 6-fold sweep
already satisfies "leave out a contiguous interval, not random rows" without
needing an arbitrary multi-point grouping. The fold holding out the **latest**
time point (240) is flagged `is_latest_block: true` in the manifest, matching
the plan's specific "hold out the latest time point" scenario; the other five
folds cover the interior/early blocks so the axis isn't reduced to one point
estimate. Section 4 shows this is exactly the hardest fold.

**Strain axis / statistical power:** only 4 entities exist in train at all,
so leave-one-strain-out is inherently a 4-cluster experiment. This is stated
explicitly here and in `GATE.json.low_power_axes.strain`: the strain-axis
macro metric and its bootstrap CI can rule out a large regression but cannot
resolve small deltas between candidates. It is a real improvement over
official validation's single unseen strain (4x the coverage, plus proper
cluster CIs) but does not fix the fundamental scarcity of strain diversity in
this dataset.

## 3. Evaluation unit and metrics

- Every plate/strain/compound/time-value is a statistical unit.
- **Entity-macro aggregation**: per-entity metrics computed first (mean
  row-wise PCC within that entity's held-out rows, RMSE/MAE over its finite
  log2 entries), then equal-weight mean across entities — a 543-row compound
  never outweighs a 64-row one.
- **Entity-cluster bootstrap**: 2,000 resamples of entities (not rows) per
  axis, macro-averaged each draw, reported as [2.5%, 97.5%] percentile CI.
- **Matched-control FC**: computed per fold via
  `src/evaluation/control_matching.matched_fc`, with the control pool
  restricted to that fold's own fold-train rows (`control_pool_mask`) so no
  fold-test truth is ever used as a control baseline.
- **Unknown-vs-known gap**: reported per axis, but with an explicit
  definition caveat — see §5.

## 4. Baseline results — HCCE `mask_compound` variant, seed 20260810, 40 epochs

Command: `python scripts/evaluate_grouped_ood.py --variant mask_compound --seed 20260810 --epochs 40 --n-bootstrap 2000`
(single seed; see §7 for why).

| Axis | n_entities | n_folds | total test rows | sample-PCC macro [95% CI] | log2 RMSE macro [95% CI] | log2 MAE macro |
|---|---:|---:|---:|---|---|---|
| plate | 144 | 12 | 5,920 | 0.9903 [0.9893, 0.9911] | 0.3769 [0.3620, 0.3929] | 0.2624 |
| strain | 4 | 4 | 5,920 | 0.9757 [0.9724, 0.9795] | 0.6073 [0.5507, 0.6473] | 0.4071 |
| compound | 40 | 10 | 5,920 | 0.9879 [0.9836, 0.9916] | 0.3874 [0.3461, 0.4355] | 0.2637 |
| time | 6 | 6 | 5,920 | 0.9897 [0.9875, 0.9911] | 0.3948 [0.3692, 0.4345] | 0.2704 |

Full CSVs: `outputs/grouped_ood/per_fold_metrics.csv` (32 rows, one per fold)
and `outputs/grouped_ood/per_entity_metrics.csv` (194 rows, one per entity
per axis).

**Read of the ranking:** plate is the easiest axis to generalize on
(highest PCC, lowest RMSE) — consistent with the model's plate-residual
design giving unseen plates a graceful source/instrument fallback rather than
memorizing a lookup table. Strain is the hardest axis by a wide margin (RMSE
+61% vs plate), which is the axis E1's strain-side interventions are meant to
target — but with only 4 clusters this number should be treated as a coarse
signal, not a precise estimate (CI half-width ~0.048 PCC / ~0.048 RMSE, both
wide relative to the plate/compound/time CIs).

**Per-entity highlights** (from `per_entity_metrics.csv`):
- Strain, full ranking (best -> worst PCC): CGD (0.9816) > CEK (0.9763) >
  DHY210 (0.9731) > BAH (0.9717). No strain collapses to near-zero PCC; the
  spread (0.972-0.982) is much tighter than the axis-level RMSE gap vs other
  axes would suggest, i.e. the strain-axis degradation is fairly uniform
  across all 4 strains rather than driven by one outlier.
- Time: the held-out **latest block (240)** is the worst of the 6 folds
  (PCC 0.9846, RMSE 0.4892) vs 0.984-0.992 / 0.360-0.393 for the other five —
  a first, direct piece of evidence that extrapolating to the latest
  timepoint is harder than interpolating among interior points, which is
  exactly the question §7.2 poses for the time axis.
- Compound: worst 3 entities are Amiodarone hydrochloride (PCC 0.939),
  Cyclopiazonic acid (0.946), 4-Hydroxytamoxifen (0.957) — all strong-effect
  perturbagens with comparatively few rows (86-93); best 3 are DMSO (0.995,
  n=368), SDS (0.995, n=268), Cisplatin (0.995, n=269) — high-replicate,
  familiar-profile conditions.
- Plate: worst 3 (WAY_CP15, WAY_B-1_P21, WAY_CP16, PCC 0.965-0.970) are all
  small plates (28-35 rows); best 3 (WAY_B-2_P20, WAY_B-1_P20, WAY_B-2_P22,
  PCC 0.995) are mid-sized plates from the same two source batches — no
  single catastrophic plate failure was observed.

**Matched-control FC coverage** (`aggregate_metrics.json`,
`matched_control_fc`): compound axis reaches 95.6-100% coverage per fold
(mean FC-PCC 0.571) because compound is not part of the control-matching
context key set. Plate/strain/time all show **0% coverage** — expected, not a
bug: `match_controls` requires an exact context match including
plate/strain/time, and that is exactly the field each of those folds removes
from the legal (fold-train) control pool, so no legal control can ever share
context with a held-out plate/strain/time-block. This is documented directly
in each axis's `matched_control_fc.note` in `aggregate_metrics.json` — FC is
only an informative axis-4 metric for the compound axis under this protocol.

## 5. Unknown-vs-known gap — definition and honest limitation

`aggregate_metrics.json[<variant>].aggregate[<axis>].unknown_vs_known` reports
a gap, but its "known" side is **not** a true held-out known-entity estimate.
Within a strict leave-one-group-out partition, any row not in the held-out
group was, by construction, part of that fold's training data — there is no
way to get a "known entity, unseen row" evaluation without a second (nested)
level of held-out rows, which was out of scope for this first pass. So
"known" here is the **in-sample fit metric** of the same fold-trained model,
evaluated on its own fold-train rows — an optimistic upper bound, disclosed
as such in the JSON itself (`unknown_vs_known.definition` field, repeated
verbatim in every axis block so it can't be missed downstream).

Observed gaps (unknown RMSE − known-in-sample RMSE):

| Axis | known-in-sample RMSE | unknown (true OOD) RMSE | gap |
|---|---:|---:|---:|
| plate | 0.2831 | 0.3842 | +0.1011 |
| strain | 0.2852 | 0.6073 | +0.3221 |
| compound | 0.2847 | 0.3645 | +0.0798 |
| time | 0.2847 | 0.3948 | +0.1101 |

The strain gap is ~3-4x every other axis's gap — the clearest single piece of
evidence in this report that strain generalization, not plate/compound/time
generalization, is HCCE's actual weak point. A proper (nested-CV) known-vs-
unknown estimate is flagged as follow-up work in §7.

## 6. Leakage assertions

`outputs/grouped_ood/leakage_assertions.json`: **32/32 fold assertions pass**
(`all_pass: true`), each checking, per fold: fold-train/fold-test entity sets
are disjoint, fold-train/fold-test row indices are disjoint, their union
covers every train row, and both sides are non-empty. `build_grouped_ood_folds.py`
raises `AssertionError` immediately on any failure (`scripts/grouped_ood_common.py::assert_no_leakage`)
rather than logging a warning and continuing; `evaluate_grouped_ood.py`
re-runs the same assertion again at evaluation time as defense in depth
before every fold is trained, and refuses to run at all if the manifest's
own `leakage_summary.all_pass` is false.

## 7. Reuse interface (for E1's six variants or any future candidate)

The core function is candidate-agnostic:

```python
from scripts.grouped_ood_common import load_universe, build_hcce_variant_functions
from scripts.evaluate_grouped_ood import evaluate_four_axis
import json

manifest = json.loads(open("outputs/grouped_ood/split_manifest.json").read())
rh, meta_train, y_train, proteins, mapping = load_universe()

# Any HCCE variant from scripts/a2b_train_variants.py:
train_fn, predict_fn = build_hcce_variant_functions(rh, mapping, variant="mask_compound_denoise")

result = evaluate_four_axis(
    train_fn, predict_fn, candidate_name="mask_compound_denoise",
    manifest=manifest, meta_train=meta_train, y_train=y_train, mapping=mapping,
    seed=20260810, epochs=40, device="cuda", n_bootstrap=2000,
)
# result["per_fold"], result["per_entity"] are DataFrames; result["aggregate"] is the per-axis dict.
```

For a non-HCCE candidate, only `train_fn`/`predict_fn` need to be supplied
with this exact contract:

```python
def train_fn(meta, y, fit_idx, mapping, seed, epochs, device) -> state
def predict_fn(state, meta_subset, device) -> np.ndarray  # (len(meta_subset), n_proteins), log2 space
```

`scripts/evaluate_grouped_ood.py`'s CLI already appends new candidates into
the same `outputs/grouped_ood/{per_fold_metrics.csv,per_entity_metrics.csv,aggregate_metrics.json}`
(dedup by `(candidate, axis, fold_id[, entity])`, keep-last), so running it
again with `--variant <other>` accumulates a comparison table rather than
overwriting the baseline row — this is how E1's six variants would plug in.

## 8. Deliverables

```
outputs/grouped_ood/
├─ split_manifest.json        4-axis fold definitions: entities, sample IDs, row counts (build_grouped_ood_folds.py)
├─ per_fold_metrics.csv       32 rows, one per (axis, fold): unknown + in-sample metrics, FC, timing
├─ per_entity_metrics.csv     194 rows, one per (axis, entity): n_rows, sample-PCC, RMSE, MAE, R2
├─ aggregate_metrics.json     per-axis entity-macro metrics + bootstrap CI + unknown-vs-known + FC, per candidate
├─ leakage_assertions.json    32/32 fold-level leakage checks, all_pass=true
└─ GATE.json                  protocol-construction gate (leakage + fold-count contract + eval completeness)
```

Scripts (new files only; `run_hcce.py`/`a2b_train_variants.py` untouched):
`scripts/grouped_ood_common.py`, `scripts/build_grouped_ood_folds.py`,
`scripts/evaluate_grouped_ood.py`, `scripts/finalize_grouped_ood_gate.py`.

## 9. What GATE.json actually gates

`GATE.json.overall_pass = true` means: the split has zero leakage, the fold
counts match the task's contract (plate 12 / strain 4 / compound 10 / time
6), and the baseline was evaluated on all four axes. **It is not a model-
selection gate.** It does not certify that `mask_compound` should replace any
frozen prediction, and it does not compare candidates — that comparison is
exactly what this harness exists to support once E1's variants are run
through it. `GATE.json.scope_note` states this explicitly so it cannot be
misquoted as a model-quality sign-off.

## 10. Limitations / not done

- **Single seed.** Only seed 20260810 was run for the baseline (~1000s wall
  time for all 32 folds on one contended RTX 3090; the machine had 4 GPUs at
  97-100% utilization from other concurrent jobs throughout this run). The
  plan's §6.6.3 asks for 3-seed mean/std/sign-agreement; the harness accepts
  `--seed` and could be re-run for 2 more seeds (~15-20 more minutes each
  under similar contention), but that was not executed here.
- **"Known" is in-sample, not held-out-known.** See §5 — a rigorous
  known-vs-unknown estimate needs a nested (double) cross-validation
  (held-out rows even for known entities), not implemented.
- **Only one candidate evaluated** (`mask_compound`, the current baseline
  variant named in the task). E1's six variants were not available to this
  run; the reuse interface in §7 is what they would call.
- **Matched-control FC is only informative for the compound axis** under
  this protocol, for the structural reason in §4 — not a bug, but it means
  §6.6.3's "matched-control FC" metric doesn't add axis coverage beyond
  compound here.
- **Strain axis is inherently low-power** (n=4 clusters) — stated in
  `GATE.json.low_power_axes` and repeated here per the task's explicit
  instruction not to hide this.
