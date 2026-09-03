# E1 x E2: Does the plate-residual mechanism (e1e/e1f) actually help on unseen plates?

Bridges `outputs/e1_unknown_fallback/E1_REPORT.md` (which could only report the plate
mechanism's benefit as **N/A** — the frozen official validation has zero unseen-plate rows)
into `outputs/grouped_ood/` (E2's train-internal, leave-one-plate-out harness, 12 folds,
144 plate entities, split_final=='train' only). This closes exactly the gap E1 flagged in
its own §7.2 limitation.

**Bottom line up front: no measurable benefit from e1e/e1f's plate mechanism was found on
the plate axis — the one axis specifically built to test it.** Direction vs the raw e1a
baseline is inconsistent across seeds (2/3 favorable, 1/3 unfavorable, for both e1e and
e1f), every measured delta is 5-15x smaller than the entity-cluster bootstrap CI half-width,
and a per-plate breakdown shows the mechanism helps roughly half the plates and hurts the
other half with no relationship to plate size. This is a negative result, reported as such.

## 1. What was built

New files (all new; the five files listed as read-only in the task were only imported):

- `scripts/evaluate_e1_grouped_ood.py` — adapter. Wraps
  `scripts.e1_neutral_unknown.fit_model_variant_e1` / `NeutralUnknownHCCEModel` (E1's
  `UnknownFallbackConfig`-driven variants e1a-e1f) into the
  `train_fn(meta, y, fit_idx, mapping, seed, epochs, device) -> state` /
  `predict_fn(state, meta_subset, device) -> ndarray` contract that
  `scripts.evaluate_grouped_ood.evaluate_four_axis` expects, exactly mirroring
  `scripts.grouped_ood_common.build_hcce_variant_functions`'s pattern for the HCCE
  variants it already supports. `predict_fn` returns `mean50 = 0.5*legacy + 0.5*film`,
  the same ensemble head E1's own report and E2's baseline both use.
- The same file also contains `RngMatchedPlaceboModel` and `fit_placebo_variant` — a
  placebo control for the RNG confound the E1 report flagged as unresolved (see §3).
- `outputs/e1_grouped_ood/` — all data products (below).

`scripts/grouped_ood_common.py`, `scripts/evaluate_grouped_ood.py`,
`scripts/e1_neutral_unknown.py`, `scripts/run_hcce.py`, `scripts/a2b_train_variants.py`
were imported only, never edited (this repo's `.git` has no functional history to diff
against, so this was verified by filesystem mtimes instead: all five files' last-modified
timestamps predate this session's start, and none was ever opened with a write tool here).
`runs/final/`, `configs/final.yaml`, `prediction.csv` were never
opened. All fitting and evaluation used only `split_final=='train'` rows (5,920), reused
verbatim from `scripts.grouped_ood_common.load_universe()`; no validation or test truth
was read anywhere in this work.

## 2. What ran

| Axis | Candidates | Seeds | Folds | Notes |
|---|---|---|---|---|
| **plate** | e1a, e1c, e1e, e1f, **e1a_rng_matched_ef** (placebo) | 20260810, 3407, 42 (all 3) | 12 | **Core deliverable — full 3-seed coverage** |
| strain | e1a, e1c, e1e, e1f | 20260810 only | 4 | Supplementary; single seed |
| compound | e1a, e1c, e1e, e1f | 20260810 only | 10 | Supplementary; single seed |
| time | e1a, e1c, e1e, e1f | 20260810 only | 6 | Supplementary; single seed |

e1b and e1d were **not** re-run through E2. The E1 report already established
`e1b == e1d` bit-for-bit on every subset where no non-compound field is unseen, and
their only-ever-difference from e1a (zero-frozen non-compound unknown embeddings,
*without* the plate-residual restructuring) is not the mechanism this task asks about —
plate-axis evidence for e1b/d would answer a different question (does zero-freezing help
strain-type fields in general) than the one asked (does the plate-residual restructuring
help unseen plates). Given the GPU/time budget, effort went to 3-seed depth on the 4
requested candidates plus the placebo instead of 1-seed breadth on 2 more candidates.

Compute: plate axis = 4 candidates x 3 seeds, run in parallel across 4 GPUs (one candidate
per GPU, `CUDA_VISIBLE_DEVICES`-pinned, 3 seeds sequential per GPU) = ~17 minutes wall time
for the plate-axis requirement. Placebo (plate axis, 3 seeds) and the 3 supplementary axes
(4 candidates x 1 seed) ran afterward on the freed GPUs, ~18 more minutes wall time.
Total: ~35 minutes wall clock, ~4,180s (plate) + ~1,040s (placebo) + ~2,170s (other axes)
= ~7,390s of GPU-seconds. GPU utilization was 0-20% from other users' jobs throughout this
window (checked via `nvidia-smi` before and during the run) — much lighter than the
"99-100%, all 4 GPUs" contention described in the task, which is why 3-seed plate-axis
coverage plus all 3 supplementary axes plus the placebo all fit inside the time budget.

## 3. The RNG confound: what it actually is (correcting E1's own hypothesis)

E1's report (§7.3) flagged an unresolved confound: "constructing zero-freeze `nn.Embedding`
objects consumes additional torch global RNG draws ... which shifts every subsequent
dropout mask." This was investigated empirically rather than taken on faith, because it
determines whether a placebo control is even the right tool.

**That specific causal story is wrong.** Verified directly (small standalone probes, not
included as a deliverable): torch 2.1.0's CPU default generator and CUDA default generator
are independent streams. A deliberate burn of 2,000,000 CPU-only random draws inserted
between two otherwise-identical model constructions produces **bit-identical** trained
weights on CUDA (state_dict comparison after 5 optimizer steps was exact). Since all of
`NeutralUnknownHCCEModel`'s extra embedding/linear construction for zero-freeze happens on
CPU (parameters are created before `.to(device)` is ever called), it cannot be reaching the
CUDA-side dropout RNG stream used during training.

**The real mechanism is different and more specific.** `NeutralUnknownHCCEModel.encode()`'s
plain path (e1a/e1c/e1b/e1d) calls exactly two CUDA `nn.Dropout(0.10)` ops on a
`(batch, 128)` tensor while building `measure` (`self.measure(...)`, then
`self.legacy_measure(...)`). e1e/e1f's `hierarchical_measure_context` calls **three**:
`measure_base(...)`, `plate_residual(...)`, then (back in `encode()`) `self.legacy_measure
(...)` — one extra `(batch,128)` Dropout call, every forward pass, every training step, for
the whole run. CUDA's default generator is a single, continuously-advancing stream across
the entire training loop (never reset between batches), so this one extra call per step
shifts every dropout mask computed afterward in that step, and the shift compounds through
every later step. Confirmed directly: with identical seeds, identical initial weights, and
a probe batch built from its own explicit generator (so it never touches the shared default
CUDA stream), e1a and e1e's "legacy" output head — built from a one-hot-only computation
path that shares **no embedding weights** with either mechanism, so it should be perfectly
RNG-independent of embedding content — still diverges (max abs diff 0.0080). That diff is
the fingerprint of an RNG-stream shift, not a weight-content difference. Repeating the same
test for e1a vs e1b (zero-freeze only, `strict_plate_residual=False`, so it keeps e1a's
exact two-call dropout structure) gives **exactly 0.0** diff on that same legacy head — i.e.
**e1b/e1d's difference from e1a is RNG-clean; only e1e/e1f are RNG-confounded.**

**The placebo control** (`RngMatchedPlaceboModel` in `scripts/evaluate_e1_grouped_ood.py`):
forward math is bit-identical to e1a (`cfg=UnknownFallbackConfig()`, no mechanism switches
on at all), but `encode()` replays one extra, discarded `F.dropout(zeros, p=0.10,
training=True)` call on a `(batch,128)` tensor right after computing `measure`, matching
e1e/e1f's extra call count and shape. This closes **most, not all**, of the gap: on the same
probe batch, the legacy-head diff shrinks from 0.0080 (e1a vs e1e) to 0.0018 (placebo vs
e1e) — roughly a 4-5x reduction. It is not a perfect replay (the extra call sits in a
different position relative to `legacy_measure` — e1e's is "between," the placebo's is
"after" — which changes `legacy_measure`'s own dropout mask even though the *total* stream
displacement afterward is the same). Every plate-axis result below **for e1e/e1f is
reported against both the raw e1a baseline and this partial-RNG-matched placebo**, and the
gap between those two comparisons is itself reported as the (unresolved) residual size of
the confound.

## 4. Plate axis — the core result (12 folds, 144 plates, 3 seeds)

Entity-macro sample-PCC and log2-RMSE, 2,000-resample entity-cluster bootstrap 95% CI
(seed 20260902, matching E2's own bootstrap seed convention):

| candidate | seed | PCC macro | PCC 95% CI | RMSE macro | RMSE 95% CI |
|---|---:|---:|---|---:|---|
| e1a | 20260810 | 0.99026 | [0.98934, 0.99113] | 0.37687 | [0.36195, 0.39292] |
| e1a | 3407 | 0.99012 | [0.98917, 0.99099] | 0.37972 | [0.36455, 0.39593] |
| e1a | 42 | 0.99008 | [0.98910, 0.99099] | 0.38118 | [0.36544, 0.39840] |
| e1c | 20260810 | 0.99029 | [0.98938, 0.99115] | 0.37643 | [0.36165, 0.39230] |
| e1c | 3407 | 0.99013 | [0.98919, 0.99101] | 0.37943 | [0.36437, 0.39554] |
| e1c | 42 | 0.99009 | [0.98912, 0.99101] | 0.38089 | [0.36501, 0.39836] |
| e1e | 20260810 | 0.99022 | [0.98931, 0.99108] | 0.37825 | [0.36330, 0.39408] |
| e1e | 3407 | 0.99023 | [0.98934, 0.99107] | 0.37780 | [0.36321, 0.39297] |
| e1e | 42 | 0.99011 | [0.98915, 0.99100] | 0.38056 | [0.36492, 0.39754] |
| e1f | 20260810 | 0.99021 | [0.98932, 0.99105] | 0.37792 | [0.36349, 0.39346] |
| e1f | 3407 | 0.99017 | [0.98927, 0.99101] | 0.37852 | [0.36413, 0.39431] |
| e1f | 42 | 0.99010 | [0.98918, 0.99097] | 0.38073 | [0.36564, 0.39708] |
| e1a_rng_matched_ef (placebo) | 20260810 | 0.99018 | [0.98924, 0.99107] | 0.37845 | [0.36354, 0.39479] |
| e1a_rng_matched_ef (placebo) | 3407 | 0.99016 | [0.98922, 0.99101] | 0.37896 | [0.36414, 0.39481] |
| e1a_rng_matched_ef (placebo) | 42 | 0.99011 | [0.98916, 0.99100] | 0.38077 | [0.36527, 0.39758] |

3-seed mean +- std:

| candidate | PCC mean +- std | RMSE mean +- std |
|---|---|---|
| e1a | 0.990149 +- 0.000095 | 0.379259 +- 0.002191 |
| e1c | 0.990170 +- 0.000101 | 0.378915 +- 0.002272 |
| e1e | 0.990186 +- 0.000066 | 0.378869 +- 0.001482 |
| e1f | 0.990159 +- 0.000058 | 0.379056 +- 0.001481 |
| e1a_rng_matched_ef | 0.990149 +- 0.000038 | 0.379394 +- 0.001221 |

**Direction consistency (RMSE, candidate - e1a; negative = better):**

| | seed 20260810 | seed 3407 | seed 42 | 3/3 consistent? |
|---|---:|---:|---:|---|
| e1c - e1a | -0.00044 | -0.00029 | -0.00030 | **Yes, all better** |
| e1e - e1a | **+0.00138** | -0.00193 | -0.00062 | **No** (2/3 better, 1/3 worse) |
| e1f - e1a | **+0.00104** | -0.00120 | -0.00045 | **No** (2/3 better, 1/3 worse) |
| e1a_rng_matched_ef - e1a | **+0.00157** | -0.00076 | -0.00041 | No (2/3 better, 1/3 worse) |

**Direction consistency (RMSE, mechanism vs its RNG-matched placebo; negative = mechanism better):**

| | seed 20260810 | seed 3407 | seed 42 | 3/3 consistent? |
|---|---:|---:|---:|---|
| e1e - placebo | -0.00020 | -0.00117 | -0.00021 | **Yes, all better** |
| e1f - placebo | -0.00053 | -0.00044 | -0.00004 | **Yes, all better** |

**Per-plate breakdown** (144 plates, RMSE, candidate vs e1a, same seed):

| candidate | seed 20260810 | seed 3407 | seed 42 |
|---|---|---|---|
| e1c | 80/144 better | 82/144 better | 85/144 better |
| e1e | 66/144 better | 80/144 better | 68/144 better |
| e1f | 67/144 better | 69/144 better | 63/144 better |

Correlation of (e1e RMSE - e1a RMSE) with plate size (n_rows, 22-71): **r = -0.028** —
no relationship. The plate mechanism was designed to give small/sparse plates a graceful
source+instrument fallback; if it worked as intended, it should disproportionately help
small plates. It does not: the biggest single win (WAY_CP8, -0.020 RMSE) and the biggest
single loss (WAY_CP14, +0.050 RMSE) are both mid-sized plates (34, 32 rows), and the 15
biggest wins/losses are otherwise indistinguishable in size from the bulk of plates that
show no visible pattern.

### Reading this table honestly

1. **e1c is a small, 3/3-seed-consistent, and per-plate-majority (80-85/144) improvement
   over e1a on the axis this whole exercise exists to test.** The effect size (-0.0003 to
   -0.0004 RMSE, roughly 0.08-0.1% relative) is far smaller than the entity-cluster
   bootstrap CI half-width (~0.008 RMSE) — not statistically distinguishable by that CI —
   but it is directionally consistent across every seed and the majority of individual
   plates, matching the E1 report's own finding that e1c is "mechanically correct,
   empirically neutral-to-slightly-positive." It is the only one of the four required
   candidates with a clean signal in any direction here.
2. **e1e/e1f show no reliable direction against raw e1a**: 2 of 3 seeds favorable, 1
   unfavorable, for both. Comparing to the RNG-matched placebo instead (the fairer
   comparison, since it isolates the mechanism from the dropout-stream shift documented in
   §3) does turn up a 3/3-seed-consistent small improvement (-0.0002 to -0.0012 RMSE for
   e1e, -0.00004 to -0.0005 for e1f) — but this improvement is 7-40x smaller than the
   entity-cluster bootstrap CI half-width, and the placebo control itself is only a
   partial (~4-5x, not complete) RNG match, so residual RNG drift cannot be fully ruled
   out as the source of even this small, consistent-looking delta.
3. **Per-plate, the mechanism helps a bare-plurality-to-minority of plates (63-80 of 144,
   i.e. 44-56%)** with no relationship to plate size — the exact axis (small/sparse
   plates) the mechanism targets shows no differential benefit. This is the most direct
   evidence against a real, exploitable effect: a genuine fallback-quality improvement for
   unseen plates should show up as a size-correlated, majority-of-plates win, and it does
   not.

**Verdict: the plate-residual mechanism (e1e/e1f) has no detectable, practically
meaningful benefit on the one held-out-plate protocol built specifically to test it.**
Whatever small, direction-consistent delta survives RNG-matching is far inside the
bootstrap noise floor and should not be treated as evidence the mechanism works, nor
strong evidence it actively hurts — it is a genuine null result, not "N/A" any more.

## 5. Supplementary axes (single seed 20260810 — context, not conclusions)

| axis | candidate | n_entities | n_folds | PCC macro | RMSE macro | RMSE 95% CI |
|---|---|---:|---:|---:|---:|---|
| strain | e1a | 4 | 4 | 0.97570 | 0.60730 | [0.55074, 0.64730] |
| strain | e1c | 4 | 4 | 0.97569 | 0.60769 | [0.55172, 0.64805] |
| strain | e1e | 4 | 4 | 0.97560 | 0.60850 | [0.55343, 0.64795] |
| strain | e1f | 4 | 4 | 0.97567 | 0.60692 | [0.55245, 0.64468] |
| compound | e1a | 40 | 10 | 0.98790 | 0.38742 | [0.34611, 0.43553] |
| compound | e1c | 40 | 10 | 0.98790 | 0.38742 | [0.34611, 0.43553] |
| compound | e1e | 40 | 10 | 0.98788 | 0.38766 | [0.34635, 0.43603] |
| compound | e1f | 40 | 10 | 0.98786 | 0.38781 | [0.34645, 0.43621] |
| time | e1a | 6 | 6 | 0.98972 | 0.39484 | [0.36918, 0.43453] |
| time | e1c | 6 | 6 | 0.98976 | 0.39331 | [0.36797, 0.43253] |
| time | e1e | 6 | 6 | 0.98973 | 0.39458 | [0.36986, 0.43031] |
| time | e1f | 6 | 6 | 0.98952 | 0.39705 | [0.37182, 0.43654] |

Single seed only — no direction-consistency claim is possible here, these numbers are
context, not a conclusion. Two structural sanity checks pass: (a) e1a and e1c are
bit-identical on the compound axis (0.38742 = 0.38742), matching the E1 report's own
finding that `mask_unknown_onehot` is a no-op whenever no non-compound field is ever
unseen in a fold's own training data — true for the compound axis, where plate/strain/etc
are all seen; (b) all four candidates land within ~0.002 RMSE of each other on every one
of these three axes, i.e. no candidate shows a large, order-of-magnitude effect on axes the
plate mechanism was never designed to touch.

## 6. Files produced

```
outputs/e1_grouped_ood/
├─ E1_GROUPED_OOD_REPORT.md      this report
├─ per_fold_metrics.csv          260 rows: (candidate x seed, axis, fold_id) -> unknown/in-sample metrics, FC, timing
├─ per_entity_metrics.csv        2,360 rows: (candidate x seed, axis, fold_id, entity) -> n_rows, sample_pcc, rmse, mae, r2
├─ aggregate_metrics.json        per (candidate x seed) -> per-axis entity-macro + bootstrap CI + unknown-vs-known gap
├─ all_axes_comparison.csv       flattened summary table (family, seed, axis, pcc/rmse macro + CI) used to build this report
├─ plate_axis_comparison.csv     plate-axis-only slice of the above
├─ logs/                         raw stdout/stderr per launch (e1a.log, e1c.log, e1e.log, e1f.log, placebo.log, otheraxes_gpu{1,2}.log)
└─ _partial_*/                   per-launch intermediate outputs (kept for audit trail; merged into the top-level CSVs/JSON above)

scripts/evaluate_e1_grouped_ood.py   the adapter + placebo model (new file; the 5 read-only
                                      files listed in the task are imported, never edited --
                                      verified via filesystem mtimes, see §1)
```

Candidate keys in the CSVs/JSON are `{e1a,e1c,e1e,e1f,e1a_rng_matched_ef}__seed{20260810,3407,42}`
(plate axis has all 3 seeds x 5 candidate families incl. placebo; strain/compound/time
have only `__seed20260810` rows for the 4 non-placebo families).

## 7. Honest limitations

1. **Supplementary axes are single-seed.** Strain (4 folds), compound (10 folds), and time
   (6 folds) were only run at seed 20260810 for all 4 candidates. GPU load was light enough
   during this run that 3-seed coverage was feasible in principle, but plate-axis depth (the
   axis the task explicitly prioritized) was completed first and the remaining budget went
   to axis *breadth* rather than a 3rd seed on non-plate axes. No direction-consistency
   claim is or should be made for strain/compound/time here.
2. **The RNG-matched placebo is a partial control, not an exact one** (~4-5x reduction in
   the measured RNG-stream artifact, not full elimination — see §3). The small,
   3/3-seed-consistent "mechanism vs placebo" delta for e1e/e1f (§4, second table) could
   still partly reflect the ~1/5 of the RNG-stream shift the placebo does not close. This
   report does not claim that delta as clean evidence the mechanism helps; it reports it
   exactly as what it is — a residual too small to act on, next to a residual confound too
   large to declare fully closed.
3. **e1b/e1d were not run through E2** (see §2) — a deliberate scope choice given the task's
   specific question ("does the plate mechanism have measurable value") and time budget, not
   an oversight. E1's own report already gives e1b/d a verdict (regress on strain_unseen)
   on the official validation; nothing here changes or extends that.
4. **Strain axis remains structurally low-power** (n=4 clusters, inherited from E2's own
   documented limitation) — not specific to this bridge, restated here because the
   supplementary strain-axis numbers in §5 inherit it.
5. `enforce_neutral_unknown_contract` is called after training for e1a/e1c/e1e/e1f (mirrors
   `scripts/run_unknown_fallback_ablation.py`'s own convention) — a no-op for e1a/e1c (no
   zero-freeze), idempotent for e1e/e1f. Not called for the placebo, which never sets
   `zero_freeze_unknown_embedding=True` to begin with (`cfg=UnknownFallbackConfig()`, the
   pure e1a config), so there is nothing for it to enforce.
6. No test truth or official-validation truth was read anywhere in this bridge; every
   number above comes from `scripts.evaluate_grouped_ood.evaluate_four_axis`'s own
   leakage-asserting fold loop over `split_final=='train'` rows, re-verified per fold at
   evaluation time (raises immediately on any violation) exactly as E2 already guarantees.

## 8. Recommendation

Unchanged from E1's own recommendation, now with the missing evidence filled in: **do not
adopt e1e/e1f.** E1 could only say the plate mechanism's benefit was untestable on the
frozen validation; this bridge tested it on the one protocol built for exactly that
question and found no practically meaningful effect, in either direction, robust to (a) 3
independent seeds, (b) a partial RNG-confound control, and (c) a per-plate breakdown that
rules out the mechanism's own stated rationale (helping small/sparse plates
disproportionately). e1c remains the only candidate with a small, consistent, low-risk
positive signal — same conclusion the E1 report already reached on the official
validation, now corroborated on an independent, train-internal protocol.
