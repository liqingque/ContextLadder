# E1: Neutral Unknown + Strict Hierarchical Residual — Results Report

Source spec: `ContextLadder_复赛综合审查与提升实验计划.md`, chapter 6 (lines 532–845).
Run date: 2026-09-03. Environment: `/home/lxm/anaconda3/envs/tl/bin/python`, torch 2.1.0,
numpy 1.24.3, CUDA (RTX 3090).

## 0. tl;dr

- **G0 (semantic correctness): PASS.** All 20 property tests pass; the frozen
  `runs/final/prediction.csv` still reproduces byte-for-byte
  (`59f99dc431aa5bd6dc5abb46a5390c64072fda505097cb9523b77198a502b17e`); no test file was ever
  opened; no strain masking was reintroduced.
- **G1 (generalization evidence): FAILS for e1b/e1d/e1e/e1f, PASSES for e1c.** Freezing the
  non-compound unknown embedding to exactly zero (`zero_freeze_unknown_embedding=True`, the
  E1b/d/e/f switch) produces a small but **3/3-seed-consistent regression** on the
  `strain_unseen` subset and a mostly-consistent (2/3-seed) regression on `both_unseen`. Masking
  the unknown one-hot column alone (e1c) is statistically neutral — no consistent direction, well
  inside noise.
- **The plate-residual mechanism (e1e/e1f) is architecturally verified correct (G0) but its
  intended benefit is untestable on this validation split**: the frozen official validation
  contains **zero unseen-plate rows** (and zero unseen-instrument, unseen-source, or
  unseen-medium rows — the *only* field that is ever unseen in this validation split is
  `strain`). Any e1e/e1f regression measured here is inherited from e1d's embedding-freeze
  effect, not evidence against the plate mechanism itself.
- **No candidate is recommended for G2 / competition adoption.** `runs/final/` is unchanged and
  remains the submission.

## 1. Deliverables and file paths

| # | Deliverable | Path |
|---|---|---|
| 1 | E1 model/config module (Step 1–5, subclasses `HCCEModel`, never edits it) | `/data/LXM/VC/scripts/e1_neutral_unknown.py` |
| 2 | Property tests T1–T6 (20 tests, all pass) | `/data/LXM/VC/tests/test_unknown_fallback_contract.py` |
| 3a | Executable variant config | `/data/LXM/VC/configs/e1_unknown_fallback.yaml` |
| 3b | Frozen adoption gates (hashed before training) | `/data/LXM/VC/configs/e1_unknown_fallback_gates.yaml` |
| 4 | Ablation runner (E1a–E1f × 3 seeds) | `/data/LXM/VC/scripts/run_unknown_fallback_ablation.py` |
| 5 | Per-run metrics + aggregate summary | `/data/LXM/VC/outputs/e1_unknown_fallback/{variant}/seed{seed}/metrics.json`, `/data/LXM/VC/outputs/e1_unknown_fallback/summary.json` |
| 6 | This report | `/data/LXM/VC/outputs/e1_unknown_fallback/E1_REPORT.md` |
| — | Gates SHA-256 log (written before the first model trained) | `/data/LXM/VC/outputs/e1_unknown_fallback/gates_sha256.json` |

`scripts/run_hcce.py::HCCEModel`, `scripts/a2b_train_variants.py`, `scripts/train.py`,
`scripts/predict.py`, `configs/final.yaml`, and everything under `runs/final/` were **not
modified**.

## 2. Gate log (frozen before training)

```
configs/e1_unknown_fallback_gates.yaml sha256 = 7a3a09f78128282aff232a2a563b56e9c9d9e2232d967100a8dad440858be09b
configs/e1_unknown_fallback.yaml       sha256 = 62ce23c7b723c074d95e235a423ed2c668992bfc29a64b61e1f823b500d99190
```

Both were computed and written to `outputs/e1_unknown_fallback/gates_sha256.json` and
`summary.json` by `run_unknown_fallback_ablation.py` before the first of the 18 training runs
started; thresholds in the gates file were not touched afterward.

## 3. G0 — semantic correctness

### 3.1 Property tests

```
python -m pytest -q tests/test_unknown_fallback_contract.py
....................                                                     [100%]
20 passed in 3.39s
```

Covers T1 (unknown embedding contract: zero at init, zero after an optimizer step, zero after a
checkpoint save/load roundtrip, compound still receives gradient), T2 (one-hot contract), T3
(plate strict fallback: `output(plate=UNK) == output(disable_plate_residual=True)` at
atol=1e-7/rtol=1e-6), T4 (two known plates are not degenerate), T5 (hierarchical dropout: masking
instrument forces plate off too, masking plate alone leaves the source/instrument base
untouched, eval mode has zero randomness), and T6 (`plate_scale`, learning rate, and dropout
rate each provably change the model/optimizer; an unrecognized config key raises `SchemaError`).

### 3.2 Real-checkpoint spot check (not just synthetic unit-test vocab)

Loading the actual trained seed-20260810 checkpoints:

| field | e1a (baseline) unknown-row max\|·\| | e1f unknown-row max\|·\| |
|---|---|---|
| strain | 2.997 | **0.0** |
| medium | 2.401 | **0.0** |
| source | 2.304 | **0.0** |
| instrument | 2.786 | **0.0** |
| plate | 2.780 | **0.0** |
| compound | (n/a, index 0 is masked-trained in both) | 3.239 (still learned, as intended) |

This is a direct confirmation of the audit document's finding #2: in the current frozen-model
architecture, index 0 for every non-compound field is **not** neutral — it is an untrained,
near-random-initialization vector (magnitude ~2.3–3.0) that a real validation row can activate
whenever that field is unseen. E1b/d/e/f's zero-freeze fixes this exactly as specified; e1c/e1a
do not touch it.

### 3.3 Frozen-model regression check

```
python scripts/predict.py --run-dir runs/final --output <tmp>/p.csv --device cuda
prediction_sha256: 59f99dc431aa5bd6dc5abb46a5390c64072fda505097cb9523b77198a502b17e
```

Matches the required hash exactly, after all E1 code was added to the repository. `HCCEModel`'s
forward logic in `scripts/run_hcce.py` was never edited.

### 3.4 Data boundary

`run_unknown_fallback_ablation.py` reads only `configs/data_paths.yaml`'s `metadata_train_val`
and `proteome_train_val`; `metadata_test` / `proteome_test` are never referenced by path or
loaded. Fitting (`fit_model_variant_e1`) is asserted to use `split_final == 'train'` indices only
(5,920 rows); the 3,038 validation rows (`split_final in {val_strain_only, val_chem_only,
val_both, val_time}`) are prediction/metric targets only.

**G0 verdict: PASS in full.**

## 4. Ablation results (frozen official validation, log2 scale)

6 variants × 3 seeds = 18 runs, 0 failures, ~10.3 minutes total (`elapsed_sec_total = 620.6`).
Head used for all metrics below: `mean50 = 0.5*legacy + 0.5*film`, the same ensemble head used
throughout `scripts/a2b_train_variants.py`'s existing ablations. All fitting uses `split_final ==
'train'`; validation rows are never fit on.

Validation subset sizes (fixed by the frozen split, identical across all variants):
`all=3038, strain_unseen=1547, compound_unseen=1065, both_unseen=269, both_seen_time_shift=157`.
(`both_seen_time_shift` = `val_time`: neither strain nor compound is unseen, only time differs —
reported for context, not one of the three required axes.)

### Subset: `all`

| variant | RMSE (mean±std) | MAE (mean±std) | sample-PCC (mean±std) | RMSE per seed | PCC per seed |
|---|---|---|---|---|---|
| e1a | 0.42558 ± 0.00335 | 0.28268 ± 0.00181 | 0.98832 ± 0.00018 | 20260810:0.42166, 3407:0.42522, 42:0.42986 | 20260810:0.98855, 3407:0.98829, 42:0.98811 |
| e1b | 0.42892 ± 0.00154 | 0.28473 ± 0.00109 | 0.98815 ± 0.00007 | 20260810:0.42688, 3407:0.42927, 42:0.43061 | 20260810:0.98824, 3407:0.98814, 42:0.98807 |
| e1c | 0.42512 ± 0.00282 | 0.28236 ± 0.00152 | 0.98835 ± 0.00014 | 20260810:0.42222, 3407:0.42420, 42:0.42893 | 20260810:0.98852, 3407:0.98835, 42:0.98817 |
| e1d | 0.42876 ± 0.00140 | 0.28460 ± 0.00097 | 0.98816 ± 0.00006 | 20260810:0.42680, 3407:0.42946, 42:0.43002 | 20260810:0.98824, 3407:0.98814, 42:0.98811 |
| e1e | 0.42816 ± 0.00217 | 0.28424 ± 0.00129 | 0.98818 ± 0.00012 | 20260810:0.42520, 3407:0.42894, 42:0.43035 | 20260810:0.98834, 3407:0.98814, 42:0.98806 |
| e1f | 0.42883 ± 0.00220 | 0.28478 ± 0.00134 | 0.98814 ± 0.00012 | 20260810:0.42573, 3407:0.43022, 42:0.43055 | 20260810:0.98831, 3407:0.98808, 42:0.98805 |

### Subset: `strain_unseen` (required axis 1 of 3)

| variant | RMSE (mean±std) | MAE (mean±std) | sample-PCC (mean±std) | RMSE per seed | PCC per seed |
|---|---|---|---|---|---|
| e1a | 0.47855 ± 0.00472 | 0.32517 ± 0.00294 | 0.98509 ± 0.00029 | 20260810:0.47301, 3407:0.47811, 42:0.48454 | 20260810:0.98547, 3407:0.98502, 42:0.98478 |
| e1b | 0.48341 ± 0.00255 | 0.32862 ± 0.00210 | 0.98481 ± 0.00013 | 20260810:0.47990, 3407:0.48442, 42:0.48590 | 20260810:0.98499, 3407:0.98476, 42:0.98469 |
| e1c | 0.47790 ± 0.00398 | 0.32467 ± 0.00251 | 0.98514 ± 0.00022 | 20260810:0.47384, 3407:0.47655, 42:0.48331 | 20260810:0.98542, 3407:0.98514, 42:0.98487 |
| e1d | 0.48315 ± 0.00239 | 0.32839 ± 0.00188 | 0.98484 ± 0.00011 | 20260810:0.47979, 3407:0.48452, 42:0.48513 | 20260810:0.98500, 3407:0.98477, 42:0.98474 |
| e1e | 0.48274 ± 0.00332 | 0.32813 ± 0.00230 | 0.98484 ± 0.00020 | 20260810:0.47821, 3407:0.48390, 42:0.48610 | 20260810:0.98511, 3407:0.98476, 42:0.98464 |
| e1f | 0.48375 ± 0.00341 | 0.32908 ± 0.00239 | 0.98478 ± 0.00020 | 20260810:0.47893, 3407:0.48584, 42:0.48647 | 20260810:0.98507, 3407:0.98466, 42:0.98461 |

### Subset: `compound_unseen` (required axis 2 of 3)

| variant | RMSE (mean±std) | MAE (mean±std) | sample-PCC (mean±std) | RMSE per seed | PCC per seed |
|---|---|---|---|---|---|
| e1a | 0.32549 ± 0.00056 | 0.21704 ± 0.00020 | 0.99332 ± 0.00003 | 20260810:0.32506, 3407:0.32513, 42:0.32629 | 20260810:0.99333, 3407:0.99334, 42:0.99328 |
| e1b | 0.32557 ± 0.00056 | 0.21703 ± 0.00021 | 0.99331 ± 0.00002 | 20260810:0.32532, 3407:0.32505, 42:0.32635 | 20260810:0.99332, 3407:0.99334, 42:0.99328 |
| e1c | 0.32549 ± 0.00056 | 0.21704 ± 0.00020 | 0.99332 ± 0.00003 | 20260810:0.32506, 3407:0.32513, 42:0.32629 | 20260810:0.99333, 3407:0.99334, 42:0.99328 |
| e1d | 0.32557 ± 0.00056 | 0.21703 ± 0.00021 | 0.99331 ± 0.00002 | 20260810:0.32532, 3407:0.32505, 42:0.32635 | 20260810:0.99332, 3407:0.99334, 42:0.99328 |
| e1e | 0.32544 ± 0.00079 | 0.21693 ± 0.00038 | 0.99332 ± 0.00003 | 20260810:0.32511, 3407:0.32468, 42:0.32652 | 20260810:0.99332, 3407:0.99335, 42:0.99328 |
| e1f | 0.32550 ± 0.00063 | 0.21695 ± 0.00025 | 0.99331 ± 0.00002 | 20260810:0.32518, 3407:0.32495, 42:0.32638 | 20260810:0.99332, 3407:0.99334, 42:0.99328 |

### Subset: `both_unseen` (required axis 3 of 3 — double-unknown, strain AND compound)

| variant | RMSE (mean±std) | MAE (mean±std) | sample-PCC (mean±std) | RMSE per seed | PCC per seed |
|---|---|---|---|---|---|
| e1a | 0.49574 ± 0.00506 | 0.33446 ± 0.00303 | 0.98421 ± 0.00032 | 20260810:0.48965, 3407:0.49555, 42:0.50203 | 20260810:0.98462, 3407:0.98417, 42:0.98385 |
| e1b | 0.50112 ± 0.00055 | 0.33782 ± 0.00015 | 0.98387 ± 0.00003 | 20260810:0.50162, 3407:0.50035, 42:0.50138 | 20260810:0.98386, 3407:0.98391, 42:0.98384 |
| e1c | 0.49489 ± 0.00391 | 0.33375 ± 0.00225 | 0.98427 ± 0.00024 | 20260810:0.49040, 3407:0.49435, 42:0.49993 | 20260810:0.98457, 3407:0.98426, 42:0.98399 |
| e1d | 0.50099 ± 0.00075 | 0.33763 ± 0.00077 | 0.98388 ± 0.00004 | 20260810:0.50145, 3407:0.50160, 42:0.49993 | 20260810:0.98386, 3407:0.98385, 42:0.98393 |
| e1e | 0.49760 ± 0.00235 | 0.33545 ± 0.00179 | 0.98408 ± 0.00014 | 20260810:0.49517, 3407:0.50077, 42:0.49686 | 20260810:0.98426, 3407:0.98391, 42:0.98408 |
| e1f | 0.49818 ± 0.00232 | 0.33591 ± 0.00186 | 0.98405 ± 0.00014 | 20260810:0.49591, 3407:0.50137, 42:0.49727 | 20260810:0.98421, 3407:0.98387, 42:0.98405 |

### Subset: `both_seen_time_shift` (context only, not required)

| variant | RMSE (mean±std) | MAE (mean±std) | sample-PCC (mean±std) |
|---|---|---|---|
| e1a | 0.32188 ± 0.00053 | 0.21628 ± 0.00030 | 0.99324 ± 0.00001 |
| e1b | 0.32114 ± 0.00035 | 0.21617 ± 0.00028 | 0.99329 ± 0.00004 |
| e1c | 0.32188 ± 0.00053 | 0.21628 ± 0.00030 | 0.99324 ± 0.00001 |
| e1d | 0.32114 ± 0.00035 | 0.21617 ± 0.00028 | 0.99329 ± 0.00004 |
| e1e | 0.32172 ± 0.00052 | 0.21624 ± 0.00047 | 0.99324 ± 0.00005 |
| e1f | 0.32208 ± 0.00011 | 0.21638 ± 0.00032 | 0.99322 ± 0.00003 |

## 5. Internal-consistency check (why the numbers above should be trusted)

The frozen validation split has an unusual, checkable property: **strain is the only field that
is ever unseen** anywhere in `split_final != 'train'` (verified: 0 unseen plate, 0 unseen
instrument, 0 unseen source, 0 unseen medium rows). Combined with the fact that `compound` is the
only field ever masked to 0 *during training*, this predicts an exact mechanical signature:

- `e1c` (one-hot masking only, no embedding change) must be **bit-identical** to `e1a` on any
  subset that contains no unseen non-compound field, because `field_one_hot`'s masking is a
  no-op whenever `ids.ne(0)` is all-true. Confirmed: `e1c == e1a` to every reported digit on
  `compound_unseen` and `both_seen_time_shift`, and differs only on `strain_unseen`/`both_unseen`.
- `e1b` and `e1d` (which share the same embedding-freeze and therefore the same RNG consumption
  during `__init__`, differing only in the deterministic one-hot mask) must train to **identical
  weights**, because `mask_unknown_onehot` never changes any training-time computation (no
  non-compound field is ever 0 in the training split). Confirmed: `e1b == e1d` to every reported
  digit on `compound_unseen` and `both_seen_time_shift`.

Both predictions hold exactly in the measured data, which is strong evidence the implementation
does what the code and tests claim, not an artifact of a metrics bug.

## 6. G1 — generalization evidence

Per-seed direction vs `e1a` (`worse` = higher RMSE / lower PCC than e1a for that seed):

| variant | strain_unseen (3 seeds worse) | compound_unseen (3 seeds worse) | both_unseen (3 seeds worse) |
|---|---|---|---|
| e1b | PCC 3/3, RMSE 3/3 | PCC 1/3, RMSE 2/3 (noise-level, Δ≈1e-5–3e-4) | PCC 3/3, RMSE 2/3 |
| e1c | PCC 1/3, RMSE 1/3 (noise-level) | PCC 0/3, RMSE 0/3 (bit-identical) | PCC 1/3, RMSE 1/3 (noise-level) |
| e1d | PCC 3/3, RMSE 3/3 | PCC 1/3, RMSE 2/3 (noise-level) | PCC 2/3, RMSE 2/3 |
| e1e | PCC 3/3, RMSE 3/3 | PCC 1/3, RMSE 2/3 (noise-level) | PCC 2/3, RMSE 2/3 |
| e1f | PCC 3/3, RMSE 3/3 | PCC 2/3, RMSE 2/3 (noise-level) | PCC 2/3, RMSE 2/3 |

Applying the frozen gate (`configs/e1_unknown_fallback_gates.yaml`, `min_axes_not_worse_than_e1a:
2`, `seed_majority_required: 2`):

- **e1c: PASS.** All three axes are within noise (no seed-majority-consistent regression
  anywhere); `compound_unseen` is exactly unaffected by construction.
- **e1b, e1d, e1e, e1f: FAIL.** `strain_unseen` is worse with 3/3-seed consistency (Δ sample-PCC
  ≈ −0.0002 to −0.0005, Δ RMSE ≈ +0.001 to +0.007); `both_unseen` is worse with 2/3-seed
  consistency in most seeds. Only `compound_unseen` is a clean "not worse" axis (and that's
  mechanical, not evidence of benefit — compound is untouched by any E1 switch). That leaves at
  most 1 of 3 required axes clean, short of the `min_axes_not_worse_than_e1a: 2` bar.
- The written `max_tolerated_regression` (RMSE +0.02 / PCC −0.02) is **not breached by any
  variant** — every measured Δ is 3–20× smaller than that band. The gate is written with a wide
  safety margin against catastrophic regression, not as the deciding criterion here; the decisive
  signal is the 3/3-seed-consistent *direction* on `strain_unseen`, which is well above e1a's own
  seed-to-seed noise floor (σ ≈ 0.003–0.005 RMSE, ≈ 0.0002–0.0005 PCC) for e1b/d/e/f specifically,
  and is absent for e1c.
- `plate_axis`: **N/A**, as declared in the gates file — 0 unseen-plate rows exist in the frozen
  validation, so e1e/e1f's actual target mechanism is untested here in either direction.

**G1 verdict: e1c passes; e1b, e1d, e1e, e1f fail.**

## 7. Honest limitations (things not done, or not fully resolved)

1. **Scope narrower than spec 6.6.2.** The full grouped pseudo-OOD harness (nested LOPO-plate /
   LOSO-strain / LOCO-compound / time-block folds refit inside the 5,920 train rows, with
   entity-cluster bootstrap CIs, `scripts/evaluate_grouped_ood.py`, `scripts/gate_e1_unknown_fallback.py`)
   was **not built**. This was a deliberate scope decision to match the user's explicit deliverable
   list (items 1–6), not an oversight, but it means G1 here is a narrower, honestly-labeled stand-in
   using the frozen validation's existing `split_final` categories, not the full spec.
2. **The plate-residual mechanism's actual benefit is untested.** The frozen official validation
   has zero unseen-plate (and zero unseen-instrument/source/medium) rows, so e1e/e1f's designed
   improvement cannot be measured on this split in either direction. It is only verified
   *mathematically correct* (G0, T3/T4/T5), not *empirically beneficial or harmful*.
3. **RNG-consumption confound between e1a and e1b/d/e/f is not fully disentangled.** Constructing
   the zero-freeze `nn.Embedding(..., padding_idx=0)` objects consumes additional torch global
   RNG draws inside `NeutralUnknownHCCEModel.__init__` (before those rows are zeroed), which
   shifts every subsequent dropout mask throughout training relative to `e1a`/`e1c` (which never
   build those extra objects). A cleaner ablation would insert equivalent dummy RNG draws into an
   `e1a`-architecture control to isolate the mechanism effect from this shift. This was not done
   given the time budget; the report states the measured regression as an empirical fact under
   the current architecture, not as a fully mechanism-isolated causal claim, though its
   direction-consistency across three independent seeds argues against pure chance.
4. **Only `strain` is ever unseen in this validation split** among the five non-compound fields.
   `medium`, `source`, and `instrument`'s neutral-zero/one-hot-mask contracts are therefore also
   empirically untested here (verified correct by construction and by T1–T2, never exercised by
   real unseen data in this harness).
5. **G2 (official six-module evaluator, `ΔW ≥ +0.010`) was not run.** Per the task instructions
   this is explicitly left to the user; nothing here should be read as a G2 result of any kind.
6. All 18 training runs completed without error; no result was discarded, rerun, or cherry-picked.

## 8. Recommendation

Do not promote any E1 variant to `runs/final/`. `runs/final/` is untouched and still reproduces
`prediction.csv` sha256 `59f99dc431aa5bd6dc5abb46a5390c64072fda505097cb9523b77198a502b17e`.

- e1c (unknown one-hot masking alone) is mechanically correct, empirically neutral, and could be
  adopted as a low-risk correctness fix independent of this gate process if the team wants the
  contract to literally match the "index 0 means unavailable" claim in the PPT/report language —
  but it produces no measured accuracy benefit on this validation harness, only a documentation/
  correctness argument.
- e1b/e1d/e1e/e1f (anything that zero-freezes the non-compound unknown embedding) measurably
  costs a small amount of `strain_unseen`/`both_unseen` accuracy and should **not** be adopted as
  currently measured. This mirrors the project's own prior finding that strain-directed
  interventions (masking at 0.05/0.10/0.25) regress validation performance — zero-freezing the
  already-untrained strain-unknown embedding is a different mechanism but lands in the same
  outcome bucket.
- If the plate-residual mechanism (e1e/e1f) is still of interest for its own sake (mathematical
  exactness of the unseen-plate fallback, useful for the report's methodological claims), it
  should be evaluated on a validation set that actually contains unseen plates — the frozen
  official validation cannot support that claim in either direction.
