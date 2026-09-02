# P1 Optimization Levers — 2026-08-26

## Freeze record

- `configs/p1_gates.json` SHA-256: `5651e83c52d7fd65a3965a478bb8f2dbf81e09904bca87e849c825e3995fa399`
- Gate-W frozen before experiment-data access. Required additional condition: `ΔRMSE <= -0.001`.

## Frozen candidate manifest

- `configs/p1_candidates.json` SHA-256: `63d7dc5fe82fa1370160c5c6e6d51a03bd5061974f64a8f45210ecdd407936b5`
- Official evaluation command:
  `python scripts/evaluate_official_modules.py --candidates configs/p1_candidates.json --baseline mask_compound_3seed_base --out outputs/p1_official_plate --workers 4 --bootstrap 2000`
- Official output: `outputs/p1_official_plate/module_metrics.json`
- Test truth was not read. Evaluator was not edited.

## Execution record

All HCCE runs used `mask_compound`, 40 epochs, embedding dimension 64, `mask_p=0.25`, train-only fitting, and validation-only model selection. SWA used a running state-dict average over the final 5 or 10 epochs and did not retain full per-epoch checkpoints.

- R1-A: 6 runs, `N in {5,10}` × seeds `{20260810,3407,42}`, GPUs 0–3, output `outputs/p1_swa/`; each N has a 3-seed 0.5 legacy + 0.5 HCCE ensemble parquet.
- R1-B: 10 runs, seeds `{20260810,3407,42,7,2024,31337,1234,5678,91011,20260903}`, output `outputs/p1_multiseed/`; top-3 were selected by validation RMSE only (`20260810,3407,1234`), and equal-10 was also exported.
- R1-C: 3 RAP-R4 stable rechecks, seeds `{20260810,3407,42}`, GPUs 0, 2, 3, output `outputs/p1_rap_r4_recheck/`; current mask-compound 3-seed validation parquet was used as fallback.

## Official results and Gate-W adjudication

The baseline is `mask_compound_3seed_base`: RMSE `0.420258878`, W `0.521834`.

> **Caliber correction, 2026-08-26.** An earlier version of this table reported baseline
> W = `0.525928043`. The frozen formula in `configs/p1_gates.json` defines the 10% term as
> `0.10*(0.5*timeFC + 0.5*S3FC)`, but the official evaluator emits **no** separate `val_time`
> module — `val_time` rows have both entities present in train and are stratified as `seen` —
> so no timeFC value exists to put in that slot. The earlier run filled the gap with another
> FC stratum, which is exactly the kind of substitution the P0.5 report refused to make.
> The table below is recomputed by `scripts/gate_w_summary.py` with the 10% term filled by
> **S3 FC alone**, the substitution recorded explicitly in
> `outputs/p1_official_plate/gate_w_summary.json`, and no time-FC number invented. This makes
> the P1 and P0.5 reports use one caliber; the baseline W of `0.521834` is identical across both.
> **No verdict changes**: every candidate was and remains NOT ADOPTED.

Here W uses official-plate S1, official evaluator FC strata, S2, S3 FC in the 10% slot, and DEP direction accuracy. `ΔW_noS2` removes the S2 term and renormalises over the remaining 0.80 weight. Point estimates below are relative to the baseline; the official evaluator's paired 2,000-bootstrap S1 intervals are in `module_metrics.json`.

| Candidate | RMSE | ΔRMSE | ΔW | ΔW_noS2 | Gate-W/RMSE decision |
|---|---:|---:|---:|---:|---|
| R1A SWA N=5 | 0.419465016 | -0.000793863 | +0.000718 | +0.000664 | NOT ADOPTED: RMSE gate fails; ΔW < +0.010 |
| R1A SWA N=10 | 0.418977261 | -0.001281618 | +0.000531 | +0.000522 | NOT ADOPTED: ΔW < +0.010 |
| R1B val-RMSE top-3 | 0.418924314 | -0.001334564 | +0.001441 | +0.001946 | NOT ADOPTED: ΔW < +0.010 |
| R1B equal-10 | 0.421670534 | +0.001411656 | +0.002089 | +0.003829 | NOT ADOPTED: RMSE gate fails; ΔW < +0.010 |
| R1C RAP-R4 seed 20260810 | 0.420819117 | +0.000560239 | +0.007260 | +0.009074 | NOT ADOPTED: RMSE gate fails; ΔW < +0.010 |
| R1C RAP-R4 seed 3407 | 0.420744990 | +0.000486112 | +0.007482 | +0.009352 | NOT ADOPTED: RMSE gate fails; ΔW < +0.010 |
| R1C RAP-R4 seed 42 | 0.420851306 | +0.000592428 | +0.007256 | +0.009069 | NOT ADOPTED: RMSE gate fails; ΔW < +0.010 |

No candidate reaches the frozen `ΔW >= +0.010` requirement. Consequently none can pass Gate-W, regardless of the confidence-interval condition. The best RMSE is R1B top-3, but its W gain is only `+0.001440`; the best W is RAP-R4 seed 3407 at `+0.007482`, while its RMSE worsens by `+0.000486`. Worth recording separately: the RAP-R4 runs give ΔS1 = `+0.032` to `+0.036` (0.235399 / 0.236028 / 0.235444 against the baseline's 0.203387), the largest S1 improvement of any candidate tested to date — larger than R1 convex mixing's `+0.02191`. It still sits below the frozen 0.053 effect floor, its RMSE worsens, and its S2 and S3 values are bitwise identical to the baseline because the mask-compound parquet was used as a fallback on those rows, so it is a partial hybrid rather than a complete candidate. Recorded, not adopted. No candidate passes both Gate-W (including ΔW_noS2) and the RMSE gate. The submission package remains unchanged.

The official-plate paired S1 bootstrap intervals were: SWA N=5 `[−0.002427, +0.008761]`, SWA N=10 `[−0.002290, +0.008126]`, top-3 `[−0.000165, +0.012914]`, equal-10 `[+0.006554, +0.029579]`, and RAP seeds 20260810/3407/42 `[−0.009207, +0.078545]` / `[−0.007640, +0.078020]` / `[−0.009363, +0.078916]`. These are S1 intervals; the pointwise W-floor failure alone is decisive for every candidate, and equal-10 additionally fails RMSE.

## Compact metric record

| Candidate | abs sample-PCC | FC all | S1 official-plate | S2 | time FC | S3 FC | DEP |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | 0.988598 | 0.491355 | 0.203387 | 0.407705 | 0.426484 | 0.344605 | 0.891936 |
| SWA N=5 | 0.988631 | 0.491635 | 0.205744 | 0.408641 | 0.426874 | 0.344755 | 0.891293 |
| SWA N=10 | 0.988648 | 0.490850 | 0.205433 | 0.408273 | 0.426297 | 0.346290 | 0.891061 |
| top-3 | 0.988663 | 0.493073 | 0.208016 | 0.407123 | 0.428572 | 0.346708 | 0.891506 |
| equal-10 | 0.988554 | 0.492232 | 0.218980 | 0.402832 | 0.423744 | 0.342234 | 0.891360 |
| RAP seed 20260810 | 0.988567 | 0.494618 | 0.235399 | 0.407705 | 0.426484 | 0.344605 | 0.892890 |
| RAP seed 3407 | 0.988571 | 0.494884 | 0.236028 | 0.407705 | 0.426484 | 0.344605 | 0.893470 |
| RAP seed 42 | 0.988566 | 0.494604 | 0.235444 | 0.407705 | 0.426484 | 0.344605 | 0.892699 |

## Final P1 conclusion

P1 is a preregistered negative result for adoption: SWA and validation-RMSE seed selection provide small RMSE improvements, but not the required W improvement; RAP-R4 improves W but misses both the W floor and RMSE gate. Keep `mask_compound_3seed_base` as the active line and proceed to P2 without package mutation.
