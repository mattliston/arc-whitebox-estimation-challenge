# Full float64 closure gave raw 2.412e-8 versus float32 2.407e-8

2026-09-04 · finding · informational · measured

PRECISION: float64 everywhere (run_retro_f64.py, budget lifted) on r35, deployed closed-loop config: 2.412e-8 vs 2.407e-8 in float32 (1e9 frame). Our card is NOT precision-limited (the organizers' float32 loss is their implementation's). Lever dead.

Recorded measurement:

    - T-131 PRECISION: float64 everywhere (run_retro_f64.py, budget lifted) on r35, deployed closed-loop config: 2.412e-8 vs 2.407e-8 in float32 (1e9 frame). Our card is NOT precision-limited (the organizers' float32 loss is their implementation's). Lever dead.

## Programs

- [run_retro_f64.py](run_retro_f64.py)
- [../2026-09-02-fourth-order-plane-repairs-overlap-learned-residual-correction/estimator_retro.py](../2026-09-02-fourth-order-plane-repairs-overlap-learned-residual-correction/estimator_retro.py)
- [../../../experiments/flop-accounting/tensor-engine-estimator-plane-emissions-agreed-1e-5-same/code/p2_width1024_depth16/public_g20_emission_retrofit_v1/run_retro_screen.py](../../../experiments/flop-accounting/tensor-engine-estimator-plane-emissions-agreed-1e-5-same/code/p2_width1024_depth16/public_g20_emission_retrofit_v1/run_retro_screen.py)
- [../2026-09-02-fourth-order-plane-repairs-overlap-learned-residual-correction/mechfeat15.py](../2026-09-02-fourth-order-plane-repairs-overlap-learned-residual-correction/mechfeat15.py)
- [../2026-09-02-fourth-order-plane-repairs-overlap-learned-residual-correction/k22_chain.py](../2026-09-02-fourth-order-plane-repairs-overlap-learned-residual-correction/k22_chain.py)

## Evidence

- [LANE_g3_tracker.md#L939](../../../experiments/error-attribution/accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md#L939)

<details>
<summary>All result files and records (1)</summary>

- [LANE_g3_tracker.md](../../../experiments/error-attribution/accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-04-full-float64-closure-gave-raw-2-412e-8-versus) · [Copy provenance](../../../PROVENANCE.json)

