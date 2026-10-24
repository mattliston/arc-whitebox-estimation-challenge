# Organizer cumulant propagation at width 1024 exposed precision and memory differences

2026-09-04 · experiment · informational · measured

: kprop K=3 AUGMENT (factor=True, f64 CPU) dies silently at width 1024 (resource_tracker leaked-semaphore = killed, likely OOM on an n^3 augmented diagram); K=4 factorized likewise. Not pursued: the class's accuracy ceiling is secondary to its cost at this width.

Recorded measurement:

    - T-128 THE ORGANIZERS' METHOD IS PUBLIC (found 2026-09-06 via the phase-2 write-up's references): Wu, Lecomte, Winer, Robinson, Hilton, Christiano, "Estimating the expected output of wide random MLPs more efficiently than sampling", arXiv:2605.05179; code github.com/alignment-research-center/mlp_cumulant_propagation (package mlp_kprop: kprop_harmonic.py, factor_k3.py, factor_k4.py, diagslice.py, flop_utils.py). Cumulant propagation ("kprop") to order K with POWER CUMULANTS (Hermite expansions of powers of the activation: the coincident-index correction), an extra full trace for odd K, and a FACTORIZED representation of the K-th cumulant of size n^(K-1) preserved through each step (runtime c L^2 n^K for K>=3): MSE ~ c_K (L/n)^K. At n=256, 4 hidden layers: K=3/4 factorized beat MC by 10-100x; at 8 hidden layers K=4 starts to lose (depth hurts). The seven phase-2 teams at util ~0.2 are almost certainly kprop K=3 (factorized, maybe augmented) ported to flopscope; their raw ~2e-8 at 1024x16. Our CP bank carries every cohort exactly (13 cohorts); kprop MERGES at each nonlinearity via the leading 1/n diagrams -> one n^2 object per order. NEXT: run their torch code on our nets (K=3 factorized / augmented) -> raw vs the 1e9 truth + their FLOP count; then port/beat.
    
    - T-128b: the organizers' kprop run on OUR net 35 (run_kprop_ours.py; their torch code, MPS float32, their FLOP counter with symmetric-tensor discounts): K=2 (Gaussian) final mse 3.02e-6 at 0.033 of 2^41 (= our Gaussian card 2.9e-6; confirms the Gaussian recursion costs ~3%); K=3 factorized SIMPLE: 3.20e-7 at 1.586 of 2^41 (10x better than Gaussian, 10x WORSE than our card's 2.9e-8, and 60% over budget). Vanilla kprop K=3 is NOT the leaders' method (they are 17x more accurate at 1/8 the cost). Our card is a strong implementation of the same physics class. Checking float64 and AUGMENT.
    
    - T-128c PRECISION: the organizers' kprop K=3 factorized in FLOAT64 (CPU, 8 s!): final mse 3.447e-8 on r35 (float32 on MPS gave 3.20e-7: a 10x precision loss). So vanilla kprop K=3 = our card's accuracy class (ours 2.9e-8 board / 2.4e-8 1e9-frame r35) at 1.59 budgets by their symmetric-discounted counter (ours 0.6 by flopscope). K=2 AUGMENT: 2.90e-6 at 0.038 (no gain). The leaders (1.85e-8 at 0.218) are ~8x cheaper than vanilla K=3 and 1.9x better: consistent with the organizers' hint = a RANK-REDUCED K=3 (factors in the top covariance directions) + something for the accuracy. NEXT: kprop's per-op cost breakdown; the rank parameter of its factored K3 representation; MSE vs rank at zero cost.
    
    - T-128d: kprop K=3 AUGMENT (factor=True, f64 CPU) dies silently at width 1024 (resource_tracker leaked-semaphore = killed, likely OOM on an n^3 augmented diagram); K=4 factorized likewise. Not pursued: the class's accuracy ceiling is secondary to its cost at this width.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_g22_operator_features_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/run_kprop_ours.py](code/run_kprop_ours.py)

## Evidence

- [LANE_g3_tracker.md#L931](../../error-attribution/accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md#L931)
- [LANE_g3_tracker.md#L932](../../error-attribution/accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md#L932)
- [LANE_g3_tracker.md#L933](../../error-attribution/accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md#L933)
- [LANE_g3_tracker.md#L948](../../error-attribution/accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md#L948)

<details>
<summary>All result files and records (1)</summary>

- [LANE_g3_tracker.md](../../error-attribution/accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-04-organizer-cumulant-propagation-width-1024-exposed-precision-memory-differences) · [Copy provenance](../../../PROVENANCE.json)

