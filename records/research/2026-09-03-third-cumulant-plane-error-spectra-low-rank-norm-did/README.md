# Third-cumulant plane-error spectra: low rank in norm did not preserve correction value

2026-09-03 · finding · informational · measured

SKEW-PLANE ACCURACY LADDER on the truth-plane host (r35; third-cumulant repeated-index plane (K21) off-diagonal rel err / k3 rel err vs truth at L6/8/10/12/14): card 0.086/0.061, 0.094/0.073, 0.110/0.091, 0.125/0.108, 0.133/0.111; truth fourth-order planes (tp) 0.076/0.044 ... 0.105/0.065; tp + cum4 feed 0.083/0.056 ... 0.107/0.067 (the cum4 feed slightly WORSENS the K21 plane while cutting the host -8%); tp + plane-to-distinct feed 0.072/0.041 ... 0.094/0.059 (host 1.803e-8); tp + both 0.079/0.053 ... 0.096/0.061 (host 1.697e-8); tp + residual carry (window 1-6, default arena) 0.059/0.032 at L6 but 0.167/0.160 at L14 (host 4.30e-8: the arena schedule drops at depth). Nothing on record takes the deep K21 plane below ~9%; the 2e-8 route via the pair consumer needs ~3%. Consumer dependence on fourth-order planes: truth K21 with the CARD's fourth-cumulant two-pair plane (K22)/fourth-cumulant triple-single plane (K31) inside the consumer -> beta 0.5 -20.6% becomes -5.8% (2.548e-8); the 'true' file (truth states) at beta 0.5: 2.339e-8.

Recorded measurement:

    - T-81 THE SKEW-PLANE ERROR IS LOW-RANK AT DEPTH (r35, truth K21 minus the card's pre-plane, off-diagonal): energy in rank 1 / 8 / 32 / 128 = 0.26 / 0.40 / 0.59 / 0.86 at L6, 0.44 / 0.62 / 0.79 / 0.94 at L10, 0.51 / 0.74 / 0.88 / 0.97 at L14 (the truth plane itself: rank 8 holds 0.82-0.90); the coherent scale part is small (truth-on-card slope 1.02-1.04; rel err 0.133 -> 0.127 after the scale). With truth fourth-order planes the remaining K21 error is less low-rank (rank 8: 0.42 / 0.59 at L10 / L14): the low-rank part of the K21 error is largely the fourth-order plane defect's shadow. Pair-consumer variance corrections from TRUTH K21 planes (G21 pc_* files, oracle) injected at half scale with the operator ON: r35 pc_lin_pairs 2.148e-8 (-20.6%), pc_edge_pairs 2.149e-8, pc_exact_pairs 2.382e-8; r100 pc_exact_pairs 2.062e-8 (-25.8%), pc_lin_pairs 2.341e-8, shorcall 2.135e-8; the engine-plane-fed 'dep' file: worse (3.75e-8). => truth-class K21 planes through the linear pair consumer at half scale ~ the 2e-8 line on both nets; testing whether a LOW-RANK (1 / 8 / 32) K21 correction carries that value.
    
    - T-81b THE K21 CORRECTION MUST BE HIGH-RANK (r35, pair consumer 'lin' mode with PAIRS=1 fed the card's K21 plus a rank-R cut of the TRUE K21 error; run_pair_consumer.py SUPPLY_K21/OUT_TAG; dv injected at beta 0.5, operator ON): full 2.148e-8 (-20.6%; dv corr 0.92-0.98 = reproduces pc_lin_pairs), rank 32 2.284e-8 (-15.5%; corr 0.77-0.91), rank 8 2.566e-8 (-5.1%; corr 0.69-0.86), rank 1 2.460e-8 (-9.0%; corr 0.62-0.81). So the 2e-8 line needs the skew plane accurate to a few percent in >= 32 directions per layer; the card's K21 planes are 8-13% off (7.7-10.7% with truth fourth-order planes) and engine-class planes fail the alignment bar. ANSWER TO MATT (2e-8): reachable in principle via truth-class skew planes -> linear pair consumer -> half-scale variance correction with the operator kept (2.15e-8 / 2.06e-8 measured); no deployable skew-plane supply at that accuracy exists; the certain move is the f32 pack (-4.9% adj).
    
    - T-81c SKEW-PLANE ACCURACY LADDER on the truth-plane host (r35; K21 off-diagonal rel err / k3 rel err vs truth at L6/8/10/12/14): card 0.086/0.061, 0.094/0.073, 0.110/0.091, 0.125/0.108, 0.133/0.111; truth fourth-order planes (tp) 0.076/0.044 ... 0.105/0.065; tp + cum4 feed 0.083/0.056 ... 0.107/0.067 (the cum4 feed slightly WORSENS the K21 plane while cutting the host -8%); tp + plane-to-distinct feed 0.072/0.041 ... 0.094/0.059 (host 1.803e-8); tp + both 0.079/0.053 ... 0.096/0.061 (host 1.697e-8); tp + residual carry (window 1-6, default arena) 0.059/0.032 at L6 but 0.167/0.160 at L14 (host 4.30e-8: the arena schedule drops at depth). Nothing on record takes the deep K21 plane below ~9%; the 2e-8 route via the pair consumer needs ~3%. Consumer dependence on fourth-order planes: truth K21 with the CARD's K22/K31 inside the consumer -> beta 0.5 -20.6% becomes -5.8% (2.548e-8); the 'true' file (truth states) at beta 0.5: 2.339e-8. ANSWER TO MATT (lowest deployable MSE): the deployed card (2.70 / 2.78e-8 offline; 2.94e-8 board); f32 pack identical raw at -4.9% util; the feed -9.2% on r35 / +1.4% on r100 (32M frame).

## Programs

- [run_pair_consumer.py](run_pair_consumer.py)
- [../../../experiments/fourth-order-terms/cross-unit-plane-functionals-failed-predict-deployed-model-residual/code/public_g19_attribution_v1/run_attribution.py](../../../experiments/fourth-order-terms/cross-unit-plane-functionals-failed-predict-deployed-model-residual/code/public_g19_attribution_v1/run_attribution.py)
- [../../../experiments/flop-accounting/self-carried-fourth-order-planes-integrated-into-width-1024/code/p2_width1024_depth16/public_g19_attribution_v1/run_collective_patch.py](../../../experiments/flop-accounting/self-carried-fourth-order-planes-integrated-into-width-1024/code/p2_width1024_depth16/public_g19_attribution_v1/run_collective_patch.py)
- [../../../experiments/error-attribution/collective-aware-tangent-added-only-0-3-0-2/code/public_g19_attribution_v1/run_tangent_v2.py](../../../experiments/error-attribution/collective-aware-tangent-added-only-0-3-0-2/code/public_g19_attribution_v1/run_tangent_v2.py)
- [../../../experiments/flop-accounting/self-carried-fourth-order-planes-integrated-into-width-1024/code/p2_width1024_depth16/public_g19_attribution_v1/run_tensor_recursion_v4.py](../../../experiments/flop-accounting/self-carried-fourth-order-planes-integrated-into-width-1024/code/p2_width1024_depth16/public_g19_attribution_v1/run_tensor_recursion_v4.py)
- [../../../experiments/learned-correction/newly-generated-fourth-order-block-path-feed-had-negligible/code/p2_width1024_depth16/public_g21_pair_consumer_v1/pairmap.py](../../../experiments/learned-correction/newly-generated-fourth-order-block-path-feed-had-negligible/code/p2_width1024_depth16/public_g21_pair_consumer_v1/pairmap.py)
- [../../../submissions/328326/estimator.py](../../../submissions/328326/estimator.py)
- [../../../experiments/flop-accounting/self-carried-fourth-order-planes-integrated-into-width-1024/code/x3_r0c1_kappa3_ceiling/tensor_crosscheck_v1/second_order_triple.py](../../../experiments/flop-accounting/self-carried-fourth-order-planes-integrated-into-width-1024/code/x3_r0c1_kappa3_ceiling/tensor_crosscheck_v1/second_order_triple.py)

## Evidence

- [LANE_g3_tracker.md#L721](../../../experiments/error-attribution/accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md#L721)
- [LANE_g3_tracker.md#L722](../../../experiments/error-attribution/accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md#L722)
- [LANE_g3_tracker.md#L723](../../../experiments/error-attribution/accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md#L723)

<details>
<summary>All result files and records (1)</summary>

- [LANE_g3_tracker.md](../../../experiments/error-attribution/accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-03-third-cumulant-plane-error-spectra-low-rank-norm-did) · [Copy provenance](../../../PROVENANCE.json)

