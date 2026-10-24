# With true fourth-order planes, accumulated skew error remained 4.5–6.5% at layers 6–14

2026-09-04 · experiment · informational · measured

With truth planes the card's per-unit k3 error is 4.4-6.5% (L6-14) = the engine's; forcing truth k3 on the truth-plane host takes r35 2.08e-8 -> 1.085e-8 (r100 2.97e-8 -> 1.93e-8); on the engine-plane host 3.32e-8 -> 2.05e-8. Judge: +0.43 perpendicular to the operator (ceiling 19% of the deployed mean squared error (MSE) on r35). The engine's one-step is exact (0.85%); the 6% is accumulated transport: candidates = second-order Edgeworth (kappa3^2 class, SECOND_ORDER), fourth-order insertions into the distinct completion (TENSOR_K4), the (2,1,1) cum4 feed (-15% of the k3 error, T-11), triangle words.

Recorded measurement:

    With truth planes the card's per-unit k3 error is 4.4-6.5% (L6-14) = the engine's; forcing truth k3 on the truth-plane host takes r35 2.08e-8 -> 1.085e-8 (r100 2.97e-8 -> 1.93e-8); on the engine-plane host 3.32e-8 -> 2.05e-8. Judge: +0.43 perpendicular to the operator (ceiling 19% of the deployed MSE on r35). The engine's one-step is exact (0.85%); the 6% is accumulated transport: candidates = second-order Edgeworth (kappa3^2 class, SECOND_ORDER), fourth-order insertions into the distinct completion (TENSOR_K4), the (2,1,1) cum4 feed (-15% of the k3 error, T-11), triangle words. Engine chain 3 (truth fourth-order + each switch) prices them. T-74/T-74b: the card's k3 given planes has NO scale defect (ratio 1.00 +- 0.005); the residual is unit-specific. SECOND_ORDER -8% rel, TENSOR_K4 0; both third-order feeds together -19% of the plane host (1.70e-8); the residual carry hurts (arena drops). T-85b: the skew-plane error (truth - card K21, 8-13% at depth) has no simple structure (best candidate planes corr <= 0.48: v m^T, v k3^T); fourth-order fix takes it to ~9%, feeds to ~9.4%; the consumer route needs ~3%. T-89/T-89d: the engine's fourth-order plane chain runs in-card (k22_chain.py, RETRO_K22_CHAIN; planes match the engine to corr 0.9999; host 3.09e-8 F2J off, one card run per net); with the feed and the relaxed-arena carry the card's skew is engine-class (per-unit k3 error 3-6% at every layer, no scale bias). The refit on that host removes only 10% (valid-40 3.23e-8): the refit door is closed.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_g29_bond_kernel_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/k22_chain.py](code/k22_chain.py)

## Evidence

- [LANE_g3_tracker.md#L695](../../error-attribution/accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md#L695)
- [engine_ablation_table_r35.txt#L1](engine_ablation_table_r35.txt#L1)

<details>
<summary>All result files and records (2)</summary>

- [LANE_g3_tracker.md](../../error-attribution/accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md) · record
- [engine_ablation_table_r35.txt](engine_ablation_table_r35.txt) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-04-true-fourth-order-planes-accumulated-skew-error-remained-4) · [Copy provenance](../../../PROVENANCE.json)

