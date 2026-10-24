# Diagonal third-cumulant closure failed despite a trained correction

2026-09-04 · experiment · rejected · measured

DIAGONAL CARD DEAD: the oracle one-step host is 2.87e-4 (13,000x the full card) - the off-diagonal third-cumulant repeated-index plane (K21) plane terms are essential to the covariance recursion itself; the closure removes 91% teacher-forced (2.6e-5) but the loop is 1.2e-6 (r35) / 1.4e-4 (r100, diverged) at util 0.21-0.24. FAMILY MAP COMPLETE: the closure replaces the fourth-order machinery (wash to -6% adjusted); no part of the third-order machinery (bank, cubic push, off-diagonal plane) can be removed or capped. The day verdict stands.

Recorded measurement:

    - T-120 DIAGONAL CARD (Gaussian covariance recursion + diagonal third-order only: K21 diag-only, CP3 cap 1024, CP4 off, no quartic; cubic push kept): util 0.233 (cap 2048: 0.274), plain host 2.54e-6 (30x the full card's 8.6e-8). Bar at util 0.233 needs raw < ~4.3e-8 grader -> the closure would have to remove ~98% of the host error in the loop (the stack card got 88% teacher-forced, 3e-7 loop). The missing terms are a known bilinear structure (the K21 plane's effect) - a definite test of whether the closure can learn the bank's function from the weight-graph messages. Co-adaptation round launched (chain_diagcard.sh: rows -> GNN_HDIAG (init HNQ) -> loops with util -> gate).
    
    - T-120 the diagonal card's rows crashed in mechfeat15 (non-finite plane-derived columns on a zeroed off-diagonal plane); guarded with nan_to_num in mechfeat15 and an fnp.where(isfinite) in _mech_fnp (both paths), round relaunched (rows resume from the 21 done).
    
    - T-120 DIAGONAL CARD DEAD: the oracle one-step host is 2.87e-4 (13,000x the full card) - the off-diagonal K21 plane terms are essential to the covariance recursion itself; the closure removes 91% teacher-forced (2.6e-5) but the loop is 1.2e-6 (r35) / 1.4e-4 (r100, diverged) at util 0.21-0.24. FAMILY MAP COMPLETE: the closure replaces the fourth-order machinery (wash to -6% adjusted); no part of the third-order machinery (bank, cubic push, off-diagonal plane) can be removed or capped. The day verdict stands. Machine idle.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/public_g22_operator_features_v1/chain_diagcard.sh](code/public_g22_operator_features_v1/chain_diagcard.sh)
- [code/public_g20_emission_retrofit_v1/estimator_retro.py](code/public_g20_emission_retrofit_v1/estimator_retro.py)
- [code/public_g22_operator_features_v1/export_gnn_payload.py](code/public_g22_operator_features_v1/export_gnn_payload.py)
- [code/public_g22_operator_features_v1/gen_rows_meanleak_mech15_cap.py](code/public_g22_operator_features_v1/gen_rows_meanleak_mech15_cap.py)
- [code/public_g22_operator_features_v1/score_vs_truthpost.py](code/public_g22_operator_features_v1/score_vs_truthpost.py)
- [code/public_g22_operator_features_v1/train_gnn_oracle.py](code/public_g22_operator_features_v1/train_gnn_oracle.py)
- [code/public_g28_mech_features_v1/mechfeat15.py](code/public_g28_mech_features_v1/mechfeat15.py)
- [code/public_g29_bond_kernel_v1/k22_chain.py](code/public_g29_bond_kernel_v1/k22_chain.py)

## Evidence

- [LANE_g3_tracker.md#L905](../accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md#L905)
- [LANE_g3_tracker.md#L906](../accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md#L906)
- [LANE_g3_tracker.md#L907](../accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md#L907)

<details>
<summary>All result files and records (1)</summary>

- [LANE_g3_tracker.md](../accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-04-diagonal-third-cumulant-closure-failed-despite-trained-correction) · [Copy provenance](../../../PROVENANCE.json)

