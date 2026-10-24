# Covariance-subspace projection tested inside the organizer third-order recursion

2026-09-04 · experiment · informational · measured

(run_kprop_rank.py, zero billed): the organizers' hint as a SUBSPACE PROJECTION inside their own K=3 factorized f64 recursion (all-distinct factors projected onto the top-k covariance eigenvectors after every nonlinear step; repeated slices exact): k=1024 3.45e-8 (identity), 512 2.29e-7, 256 7.31e-7, 128 1.41e-6. The all-distinct third-order content outside the top-512 directions is worth 7x in mean squared error (MSE) through the recursion. Their factor bank: 2048 factors born per layer, 32k by the end (a cohort bank like ours).

Recorded measurement:

    - T-129 (run_kprop_rank.py, zero billed): the organizers' hint as a SUBSPACE PROJECTION inside their own K=3 factorized f64 recursion (all-distinct factors projected onto the top-k covariance eigenvectors after every nonlinear step; repeated slices exact): k=1024 3.45e-8 (identity), 512 2.29e-7, 256 7.31e-7, 128 1.41e-6. The all-distinct third-order content outside the top-512 directions is worth 7x in MSE through the recursion. Their factor bank: 2048 factors born per layer, 32k by the end (a cohort bank like ours). Rank reduction of the bank is NOT the leaders' trick (consistent with T-95..97 receipts). The leaders' 1.85e-8 at 0.218 remains undecoded from the public record. Next zero-cost reads: K=3 AUGMENT f64 and K=4 factorized f64 (accuracy ceiling of the class regardless of cost).

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_g22_operator_features_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/run_kprop_rank.py](code/run_kprop_rank.py)

## Evidence

- [LANE_g3_tracker.md#L934](../../error-attribution/accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md#L934)

<details>
<summary>All result files and records (1)</summary>

- [LANE_g3_tracker.md](../../error-attribution/accumulated-tracker-readout-reached-0-346-alone-0-811/LANE_g3_tracker.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-04-covariance-subspace-projection-tested-inside-organizer-third-order-recursion) · [Copy provenance](../../../PROVENANCE.json)

