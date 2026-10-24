# Variance-floor equality missed three tiny positive excursions; a twice-floor threshold repaired them

2026-08-24 · experiment · adopted · measured

Three original witnesses had variance minima just above 1e-30 (1.0000000000000003e-30, 1.0000000000000008e-30, and 1.0000000000000004e-30), so an exact equality check missed the floor event. Subsequent correlation normalization amplified tiny denominators into parent-repair excursions 0.00249167098660108, 0.008121398313472243, and 0.010735367118163627. Changing only variance==floor to variance<=2*floor repairs all three trajectories from 7.825e-7/9.667e-6/3.392e-4 to ordinary e-8 values.

Recorded measurement:

    Three original witnesses had variance minima just above 1e-30 (1.0000000000000003e-30, 1.0000000000000008e-30, and 1.0000000000000004e-30), so an exact equality check missed the floor event. Subsequent correlation normalization amplified tiny denominators into parent-repair excursions 0.00249167098660108, 0.008121398313472243, and 0.010735367118163627. Changing only variance==floor to variance<=2*floor repairs all three trajectories from 7.825e-7/9.667e-6/3.392e-4 to ordinary e-8 values.

Program source is in [`code/`](code/), preserving the relative layout below `.`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/estimator.py](code/estimator.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fullbudget_k21_k4_closure_q24_manual_rho_hybrid_floor_v1/estimator.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fullbudget_k21_k4_closure_q24_manual_rho_hybrid_floor_v1/estimator.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fullbudget_k21_k4_closure_q24_manual_rho_hybrid_near_floor_min_v1/estimator.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fullbudget_k21_k4_closure_q24_manual_rho_hybrid_near_floor_min_v1/estimator.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fullbudget_k21_k4_closure_q24_manual_rho_hybrid_near_floor_min_v1/verify_tiny.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fullbudget_k21_k4_closure_q24_manual_rho_hybrid_near_floor_min_v1/verify_tiny.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fullbudget_k21_k4_closure_q24_manual_rho_hybrid_near_floor_v1/estimator.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fullbudget_k21_k4_closure_q24_manual_rho_hybrid_near_floor_v1/estimator.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fullbudget_k21_k4_closure_q24_manual_rho_hybrid_near_floor_v1/verify_tiny.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fullbudget_k21_k4_closure_q24_manual_rho_hybrid_near_floor_v1/verify_tiny.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fullbudget_k21_k4_closure_q24_manual_rho_raw_diag_load_rescale_v1/estimator.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fullbudget_k21_k4_closure_q24_manual_rho_raw_diag_load_rescale_v1/estimator.py)

## Evidence

- [FREEZE_RECEIPT.json#L27](../../flop-accounting/known-arithmetic-reductions-fourth-cumulant-rank-cuts-rejected-measured/FREEZE_RECEIPT.json#L27)
- [FREEZE_RECEIPT.json#L1](../../flop-accounting/twice-floor-variance-repair-legal-meter-2-126-977/FREEZE_RECEIPT.json#L1)

<details>
<summary>All result files and records (2)</summary>

- [FREEZE_RECEIPT.json](../../flop-accounting/known-arithmetic-reductions-fourth-cumulant-rank-cuts-rejected-measured/FREEZE_RECEIPT.json) · result
- [FREEZE_RECEIPT.json](../../flop-accounting/twice-floor-variance-repair-legal-meter-2-126-977/FREEZE_RECEIPT.json) · result

</details>

## Related work

- ← shipped in: [Submission #328277: variance-floor trigger widened to twice the floor, raw 8.758639268080515e-8](../../../submissions/328277/README.md)

[Research graph](../../../graph/index.html#node=2026-08-24-variance-floor-equality-missed-three-tiny-positive-excursions-twice) · [Copy provenance](../../../PROVENANCE.json)

