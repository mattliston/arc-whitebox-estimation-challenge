# Two-high fourth residual: affine cost gate closed

2026-09-06T11:25Z · experiment · rejected · measured

Small dense algebra passed: projection idempotence/orthogonality, exact pushed fourth-cumulant two-pair plane (K22)/fourth-cumulant triple-single plane (K31)/k4, radial slices, and multivariate batch k-statistics; maximum absolute discrepancy 3.29e-14. This qualifies the mathematics, not a width1024 carrier. No new slices, target access, fit, rank/frame/layer/strength sweep, complete carrier, panel, candidate or submission.

Recorded measurement:

    Small dense algebra passed: projection idempotence/orthogonality, exact pushed K22/K31/k4, radial slices, and multivariate batch k-statistics; maximum absolute discrepancy 3.29e-14. This qualifies the mathematics, not a width1024 carrier. Target-free post12 rank35 frame froze the top four actual covariance eigenvectors; orthogonality 2.00e-15, eigen residual 1.17e-15, fourth/fifth gap .232765. Their positive covariance trace fraction is .148156. The fixed ten-matrix affine model alone costs an additional 300,794,077,498 FLOPs on rank35 and 300,359,880,038 on rank100: 61.31% and 61.35% of current measured bills. The unchanged prerequisite was <=50%. Nonlinear acquisition, projection changes, eigenspace updates, consumer contractions and new third-source carry remain omitted. FAIL before sampling. This is the price of this proposed implementation using the qualified kernel model, not a universal arithmetic lower bound.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_phase2_reexamination_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/certify_two_high_fourth_v1.py](code/certify_two_high_fourth_v1.py)
- [code/freeze_price_two_high_fourth_v1.py](code/freeze_price_two_high_fourth_v1.py)
- [code/two_high_fourth_v1.py](code/two_high_fourth_v1.py)

## Evidence

- [TWO_HIGH_FOURTH_ORACLE_V1_COST_CLOSED.md#L1](TWO_HIGH_FOURTH_ORACLE_V1_COST_CLOSED.md#L1)
- [TWO_HIGH_FOURTH_ORACLE_V1_COST_CLOSED.md#L9](TWO_HIGH_FOURTH_ORACLE_V1_COST_CLOSED.md#L9)
- [TWO_HIGH_FOURTH_ORACLE_V1_COST_CLOSED.md#L13](TWO_HIGH_FOURTH_ORACLE_V1_COST_CLOSED.md#L13)
- [TWO_HIGH_FOURTH_ORACLE_V1_COST_CLOSED.md#L18](TWO_HIGH_FOURTH_ORACLE_V1_COST_CLOSED.md#L18)

<details>
<summary>All result files and records (4)</summary>

- [TWO_HIGH_FOURTH_ORACLE_V1_COST_CLOSED.md](TWO_HIGH_FOURTH_ORACLE_V1_COST_CLOSED.md) · record
- [two_high_fourth_certificate_v1.json](two_high_fourth_certificate_v1.json) · result
- [two_high_fourth_geometry_cost_v1.json](two_high_fourth_geometry_cost_v1.json) · result
- [TWO_HIGH_FOURTH_ORACLE_DESIGN_V1.md](TWO_HIGH_FOURTH_ORACLE_DESIGN_V1.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-06-two-high-fourth-residual-affine-cost-gate-closed) · [Copy provenance](../../../PROVENANCE.json)

