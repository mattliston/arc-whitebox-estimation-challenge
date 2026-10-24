# Separating marginal kurtosis updates from fourth-order pair suppliers still worsens native error

2026-10-01T04:50Z · experiment · rejected · measured

The row-sum and diagonal kurtosis correction is retained in per-unit marginal moments, features and recurrence, while the pair-map fourth-order supplier and covariance correction read the saved pre-addition kurtosis. This tests whether the prior failure came from coupling the new marginal value into the approximate (3,1) fourth-order plane. The fixed-strength, full-scope native screen still worsens raw and adjusted error.

Recorded measurement:

    n=4; raw +0.409898714%; adjusted +0.599728352%; charged work +0.184611211%; raw wins 1/4; adjusted wins 1/4; adjusted 95% interval [-0.39370659081577886, 1.5879277358971766]%

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260926_v2`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/hill_climb_rrpi_k4_marginal_only_v1/candidate.py](code/hill_climb_rrpi_k4_marginal_only_v1/candidate.py)
- [code/hill_climb_rrpi_k4_marginal_only_v1/candidate_zero.py](code/hill_climb_rrpi_k4_marginal_only_v1/candidate_zero.py)
- [code/hill_climb_rrpi_k4_marginal_only_v1/build.py](code/hill_climb_rrpi_k4_marginal_only_v1/build.py)
- [code/hill_climb_pair_rr_inverse_v1/candidate.py](code/hill_climb_pair_rr_inverse_v1/candidate.py)
- [code/hill_climb_rrpi_k4_rowdiag_v1/candidate.py](code/hill_climb_rrpi_k4_rowdiag_v1/candidate.py)

## Evidence

- [HILL_CLIMB_CURRENT.md#L25](../../../records/research/2026-09-30-adaptive-confirmation-must-preserve-original-failed-gate-uncertainty-claim/HILL_CLIMB_CURRENT.md#L25)
- [HILL_CLIMB_CURRENT.md#L30](../../../records/research/2026-09-30-adaptive-confirmation-must-preserve-original-failed-gate-uncertainty-claim/HILL_CLIMB_CURRENT.md#L30)
- [HILL_CLIMB_CURRENT.md#L31](../../../records/research/2026-09-30-adaptive-confirmation-must-preserve-original-failed-gate-uncertainty-claim/HILL_CLIMB_CURRENT.md#L31)
- [identity_assessment.json#L35](identity_assessment.json#L35)
- [fit4_assessment.json#L39](fit4_assessment.json#L39)
- [closeout.json#L48](closeout.json#L48)

<details>
<summary>All result files and records (5)</summary>

- [HILL_CLIMB_CURRENT.md](../../../records/research/2026-09-30-adaptive-confirmation-must-preserve-original-failed-gate-uncertainty-claim/HILL_CLIMB_CURRENT.md) · record
- [source_delta.json](source_delta.json) · result
- [identity_assessment.json](identity_assessment.json) · result
- [fit4_assessment.json](fit4_assessment.json) · result
- [closeout.json](closeout.json) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-01-separating-marginal-kurtosis-updates-fourth-order-pair-suppliers-still) · [Copy provenance](../../../PROVENANCE.json)

