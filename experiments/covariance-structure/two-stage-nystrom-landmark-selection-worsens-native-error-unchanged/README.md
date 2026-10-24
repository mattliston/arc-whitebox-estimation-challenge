# Two-stage Nyström landmark selection worsens native error at unchanged compression ranks

2026-10-01T05:02Z · experiment · rejected · measured

A two-stage landmark rule selects 64 columns by the existing raw Gram diagonal, performs one float64 regularized Schur-residual update and then selects the remaining columns. The Gram weighting, compression ranks 384/320/384 and common rank 256 remain fixed, and the first selected columns are reused. The four-network endpoint adds work and raises adjusted error by 1.265%.

Recorded measurement:

    n=4; raw +1.002703077%; adjusted +1.265388181%; charged work +0.269065066%; raw wins 1/4; adjusted wins 1/4; adjusted 95% interval [-0.11448776888922652, 2.3296187916803213]%

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260926_v2`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/hill_climb_rrpi_join_block64_v1/candidate.py](code/hill_climb_rrpi_join_block64_v1/candidate.py)
- [code/hill_climb_rrpi_join_block64_v1/candidate_identity.py](code/hill_climb_rrpi_join_block64_v1/candidate_identity.py)
- [code/hill_climb_rrpi_join_block64_v1/build.py](code/hill_climb_rrpi_join_block64_v1/build.py)
- [code/hill_climb_pair_rr_inverse_v1/candidate.py](code/hill_climb_pair_rr_inverse_v1/candidate.py)

## Evidence

- [HILL_CLIMB_CURRENT.md#L18](../../../records/research/2026-09-30-adaptive-confirmation-must-preserve-original-failed-gate-uncertainty-claim/HILL_CLIMB_CURRENT.md#L18)
- [HILL_CLIMB_CURRENT.md#L22](../../../records/research/2026-09-30-adaptive-confirmation-must-preserve-original-failed-gate-uncertainty-claim/HILL_CLIMB_CURRENT.md#L22)
- [identity_assessment.json#L35](identity_assessment.json#L35)
- [fit4_assessment.json#L39](fit4_assessment.json#L39)
- [closeout.json#L26](closeout.json#L26)

<details>
<summary>All result files and records (5)</summary>

- [HILL_CLIMB_CURRENT.md](../../../records/research/2026-09-30-adaptive-confirmation-must-preserve-original-failed-gate-uncertainty-claim/HILL_CLIMB_CURRENT.md) · record
- [source_delta.json](source_delta.json) · result
- [identity_assessment.json](identity_assessment.json) · result
- [fit4_assessment.json](fit4_assessment.json) · result
- [closeout.json](closeout.json) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-01-two-stage-nystrom-landmark-selection-worsens-native-error-unchanged) · [Copy provenance](../../../PROVENANCE.json)

