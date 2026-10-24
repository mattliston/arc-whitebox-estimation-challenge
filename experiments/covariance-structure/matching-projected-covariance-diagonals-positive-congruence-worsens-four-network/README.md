# Matching projected covariance diagonals by positive congruence worsens the four-network screen

2026-10-01T05:12Z · experiment · rejected · measured

At joins 7 and 10, a diagonal congruence D K D rescales the existing covariance core so its diagonal matches the exact projected diagonal, with Dii equal to the square root of target diagonal divided by current diagonal. The implementation keeps the Gram construction, basis and ranks fixed and adds explicit finite-positive checks before returning float32 state. Identity parity passes, but the enabled endpoint increases raw and adjusted error.

Recorded measurement:

    n=4; raw +0.140158191%; adjusted +0.227378450%; charged work +0.086058917%; raw wins 2/4; adjusted wins 1/4; adjusted 95% interval [-0.007767834559035958, 0.4413191917373549]%

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260926_v2`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/hill_climb_rrpi_core_diag_match_finite_v1/candidate.py](code/hill_climb_rrpi_core_diag_match_finite_v1/candidate.py)
- [code/hill_climb_rrpi_core_diag_match_finite_v1/candidate_identity.py](code/hill_climb_rrpi_core_diag_match_finite_v1/candidate_identity.py)
- [code/hill_climb_rrpi_core_diag_match_finite_v1/build.py](code/hill_climb_rrpi_core_diag_match_finite_v1/build.py)
- [code/hill_climb_pair_rr_inverse_v1/candidate.py](code/hill_climb_pair_rr_inverse_v1/candidate.py)
- [code/hill_climb_rrpi_core_diag_match_v1/build.py](code/hill_climb_rrpi_core_diag_match_v1/build.py)
- [code/hill_climb_rrpi_core_diag_match_v1/candidate.py](code/hill_climb_rrpi_core_diag_match_v1/candidate.py)
- [code/hill_climb_rrpi_core_diag_match_v1/candidate_identity.py](code/hill_climb_rrpi_core_diag_match_v1/candidate_identity.py)

## Evidence

- [HILL_CLIMB_CURRENT.md#L10](../../../records/research/2026-09-30-adaptive-confirmation-must-preserve-original-failed-gate-uncertainty-claim/HILL_CLIMB_CURRENT.md#L10)
- [HILL_CLIMB_CURRENT.md#L14](../../../records/research/2026-09-30-adaptive-confirmation-must-preserve-original-failed-gate-uncertainty-claim/HILL_CLIMB_CURRENT.md#L14)
- [HILL_CLIMB_CURRENT.md#L19](../../../records/research/2026-09-30-adaptive-confirmation-must-preserve-original-failed-gate-uncertainty-claim/HILL_CLIMB_CURRENT.md#L19)
- [identity_assessment.json#L35](identity_assessment.json#L35)
- [fit4_assessment.json#L39](fit4_assessment.json#L39)
- [closeout.json#L26](closeout.json#L26)

<details>
<summary>All result files and records (6)</summary>

- [HILL_CLIMB_CURRENT.md](../../../records/research/2026-09-30-adaptive-confirmation-must-preserve-original-failed-gate-uncertainty-claim/HILL_CLIMB_CURRENT.md) · record
- [source_delta.json](source_delta.json) · result
- [source_review.json](source_review.json) · result
- [identity_assessment.json](identity_assessment.json) · result
- [fit4_assessment.json](fit4_assessment.json) · result
- [closeout.json](closeout.json) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-01-matching-projected-covariance-diagonals-positive-congruence-worsens-four-network) · [Copy provenance](../../../PROVENANCE.json)

