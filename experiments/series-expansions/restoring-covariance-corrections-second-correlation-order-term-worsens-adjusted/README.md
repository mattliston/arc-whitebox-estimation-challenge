# Restoring the covariance correction’s second correlation-order term worsens adjusted error by 0.214%

2026-10-01T00:29Z · experiment · rejected · measured

The existing covariance increment, denoted GN in the source, regains only its historical rho²/2 terms multiplying fourth- or fifth-order left derivatives and a third-order right derivative. The main pair map retains its first-order non-Gaussian expansion; this is a deliberate partial restoration, not correction of an accidental omission. Independent algebra passes, but the four-network native endpoint fails both error gates.

Recorded measurement:

    n=4; raw +0.182670399%; adjusted +0.214003500%; charged work +0.031106841%; raw wins 2/4; adjusted wins 2/4; adjusted 95% interval [-0.23769673965310378, 0.6936042979039492]%

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260926_v2`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/hill_climb_rrpi_gn_r2_restore_v1/candidate.py](code/hill_climb_rrpi_gn_r2_restore_v1/candidate.py)
- [code/hill_climb_rrpi_gn_r2_restore_v1/build.py](code/hill_climb_rrpi_gn_r2_restore_v1/build.py)
- [code/hill_climb_pair_rr_inverse_v1/candidate.py](code/hill_climb_pair_rr_inverse_v1/candidate.py)

## Evidence

- [HILL_CLIMB_CURRENT.md#L196](../../../records/research/2026-09-30-adaptive-confirmation-must-preserve-original-failed-gate-uncertainty-claim/HILL_CLIMB_CURRENT.md#L196)
- [HILL_CLIMB_CURRENT.md#L202](../../../records/research/2026-09-30-adaptive-confirmation-must-preserve-original-failed-gate-uncertainty-claim/HILL_CLIMB_CURRENT.md#L202)
- [algebra_result.json#L1](algebra_result.json#L1)
- [fit4_assessment.json#L42](fit4_assessment.json#L42)
- [closeout.md#L1](closeout.md#L1)

<details>
<summary>All result files and records (5)</summary>

- [HILL_CLIMB_CURRENT.md](../../../records/research/2026-09-30-adaptive-confirmation-must-preserve-original-failed-gate-uncertainty-claim/HILL_CLIMB_CURRENT.md) · record
- [source_delta.json](source_delta.json) · result
- [algebra_result.json](algebra_result.json) · result
- [fit4_assessment.json](fit4_assessment.json) · result
- [closeout.md](closeout.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-01-restoring-covariance-corrections-second-correlation-order-term-worsens-adjusted) · [Copy provenance](../../../PROVENANCE.json)

