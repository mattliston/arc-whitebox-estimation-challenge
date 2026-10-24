# Projecting the covariance increment to annihilate the current mean worsens all four screened networks

2026-10-01T02:07Z · experiment · rejected · measured

At layers 10–12, the covariance increment denoted GN is projected onto the symmetric zero-diagonal subspace that annihilates the current pair-map mean. The host mean is taken before the later learned update, and scalar coefficients remain unchanged. The constrained-projection algebra agrees to approximately 5e-16, but raw and adjusted errors worsen on all four native networks.

Recorded measurement:

    n=4; raw +0.337642522%; adjusted +0.361194099%; charged work +0.023970399%; raw wins 0/4; adjusted wins 0/4; adjusted 95% interval [0.29363898291769885, 0.43210060365432845]%; maximum constrained least-squares algebra discrepancy 4.996e-16

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260926_v2/hill_climb_rrpi_gn_perp_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/fragment.py](code/fragment.py)
- [code/candidate.py](code/candidate.py)
- [code/algebra.py](code/algebra.py)

## Evidence

- [HILL_CLIMB_CURRENT.md#L141](../../../records/research/2026-09-30-adaptive-confirmation-must-preserve-original-failed-gate-uncertainty-claim/HILL_CLIMB_CURRENT.md#L141)
- [HILL_CLIMB_CURRENT.md#L145](../../../records/research/2026-09-30-adaptive-confirmation-must-preserve-original-failed-gate-uncertainty-claim/HILL_CLIMB_CURRENT.md#L145)
- [algebra_result.json#L1](algebra_result.json#L1)
- [fit4_assessment.json#L42](fit4_assessment.json#L42)
- [closeout.json#L17](closeout.json#L17)

<details>
<summary>All result files and records (4)</summary>

- [HILL_CLIMB_CURRENT.md](../../../records/research/2026-09-30-adaptive-confirmation-must-preserve-original-failed-gate-uncertainty-claim/HILL_CLIMB_CURRENT.md) · record
- [algebra_result.json](algebra_result.json) · result
- [fit4_assessment.json](fit4_assessment.json) · result
- [closeout.json](closeout.json) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-01-projecting-covariance-increment-annihilate-current-mean-worsens-all-four) · [Copy provenance](../../../PROVENANCE.json)

