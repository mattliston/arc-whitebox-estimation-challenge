# Delaying the middle compression join to layer 11 adds cost without a useful accuracy gain

2026-10-01T01:56Z · experiment · rejected · measured

Moving only the middle join from layer 10 to layer 11 preserves rank 320 and the other two joins, with corresponding materialization and cache changes. The four-network raw-error reduction is only 0.031%, while charged work rises 2.306%, so every adjusted score worsens.

Recorded measurement:

    n=4; raw -0.031239082%; adjusted +2.275470486%; charged work +2.306056835%; raw wins 3/4; adjusted wins 0/4; adjusted 95% interval [1.350960455446315, 3.727706782263729]%

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260926_v2/hill_climb_rrpi_middle_join11_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/candidate.py](code/candidate.py)

## Evidence

- [HILL_CLIMB_CURRENT.md#L150](../../../records/research/2026-09-30-adaptive-confirmation-must-preserve-original-failed-gate-uncertainty-claim/HILL_CLIMB_CURRENT.md#L150)
- [HILL_CLIMB_CURRENT.md#L155](../../../records/research/2026-09-30-adaptive-confirmation-must-preserve-original-failed-gate-uncertainty-claim/HILL_CLIMB_CURRENT.md#L155)
- [fit4_assessment.json#L42](fit4_assessment.json#L42)
- [closeout.md#L1](closeout.md#L1)

<details>
<summary>All result files and records (5)</summary>

- [HILL_CLIMB_CURRENT.md](../../../records/research/2026-09-30-adaptive-confirmation-must-preserve-original-failed-gate-uncertainty-claim/HILL_CLIMB_CURRENT.md) · record
- [source_delta.json](source_delta.json) · result
- [fit4_assessment.json](fit4_assessment.json) · result
- [closeout.md](closeout.md) · record
- [native_fit4_guard.json](native_fit4_guard.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-10-01-delaying-middle-compression-join-layer-11-adds-cost-without) · [Copy provenance](../../../PROVENANCE.json)

