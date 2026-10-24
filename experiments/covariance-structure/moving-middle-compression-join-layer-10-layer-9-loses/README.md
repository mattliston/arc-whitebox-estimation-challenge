# Moving the middle compression join from layer 10 to layer 9 loses more accuracy than its cost saving

2026-10-01T00:29Z · experiment · rejected · measured

The middle compression moves from layer 10 to layer 9 with matched rank 320, materialization, common-basis approximation and cache admission; the cohort roster and other joins stay fixed. Charged work falls 2.152%, but raw error rises 7.488% and all four adjusted scores worsen.

Recorded measurement:

    n=4; raw +7.487584153%; adjusted +5.175523022%; charged work -2.152112842%; raw wins 0/4; adjusted wins 0/4; adjusted 95% interval [3.6623237510178486, 6.782189376520931]%

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260926_v2/hill_climb_rrpi_middle_join9_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/candidate.py](code/candidate.py)

## Evidence

- [HILL_CLIMB_CURRENT.md#L195](../../../records/research/2026-09-30-adaptive-confirmation-must-preserve-original-failed-gate-uncertainty-claim/HILL_CLIMB_CURRENT.md#L195)
- [HILL_CLIMB_CURRENT.md#L206](../../../records/research/2026-09-30-adaptive-confirmation-must-preserve-original-failed-gate-uncertainty-claim/HILL_CLIMB_CURRENT.md#L206)
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

[Research graph](../../../graph/index.html#node=2026-10-01-moving-middle-compression-join-layer-10-layer-9-loses) · [Copy provenance](../../../PROVENANCE.json)

