# Angular input-law propagation with unchanged learned parameters worsens all four native networks

2026-10-01T03:33Z · experiment · rejected · measured

The angular input-law variant sets the initial radial correction to −6/(n+2), enables the existing second-order Edgeworth block in the layer-zero Gaussian map and scales all returned means by 0.9997558892136928. Learned parameters and the remaining propagation are unchanged. Identity checks pass, but the physical endpoint loses on every screened network; a separately planned matched refit remains a distinct experiment.

Recorded measurement:

    n=4; raw +4.057580910%; adjusted +4.190318921%; charged work +0.134429442%; raw wins 0/4; adjusted wins 0/4; adjusted 95% interval [1.8679840712064966, 6.962347975728833]%

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260926_v2/hill_climb_rrpi_angular_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/candidate.py](code/candidate.py)
- [code/identity.py](code/identity.py)

## Evidence

- [HILL_CLIMB_CURRENT.md#L79](../../../records/research/2026-09-30-adaptive-confirmation-must-preserve-original-failed-gate-uncertainty-claim/HILL_CLIMB_CURRENT.md#L79)
- [HILL_CLIMB_CURRENT.md#L83](../../../records/research/2026-09-30-adaptive-confirmation-must-preserve-original-failed-gate-uncertainty-claim/HILL_CLIMB_CURRENT.md#L83)
- [identity_assessment.json#L35](identity_assessment.json#L35)
- [fit4_assessment.json#L39](fit4_assessment.json#L39)
- [direct_closeout.json#L25](direct_closeout.json#L25)

<details>
<summary>All result files and records (5)</summary>

- [HILL_CLIMB_CURRENT.md](../../../records/research/2026-09-30-adaptive-confirmation-must-preserve-original-failed-gate-uncertainty-claim/HILL_CLIMB_CURRENT.md) · record
- [source_delta.json](source_delta.json) · result
- [identity_assessment.json](identity_assessment.json) · result
- [fit4_assessment.json](fit4_assessment.json) · result
- [direct_closeout.json](direct_closeout.json) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-01-angular-input-law-propagation-unchanged-learned-parameters-worsens-all) · [Copy provenance](../../../PROVENANCE.json)

