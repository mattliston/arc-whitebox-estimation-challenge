# Transferring the angular-trained mean correction to Gaussian propagation worsens native error

2026-10-01T04:04Z · experiment · rejected · measured

Only the newly fitted angular model’s learned-parameter payload is transplanted onto the original Gaussian propagation. This isolated component test performs no additional fitting and changes no charged work, yet raw error rises 1.939% and three of four networks worsen. Both the full angular refit and this payload-only transfer are therefore rejected.

Recorded measurement:

    n=4; raw +1.939382816%; adjusted +1.950706246%; charged work +0.000000000%; raw wins 1/4; adjusted wins 1/4; adjusted 95% interval [0.3411549392239088, 3.288581910098576]%

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260926_v2/hill_climb_rrpi_angular_gnn_transfer_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/candidate.py](code/candidate.py)

## Evidence

- [HILL_CLIMB_CURRENT.md#L58](../../../records/research/2026-09-30-adaptive-confirmation-must-preserve-original-failed-gate-uncertainty-claim/HILL_CLIMB_CURRENT.md#L58)
- [HILL_CLIMB_CURRENT.md#L65](../../../records/research/2026-09-30-adaptive-confirmation-must-preserve-original-failed-gate-uncertainty-claim/HILL_CLIMB_CURRENT.md#L65)
- [fit4_assessment.json#L39](fit4_assessment.json#L39)
- [closeout.json#L25](closeout.json#L25)

<details>
<summary>All result files and records (5)</summary>

- [HILL_CLIMB_CURRENT.md](../../../records/research/2026-09-30-adaptive-confirmation-must-preserve-original-failed-gate-uncertainty-claim/HILL_CLIMB_CURRENT.md) · record
- [source_delta.json](source_delta.json) · result
- [fit4_assessment.json](fit4_assessment.json) · result
- [closeout.json](closeout.json) · record
- [native_fit4_guard.json](native_fit4_guard.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-10-01-transferring-angular-trained-mean-correction-gaussian-propagation-worsens-native) · [Copy provenance](../../../PROVENANCE.json)

