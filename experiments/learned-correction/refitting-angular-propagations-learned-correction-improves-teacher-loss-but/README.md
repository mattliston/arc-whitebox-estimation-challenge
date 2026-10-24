# Refitting the angular propagation’s learned correction improves teacher loss but fails native execution

2026-10-01T03:55Z · experiment · rejected · measured

Forty fresh teacher-forced native trajectories support a fixed five-epoch fit of the angular model’s learned mean correction, with 32 fitting networks and eight training-held-out networks. The held-out teacher loss improves by 2.108%, but the native closed-loop endpoint still worsens raw error by 3.089% against the original Gaussian parent on all four screened networks. No wider qualification follows.

Recorded measurement:

    five epochs, 160 updates; held-out teacher error −2.107919604%; n=4; raw +3.088966175%; adjusted +3.231604781%; charged work +0.134429442%; raw wins 0/4; adjusted wins 0/4; adjusted 95% interval [1.9034119481059486, 4.823546043192994]%; captures 1057.982127 seconds, training 23.499424 seconds

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260926_v2/hill_climb_rrpi_angular_coadapt_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/collect.py](code/collect.py)
- [code/gnn.py](code/gnn.py)
- [code/replay.py](code/replay.py)
- [code/train.py](code/train.py)
- [code/candidate.py](code/candidate.py)
- [code/common.py](code/common.py)

## Evidence

- [HILL_CLIMB_CURRENT.md#L64](../../../records/research/2026-09-30-adaptive-confirmation-must-preserve-original-failed-gate-uncertainty-claim/HILL_CLIMB_CURRENT.md#L64)
- [HILL_CLIMB_CURRENT.md#L69](../../../records/research/2026-09-30-adaptive-confirmation-must-preserve-original-failed-gate-uncertainty-claim/HILL_CLIMB_CURRENT.md#L69)
- [HILL_CLIMB_CURRENT.md#L76](../../../records/research/2026-09-30-adaptive-confirmation-must-preserve-original-failed-gate-uncertainty-claim/HILL_CLIMB_CURRENT.md#L76)
- [HILL_CLIMB_CURRENT.md#L77](../../../records/research/2026-09-30-adaptive-confirmation-must-preserve-original-failed-gate-uncertainty-claim/HILL_CLIMB_CURRENT.md#L77)
- [replay_result.json#L1](replay_result.json#L1)
- [fit_result.json#L1](fit_result.json#L1)
- [endpoint_fit4_assessment.json#L39](endpoint_fit4_assessment.json#L39)
- [closeout.json#L32](closeout.json#L32)

<details>
<summary>All result files and records (6)</summary>

- [HILL_CLIMB_CURRENT.md](../../../records/research/2026-09-30-adaptive-confirmation-must-preserve-original-failed-gate-uncertainty-claim/HILL_CLIMB_CURRENT.md) · record
- [replay_result.json](replay_result.json) · result
- [protocol.json](protocol.json) · record
- [fit_result.json](fit_result.json) · result
- [endpoint_fit4_assessment.json](endpoint_fit4_assessment.json) · result
- [closeout.json](closeout.json) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-01-refitting-angular-propagations-learned-correction-improves-teacher-loss-but) · [Copy provenance](../../../PROVENANCE.json)

