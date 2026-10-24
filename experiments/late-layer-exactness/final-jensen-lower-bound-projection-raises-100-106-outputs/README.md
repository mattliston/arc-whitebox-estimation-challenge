# Final Jensen lower-bound projection raises 100 of 106 outputs and increases raw error 0.318748%

2026-09-30 · experiment · rejected · measured

Saved native outputs were projected onto max(final,W15.T served_output14), using float64 contraction then float32 output. The result was a saved-output diagnostic, without an estimator run or actual new compute bill.

Recorded measurement:

    Raw+.318748%,estimated adjusted+.319673%,1win,maxreg+.804495%;100/106 units raised.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260926_v2`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/hill_climb_final_jensen_v1/screen_saved.py](code/hill_climb_final_jensen_v1/screen_saved.py)
- [code/hill_climb_pair_rr_inverse_v1/candidate.py](code/hill_climb_pair_rr_inverse_v1/candidate.py)

## Evidence

- [HILL_CLIMB_CURRENT.md#L563](../../../records/research/2026-09-30-adaptive-confirmation-must-preserve-original-failed-gate-uncertainty-claim/HILL_CLIMB_CURRENT.md#L563)

<details>
<summary>All result files and records (3)</summary>

- [HILL_CLIMB_CURRENT.md](../../../records/research/2026-09-30-adaptive-confirmation-must-preserve-original-failed-gate-uncertainty-claim/HILL_CLIMB_CURRENT.md) · record
- [assessment.json](assessment.json) · result
- [protocol.json](protocol.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-09-30-final-jensen-lower-bound-projection-raises-100-106-outputs) · [Copy provenance](../../../PROVENANCE.json)

