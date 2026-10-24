# Uniform learned-correction gain of 1.02 improves fitting error but regresses on eight checks

2026-09-30 · experiment · rejected · measured

Fixed0.98/1.02 probes and a one-dimensional ridge-box fit selected the1.02 endpoint. A matched gain-one control reproduced every fitting output, ruling out the extra operation as the fitting-baseline discrepancy, while the selected endpoint failed separate checks.

Recorded measurement:

    FIT4 raw-.12726654%,adjusted-.1249272%,1/4wins;CHECK8 raw+.2844624142%,adjusted+.2807427478%,2/8wins,CIpositive;identity all16/all4 bytes exact,cold+15F/warm0.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/codex_resume_20260926_v2/hill_climb_uniform_gnn_gain_v1/fit.py](code/codex_resume_20260926_v2/hill_climb_uniform_gnn_gain_v1/fit.py)
- [code/codex_resume_20260919_v1/current_p_native_lifecycle_v1/guard.py](code/codex_resume_20260919_v1/current_p_native_lifecycle_v1/guard.py)
- [code/codex_resume_20260926_v2/hill_climb_e1_ynt_psdb_d5_v1/native_probe.py](code/codex_resume_20260926_v2/hill_climb_e1_ynt_psdb_d5_v1/native_probe.py)
- [code/codex_resume_20260926_v2/hill_climb_pair_rr_inverse_v1/candidate.py](code/codex_resume_20260926_v2/hill_climb_pair_rr_inverse_v1/candidate.py)
- [code/codex_resume_20260926_v2/hill_climb_uniform_gnn_gain_v1/assess_endpoint.py](code/codex_resume_20260926_v2/hill_climb_uniform_gnn_gain_v1/assess_endpoint.py)
- [code/codex_resume_20260926_v2/hill_climb_uniform_gnn_gain_v1/assess_identity.py](code/codex_resume_20260926_v2/hill_climb_uniform_gnn_gain_v1/assess_identity.py)
- [code/codex_resume_20260926_v2/hill_climb_uniform_gnn_gain_v1/endpoint.py](code/codex_resume_20260926_v2/hill_climb_uniform_gnn_gain_v1/endpoint.py)
- [code/codex_resume_20260926_v2/hill_climb_uniform_gnn_gain_v1/identity.py](code/codex_resume_20260926_v2/hill_climb_uniform_gnn_gain_v1/identity.py)
- [code/codex_resume_20260926_v2/hill_climb_uniform_gnn_gain_v1/minus.py](code/codex_resume_20260926_v2/hill_climb_uniform_gnn_gain_v1/minus.py)
- [code/codex_resume_20260926_v2/hill_climb_uniform_gnn_gain_v1/plus.py](code/codex_resume_20260926_v2/hill_climb_uniform_gnn_gain_v1/plus.py)
- [code/codex_resume_20260926_v2/hill_climb_uniform_gnn_gain_v1/prepare.py](code/codex_resume_20260926_v2/hill_climb_uniform_gnn_gain_v1/prepare.py)
- [code/codex_resume_20260926_v2/native_screen_engine_v1/screen.py](code/codex_resume_20260926_v2/native_screen_engine_v1/screen.py)

## Evidence

- [HILL_CLIMB_CURRENT.md#L242](../../../records/research/2026-09-30-adaptive-confirmation-must-preserve-original-failed-gate-uncertainty-claim/HILL_CLIMB_CURRENT.md#L242)

<details>
<summary>All result files and records (5)</summary>

- [HILL_CLIMB_CURRENT.md](../../../records/research/2026-09-30-adaptive-confirmation-must-preserve-original-failed-gate-uncertainty-claim/HILL_CLIMB_CURRENT.md) · record
- [protocol.json](protocol.json) · result
- [fit_result.json](fit_result.json) · result
- [identity_assessment.json](identity_assessment.json) · result
- [closeout.json](closeout.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-09-30-uniform-learned-correction-gain-1-02-improves-fitting-error) · [Copy provenance](../../../PROVENANCE.json)

