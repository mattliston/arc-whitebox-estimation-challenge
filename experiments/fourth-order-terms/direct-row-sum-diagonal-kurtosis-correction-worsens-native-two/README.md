# Direct row-sum and diagonal kurtosis correction worsens the native two-network screen

2026-10-01T01:00Z · experiment · rejected · measured

Adding the row-sum and diagonal fourth-cumulant contraction at unit strength through layers 2–15 worsens error with the existing learned correction held fixed. The actual charged-work increase is approximately twice the original source estimate; a separate matched terminal-readout fit is still required to assess that narrower adaptation hypothesis.

Recorded measurement:

    n=2; raw +0.938239792%; adjusted +1.123645899%; charged work +0.184181167%; raw wins 0/2; adjusted wins 0/2; adjusted 95% interval [0.7383507872369188, 1.527667995999793]%; original work estimate approximately +0.09%; layers 0–1 byte-identical

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260926_v2`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/hill_climb_rrpi_k4_rowdiag_v1/candidate.py](code/hill_climb_rrpi_k4_rowdiag_v1/candidate.py)
- [code/hill_climb_rrpi_k4_rowdiag_v1/build.py](code/hill_climb_rrpi_k4_rowdiag_v1/build.py)
- [code/hill_climb_pair_rr_inverse_v1/candidate.py](code/hill_climb_pair_rr_inverse_v1/candidate.py)
- [code/hill_climb_rrpi_k4_rowdiag_v1/candidate_zero.py](code/hill_climb_rrpi_k4_rowdiag_v1/candidate_zero.py)
- [code/hill_climb_rrpi_k4_rowdiag_v1/fragment.py](code/hill_climb_rrpi_k4_rowdiag_v1/fragment.py)

## Evidence

- [HILL_CLIMB_CURRENT.md#L176](../../../records/research/2026-09-30-adaptive-confirmation-must-preserve-original-failed-gate-uncertainty-claim/HILL_CLIMB_CURRENT.md#L176)
- [HILL_CLIMB_CURRENT.md#L187](../../../records/research/2026-09-30-adaptive-confirmation-must-preserve-original-failed-gate-uncertainty-claim/HILL_CLIMB_CURRENT.md#L187)
- [physical_native2_assessment.json#L35](physical_native2_assessment.json#L35)

<details>
<summary>All result files and records (4)</summary>

- [HILL_CLIMB_CURRENT.md](../../../records/research/2026-09-30-adaptive-confirmation-must-preserve-original-failed-gate-uncertainty-claim/HILL_CLIMB_CURRENT.md) · record
- [physical_native2_assessment.json](physical_native2_assessment.json) · result
- [physical_native2_guard.json](physical_native2_guard.json) · result
- [cost_forecast_correction.json](cost_forecast_correction.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-10-01-direct-row-sum-diagonal-kurtosis-correction-worsens-native-two) · [Copy provenance](../../../PROVENANCE.json)

