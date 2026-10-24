# Smoothing the covariance-correction kink mask raises adjusted error by 0.259%

2026-10-01T00:04Z · experiment · rejected · measured

Replacing two hard kink assignments in the covariance correction by clip(2−|alpha|,0,1) leaves the learned parameters and other physical terms fixed. The four-network native screen worsens both raw and adjusted error, so the trial ends without broader qualification or retention.

Recorded measurement:

    n=4; raw +0.259904951%; adjusted +0.259351265%; charged work +0.000020252%; raw wins 2/4; adjusted wins 2/4; adjusted 95% interval [-0.1811206993857395, 0.8807732559717785]%

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260926_v2`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/hill_climb_rrpi_smooth_kink_v1/candidate.py](code/hill_climb_rrpi_smooth_kink_v1/candidate.py)
- [code/hill_climb_rrpi_smooth_kink_v1/prepare.py](code/hill_climb_rrpi_smooth_kink_v1/prepare.py)
- [code/hill_climb_rrpi_smooth_kink_v1/assess.py](code/hill_climb_rrpi_smooth_kink_v1/assess.py)
- [code/hill_climb_pair_rr_inverse_v1/candidate.py](code/hill_climb_pair_rr_inverse_v1/candidate.py)
- [code/hill_climb_uniform_gnn_gain_v1/endpoint.py](code/hill_climb_uniform_gnn_gain_v1/endpoint.py)
- [code/hill_climb_uniform_gnn_gain_v1/fit.py](code/hill_climb_uniform_gnn_gain_v1/fit.py)
- [code/native_screen_engine_v1/screen.py](code/native_screen_engine_v1/screen.py)

## Evidence

- [HILL_CLIMB_CURRENT.md#L209](../../../records/research/2026-09-30-adaptive-confirmation-must-preserve-original-failed-gate-uncertainty-claim/HILL_CLIMB_CURRENT.md#L209)
- [HILL_CLIMB_CURRENT.md#L211](../../../records/research/2026-09-30-adaptive-confirmation-must-preserve-original-failed-gate-uncertainty-claim/HILL_CLIMB_CURRENT.md#L211)
- [HILL_CLIMB_CURRENT.md#L214](../../../records/research/2026-09-30-adaptive-confirmation-must-preserve-original-failed-gate-uncertainty-claim/HILL_CLIMB_CURRENT.md#L214)
- [fit4_assessment.json#L42](fit4_assessment.json#L42)
- [closeout.json#L32](closeout.json#L32)

<details>
<summary>All result files and records (4)</summary>

- [HILL_CLIMB_CURRENT.md](../../../records/research/2026-09-30-adaptive-confirmation-must-preserve-original-failed-gate-uncertainty-claim/HILL_CLIMB_CURRENT.md) · record
- [fit4_assessment.json](fit4_assessment.json) · result
- [closeout.json](closeout.json) · record
- [native_fit4_guard.json](native_fit4_guard.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-10-01-smoothing-covariance-correction-kink-mask-raises-adjusted-error-0) · [Copy provenance](../../../PROVENANCE.json)

