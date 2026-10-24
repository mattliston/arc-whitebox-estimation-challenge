# Fitting one kurtosis-correction amplitude fails to realize its predicted error reduction

2026-10-01T01:34Z · experiment · rejected · measured

Fixed native probes at strengths −1 and +1 determine a single regularized secant amplitude while leaving the learned parameters unchanged. The fitted strength predicts a 0.331% parent-weighted loss reduction, but its actual native endpoint worsens both raw and adjusted error; the forecast excluded the extra candidate work.

Recorded measurement:

    four networks; alpha 0.49427291965304243; predicted loss reduction 0.3314857887%; actual raw +0.0201394486%, adjusted +0.2080722041%, charged work +0.1844390701%, 1/4 wins; −1 raw +2.521858993%, +1 raw +0.116669691%

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260926_v2`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/hill_climb_rrpi_k4_rowdiag_amplitude_v1/candidate_minus.py](code/hill_climb_rrpi_k4_rowdiag_amplitude_v1/candidate_minus.py)
- [code/hill_climb_rrpi_k4_rowdiag_amplitude_v1/candidate_plus.py](code/hill_climb_rrpi_k4_rowdiag_amplitude_v1/candidate_plus.py)
- [code/hill_climb_rrpi_k4_rowdiag_amplitude_v1/endpoint.py](code/hill_climb_rrpi_k4_rowdiag_amplitude_v1/endpoint.py)
- [code/hill_climb_rrpi_k4_rowdiag_amplitude_v1/fit.py](code/hill_climb_rrpi_k4_rowdiag_amplitude_v1/fit.py)
- [code/hill_climb_pair_rr_inverse_v1/candidate.py](code/hill_climb_pair_rr_inverse_v1/candidate.py)
- [code/hill_climb_rrpi_k4_rowdiag_amplitude_v1/assess.py](code/hill_climb_rrpi_k4_rowdiag_amplitude_v1/assess.py)
- [code/hill_climb_rrpi_k4_rowdiag_amplitude_v1/prepare.py](code/hill_climb_rrpi_k4_rowdiag_amplitude_v1/prepare.py)
- [code/hill_climb_rrpi_k4_rowdiag_coadapt_v1/assess_check8.py](code/hill_climb_rrpi_k4_rowdiag_coadapt_v1/assess_check8.py)
- [code/hill_climb_rrpi_k4_rowdiag_coadapt_v1/common.py](code/hill_climb_rrpi_k4_rowdiag_coadapt_v1/common.py)
- [code/hill_climb_rrpi_k4_rowdiag_coadapt_v1/prepare_check8.py](code/hill_climb_rrpi_k4_rowdiag_coadapt_v1/prepare_check8.py)
- [code/hill_climb_rrpi_k4_rowdiag_v1/candidate.py](code/hill_climb_rrpi_k4_rowdiag_v1/candidate.py)
- [code/hill_climb_rrpi_smooth_kink_v1/assess.py](code/hill_climb_rrpi_smooth_kink_v1/assess.py)

<details>
<summary>Additional programs and dependencies (5)</summary>

- [code/hill_climb_rrpi_smooth_kink_v1/candidate.py](code/hill_climb_rrpi_smooth_kink_v1/candidate.py)
- [code/hill_climb_rrpi_smooth_kink_v1/prepare.py](code/hill_climb_rrpi_smooth_kink_v1/prepare.py)
- [code/hill_climb_uniform_gnn_gain_v1/endpoint.py](code/hill_climb_uniform_gnn_gain_v1/endpoint.py)
- [code/hill_climb_uniform_gnn_gain_v1/fit.py](code/hill_climb_uniform_gnn_gain_v1/fit.py)
- [code/native_screen_engine_v1/screen.py](code/native_screen_engine_v1/screen.py)

</details>

## Evidence

- [HILL_CLIMB_CURRENT.md#L159](../../../records/research/2026-09-30-adaptive-confirmation-must-preserve-original-failed-gate-uncertainty-claim/HILL_CLIMB_CURRENT.md#L159)
- [HILL_CLIMB_CURRENT.md#L163](../../../records/research/2026-09-30-adaptive-confirmation-must-preserve-original-failed-gate-uncertainty-claim/HILL_CLIMB_CURRENT.md#L163)
- [HILL_CLIMB_CURRENT.md#L164](../../../records/research/2026-09-30-adaptive-confirmation-must-preserve-original-failed-gate-uncertainty-claim/HILL_CLIMB_CURRENT.md#L164)
- [fit_result.json#L69](fit_result.json#L69)
- [endpoint_fit4_assessment.json#L38](endpoint_fit4_assessment.json#L38)
- [closeout.md#L1](closeout.md#L1)

<details>
<summary>All result files and records (4)</summary>

- [HILL_CLIMB_CURRENT.md](../../../records/research/2026-09-30-adaptive-confirmation-must-preserve-original-failed-gate-uncertainty-claim/HILL_CLIMB_CURRENT.md) · record
- [fit_result.json](fit_result.json) · result
- [endpoint_fit4_assessment.json](endpoint_fit4_assessment.json) · result
- [closeout.md](closeout.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-01-fitting-one-kurtosis-correction-amplitude-fails-realize-predicted-error) · [Copy provenance](../../../PROVENANCE.json)

