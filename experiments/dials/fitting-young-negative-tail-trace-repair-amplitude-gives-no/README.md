# Fitting the young negative-tail trace repair amplitude gives no held-out improvement

2026-10-01T03:15Z · experiment · rejected · measured

The trace repair for cropped negative-tail rows of young carried columns receives one scalar amplitude, with crop thresholds and all other state fixed. A single secant fit chooses gain 0.6574173127823337 and predicts a much larger gain than the actual four-network improvement. Both raw and adjusted scores then worsen on the eight-network check, closing the experiment.

Recorded measurement:

    gain 0.6574173127823337; predicted parent-weighted loss reduction 0.278131197551%; n=4; raw -0.026930529%; adjusted -0.029563887%; charged work +0.000006231%; raw wins 2/4; adjusted wins 2/4; adjusted 95% interval [-0.5093981749801135, 0.4978309841404238]%; held-out n=8; raw +0.019383236%; adjusted +0.021910034%; charged work +0.000006130%; raw wins 4/8; adjusted wins 4/8; adjusted 95% interval [-0.2785387590183763, 0.35486814774956577]%

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260926_v2`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/hill_climb_rrpi_ynt_amplitude_v1/direction.py](code/hill_climb_rrpi_ynt_amplitude_v1/direction.py)
- [code/hill_climb_rrpi_ynt_amplitude_v1/endpoint.py](code/hill_climb_rrpi_ynt_amplitude_v1/endpoint.py)
- [code/hill_climb_rrpi_ynt_amplitude_v1/fit.py](code/hill_climb_rrpi_ynt_amplitude_v1/fit.py)
- [code/hill_climb_pair_rr_inverse_v1/candidate.py](code/hill_climb_pair_rr_inverse_v1/candidate.py)
- [code/hill_climb_rrpi_k4_rowdiag_coadapt_v1/common.py](code/hill_climb_rrpi_k4_rowdiag_coadapt_v1/common.py)
- [code/hill_climb_rrpi_smooth_kink_v1/assess.py](code/hill_climb_rrpi_smooth_kink_v1/assess.py)
- [code/hill_climb_rrpi_smooth_kink_v1/candidate.py](code/hill_climb_rrpi_smooth_kink_v1/candidate.py)
- [code/hill_climb_rrpi_smooth_kink_v1/prepare.py](code/hill_climb_rrpi_smooth_kink_v1/prepare.py)
- [code/hill_climb_rrpi_ynt_amplitude_v1/assess.py](code/hill_climb_rrpi_ynt_amplitude_v1/assess.py)
- [code/hill_climb_rrpi_ynt_amplitude_v1/prepare.py](code/hill_climb_rrpi_ynt_amplitude_v1/prepare.py)
- [code/hill_climb_uniform_gnn_gain_v1/endpoint.py](code/hill_climb_uniform_gnn_gain_v1/endpoint.py)
- [code/hill_climb_uniform_gnn_gain_v1/fit.py](code/hill_climb_uniform_gnn_gain_v1/fit.py)

<details>
<summary>Additional programs and dependencies (1)</summary>

- [code/native_screen_engine_v1/screen.py](code/native_screen_engine_v1/screen.py)

</details>

## Evidence

- [HILL_CLIMB_CURRENT.md#L88](../../../records/research/2026-09-30-adaptive-confirmation-must-preserve-original-failed-gate-uncertainty-claim/HILL_CLIMB_CURRENT.md#L88)
- [HILL_CLIMB_CURRENT.md#L94](../../../records/research/2026-09-30-adaptive-confirmation-must-preserve-original-failed-gate-uncertainty-claim/HILL_CLIMB_CURRENT.md#L94)
- [HILL_CLIMB_CURRENT.md#L110](../../../records/research/2026-09-30-adaptive-confirmation-must-preserve-original-failed-gate-uncertainty-claim/HILL_CLIMB_CURRENT.md#L110)
- [fit_result.json#L76](fit_result.json#L76)
- [endpoint_fit4_assessment.json#L38](endpoint_fit4_assessment.json#L38)
- [endpoint_check8_assessment.json#L42](endpoint_check8_assessment.json#L42)
- [closeout.json#L1](closeout.json#L1)

<details>
<summary>All result files and records (5)</summary>

- [HILL_CLIMB_CURRENT.md](../../../records/research/2026-09-30-adaptive-confirmation-must-preserve-original-failed-gate-uncertainty-claim/HILL_CLIMB_CURRENT.md) · record
- [fit_result.json](fit_result.json) · result
- [endpoint_fit4_assessment.json](endpoint_fit4_assessment.json) · result
- [endpoint_check8_assessment.json](endpoint_check8_assessment.json) · result
- [closeout.json](closeout.json) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-01-fitting-young-negative-tail-trace-repair-amplitude-gives-no) · [Copy provenance](../../../PROVENANCE.json)

