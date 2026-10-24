# A fitted mean-annihilating covariance component improves four training networks but fails eight held-out networks

2026-10-01T02:20Z · experiment · rejected · measured

A single coefficient interpolates or extrapolates between the existing covariance increment D and its mean-annihilating projection Q at layers 10–12. A regularized secant fit chooses alpha −1.4207791435459642 for D+alpha(Q−D), giving a small four-network gain that does not persist on the fixed eight-network check. The endpoint is rejected without wider qualification.

Recorded measurement:

    n=4; raw -0.180799076%; adjusted -0.148414505%; charged work +0.031148901%; raw wins 3/4; adjusted wins 3/4; adjusted 95% interval [-0.43912484474571967, 0.1160795136018189]%; held-out n=8; raw +0.006941804%; adjusted +0.033173990%; charged work +0.030642248%; raw wins 5/8; adjusted wins 4/8; adjusted 95% interval [-0.1436396486156566, 0.24020433524298834]%

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260926_v2`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/hill_climb_rrpi_gn_mean_component_fit_v1/endpoint.py](code/hill_climb_rrpi_gn_mean_component_fit_v1/endpoint.py)
- [code/hill_climb_rrpi_gn_mean_component_fit_v1/fit.py](code/hill_climb_rrpi_gn_mean_component_fit_v1/fit.py)
- [code/hill_climb_pair_rr_inverse_v1/candidate.py](code/hill_climb_pair_rr_inverse_v1/candidate.py)
- [code/hill_climb_rrpi_gn_mean_component_fit_v1/assess.py](code/hill_climb_rrpi_gn_mean_component_fit_v1/assess.py)
- [code/hill_climb_rrpi_gn_mean_component_fit_v1/prepare.py](code/hill_climb_rrpi_gn_mean_component_fit_v1/prepare.py)
- [code/hill_climb_rrpi_gn_perp_v1/assess.py](code/hill_climb_rrpi_gn_perp_v1/assess.py)
- [code/hill_climb_rrpi_gn_perp_v1/candidate.py](code/hill_climb_rrpi_gn_perp_v1/candidate.py)
- [code/hill_climb_rrpi_gn_perp_v1/fragment.py](code/hill_climb_rrpi_gn_perp_v1/fragment.py)
- [code/hill_climb_rrpi_gn_perp_v1/prepare.py](code/hill_climb_rrpi_gn_perp_v1/prepare.py)
- [code/hill_climb_rrpi_k4_rowdiag_coadapt_v1/common.py](code/hill_climb_rrpi_k4_rowdiag_coadapt_v1/common.py)
- [code/hill_climb_rrpi_smooth_kink_v1/assess.py](code/hill_climb_rrpi_smooth_kink_v1/assess.py)
- [code/hill_climb_rrpi_smooth_kink_v1/candidate.py](code/hill_climb_rrpi_smooth_kink_v1/candidate.py)

<details>
<summary>Additional programs and dependencies (4)</summary>

- [code/hill_climb_rrpi_smooth_kink_v1/prepare.py](code/hill_climb_rrpi_smooth_kink_v1/prepare.py)
- [code/hill_climb_uniform_gnn_gain_v1/endpoint.py](code/hill_climb_uniform_gnn_gain_v1/endpoint.py)
- [code/hill_climb_uniform_gnn_gain_v1/fit.py](code/hill_climb_uniform_gnn_gain_v1/fit.py)
- [code/native_screen_engine_v1/screen.py](code/native_screen_engine_v1/screen.py)

</details>

## Evidence

- [HILL_CLIMB_CURRENT.md#L130](../../../records/research/2026-09-30-adaptive-confirmation-must-preserve-original-failed-gate-uncertainty-claim/HILL_CLIMB_CURRENT.md#L130)
- [HILL_CLIMB_CURRENT.md#L135](../../../records/research/2026-09-30-adaptive-confirmation-must-preserve-original-failed-gate-uncertainty-claim/HILL_CLIMB_CURRENT.md#L135)
- [HILL_CLIMB_CURRENT.md#L136](../../../records/research/2026-09-30-adaptive-confirmation-must-preserve-original-failed-gate-uncertainty-claim/HILL_CLIMB_CURRENT.md#L136)
- [HILL_CLIMB_CURRENT.md#L142](../../../records/research/2026-09-30-adaptive-confirmation-must-preserve-original-failed-gate-uncertainty-claim/HILL_CLIMB_CURRENT.md#L142)
- [fit_result.json#L70](fit_result.json#L70)
- [endpoint_fit4_assessment.json#L38](endpoint_fit4_assessment.json#L38)
- [endpoint_check8_assessment.json#L42](endpoint_check8_assessment.json#L42)
- [closeout.json#L22](closeout.json#L22)

<details>
<summary>All result files and records (5)</summary>

- [HILL_CLIMB_CURRENT.md](../../../records/research/2026-09-30-adaptive-confirmation-must-preserve-original-failed-gate-uncertainty-claim/HILL_CLIMB_CURRENT.md) · record
- [fit_result.json](fit_result.json) · result
- [endpoint_fit4_assessment.json](endpoint_fit4_assessment.json) · result
- [endpoint_check8_assessment.json](endpoint_check8_assessment.json) · result
- [closeout.json](closeout.json) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-01-fitted-mean-annihilating-covariance-component-improves-four-training-networks) · [Copy provenance](../../../PROVENANCE.json)

