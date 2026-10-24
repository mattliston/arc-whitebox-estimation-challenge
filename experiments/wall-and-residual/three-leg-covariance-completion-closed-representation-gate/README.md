# Three-leg covariance completion: closed at representation gate

2026-09-06T05:55Z · experiment · rejected · measured

Independent dense n6/n9 tensor, plane, push, adjoint and star-restoration identities pass. Iterative inverse certificates reach<1e-10 residual. At rank35 post12, replacing the21568-column CP component with the predeclared two-PCG covariance-metric completion and EXACT repeated-plane star restoration gives weighted next-W13 third-cumulant repeated-index plane (K21) error .3378142155 relative to the original complete CP+star+trace tensor (gate .03 FAIL).

Recorded measurement:

    Independent dense n6/n9 tensor, plane, push, adjoint and star-restoration identities pass. Iterative inverse certificates reach<1e-10 residual. The full-width current bank covariance is SPD: standardized live-block eigenvalues .01457538..31.44265 (832 active), so no jitter was needed. The30-step diagnostic reaches3.85548e-7 linear residual, NOT its1e-10 convergence criterion, and still has .3167027527 weighted next-plane error. Do not call it an exact converged projection. Both arms' post repeated entries are restored to<1.46e-17 relative, including the practical arm. The current repeated plane alone does not supply the missing distinct structure in this representation. No iteration/metric/regularizer sweep or cost/rollout expansion is justified by this fixed test. 5.405s, peakRSS4,640,669,696bytes, one nice10 four-thread CPU process. Receipts: covariance_metric_completion_certificate_v1.json and covariance_metric_completion_wide_r35_l12_v1.json/.npz. No board score.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/astra_phase2_reexamination_v1/certify_covariance_metric_completion_v1.py](code/astra_phase2_reexamination_v1/certify_covariance_metric_completion_v1.py)
- [code/astra_phase2_reexamination_v1/check_covariance_metric_completion_wide_v1.py](code/astra_phase2_reexamination_v1/check_covariance_metric_completion_wide_v1.py)
- [code/astra_phase2_reexamination_v1/compact_fullpair_final_reference_v1.py](code/astra_phase2_reexamination_v1/compact_fullpair_final_reference_v1.py)
- [code/astra_phase2_reexamination_v1/compact_scalar_final_v1.py](code/astra_phase2_reexamination_v1/compact_scalar_final_v1.py)
- [code/astra_phase2_reexamination_v1/covariance_metric_completion_v1.py](code/astra_phase2_reexamination_v1/covariance_metric_completion_v1.py)
- [code/astra_phase2_reexamination_v1/k3_radial_numpy.py](code/astra_phase2_reexamination_v1/k3_radial_numpy.py)
- [code/astra_phase2_reexamination_v1/radial_binary_compact.py](code/astra_phase2_reexamination_v1/radial_binary_compact.py)
- [code/astra_phase2_reexamination_v1/radial_compact_feed_fullpair_v1.py](code/astra_phase2_reexamination_v1/radial_compact_feed_fullpair_v1.py)
- [code/astra_phase2_reexamination_v1/radial_compact_radialfeed_v1.py](code/astra_phase2_reexamination_v1/radial_compact_radialfeed_v1.py)
- [code/astra_phase2_reexamination_v1/radial_cubic.py](code/astra_phase2_reexamination_v1/radial_cubic.py)
- [code/astra_phase2_reexamination_v1/radial_features.py](code/astra_phase2_reexamination_v1/radial_features.py)
- [code/astra_phase2_reexamination_v1/radial_gnn_numpy.py](code/astra_phase2_reexamination_v1/radial_gnn_numpy.py)

<details>
<summary>Additional programs and dependencies (2)</summary>

- [code/astra_phase2_reexamination_v1/radial_mehler.py](code/astra_phase2_reexamination_v1/radial_mehler.py)
- [code/codex_resume_20260926_v2/hill_climb_common_energy_capacity_v1/inspect.py](code/codex_resume_20260926_v2/hill_climb_common_energy_capacity_v1/inspect.py)

</details>

## Evidence

- [COVARIANCE_METRIC_COMPLETION_V1_CLOSED.md#L1](COVARIANCE_METRIC_COMPLETION_V1_CLOSED.md#L1)
- [COVARIANCE_METRIC_COMPLETION_V1_CLOSED.md#L8](COVARIANCE_METRIC_COMPLETION_V1_CLOSED.md#L8)
- [COVARIANCE_METRIC_COMPLETION_V1_CLOSED.md#L19](COVARIANCE_METRIC_COMPLETION_V1_CLOSED.md#L19)
- [COVARIANCE_METRIC_COMPLETION_V1_CLOSED.md#L24](COVARIANCE_METRIC_COMPLETION_V1_CLOSED.md#L24)

<details>
<summary>All result files and records (4)</summary>

- [COVARIANCE_METRIC_COMPLETION_V1_CLOSED.md](COVARIANCE_METRIC_COMPLETION_V1_CLOSED.md) · record
- [covariance_metric_completion_certificate_v1.json](covariance_metric_completion_certificate_v1.json) · result
- [covariance_metric_completion_wide_r35_l12_v1.json](covariance_metric_completion_wide_r35_l12_v1.json) · result
- [COVARIANCE_METRIC_COMPLETION_DESIGN_V1.md](COVARIANCE_METRIC_COMPLETION_DESIGN_V1.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-06-three-leg-covariance-completion-closed-representation-gate) · [Copy provenance](../../../PROVENANCE.json)

