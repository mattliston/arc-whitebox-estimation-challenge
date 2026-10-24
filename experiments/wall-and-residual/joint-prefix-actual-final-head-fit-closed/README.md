# Joint prefix and actual final-head fit: closed

2026-09-06T19:16Z · experiment · rejected · measured

- Original supplied P/Q NumPy raw final-layer mean squared error (MSE): 2.247525992904174e-8. - Jointly fitted P/Q NumPy raw final-layer MSE: 2.244005465955291e-8. - Pooled ratio: 0.9984335990061971 (0.1566401% improvement). - Wins: 22/40. - NumPy/MPS fidelity gate: passed on all 40 networks. - Declared improvement gates: pooled ratio <=0.90 and at least 24/40 wins. - Verdict: FAIL. No candidate port, panel run, submission, earlier-checkpoint selection, amplitude sweep, optimizer sweep, or epoch extension. - joint_prefix_final_fit_v1/receipt.json and failure.json - joint_prefix_final_terminal_integrity_v1.json - joint_prefix_final_provenance_v1/receipt.json - joint_prefix_final_numpy_baselines_v1/receipt.json - joint_prefix_final_judge_cached_v2/receipt.json

Recorded measurement:

    - Original supplied P/Q NumPy raw final-layer MSE: 2.247525992904174e-8. - Jointly fitted P/Q NumPy raw final-layer MSE: 2.244005465955291e-8. - Pooled ratio: 0.9984335990061971 (0.1566401% improvement). - Wins: 22/40. - NumPy/MPS fidelity gate: passed on all 40 networks. - Declared improvement gates: pooled ratio <=0.90 and at least 24/40 wins. - Verdict: FAIL. No candidate port, panel run, submission, earlier-checkpoint selection, amplitude sweep, optimizer sweep, or epoch extension. Independent judge completed at 2026-09-06 19:13:29.076114 UTC, exit 0, in 252.600 seconds with peak process RSS 6,413,893,632 bytes. All workers released. Official bank remains #329926, adjusted 4.982231460183304e-9.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/astra_phase2_reexamination_v1/audit_joint_terminal_artifacts_v1.py](code/astra_phase2_reexamination_v1/audit_joint_terminal_artifacts_v1.py)
- [code/astra_phase2_reexamination_v1/centered_bank_autodiff_v1.py](code/astra_phase2_reexamination_v1/centered_bank_autodiff_v1.py)
- [code/astra_phase2_reexamination_v1/centered_bank_autodiff_v3.py](code/astra_phase2_reexamination_v1/centered_bank_autodiff_v3.py)
- [code/astra_phase2_reexamination_v1/centered_torch_numerics_v1.py](code/astra_phase2_reexamination_v1/centered_torch_numerics_v1.py)
- [code/astra_phase2_reexamination_v1/centered_torch_numerics_v3.py](code/astra_phase2_reexamination_v1/centered_torch_numerics_v3.py)
- [code/astra_phase2_reexamination_v1/check_joint_prefix_final_loader_v1.py](code/astra_phase2_reexamination_v1/check_joint_prefix_final_loader_v1.py)
- [code/astra_phase2_reexamination_v1/compact_final_head_v1.py](code/astra_phase2_reexamination_v1/compact_final_head_v1.py)
- [code/astra_phase2_reexamination_v1/compact_fullpair_final_reference_v1.py](code/astra_phase2_reexamination_v1/compact_fullpair_final_reference_v1.py)
- [code/astra_phase2_reexamination_v1/compact_scalar_final_v1.py](code/astra_phase2_reexamination_v1/compact_scalar_final_v1.py)
- [code/astra_phase2_reexamination_v1/current_transport_shadow_v1.py](code/astra_phase2_reexamination_v1/current_transport_shadow_v1.py)
- [code/astra_phase2_reexamination_v1/current_transport_shadow_v3.py](code/astra_phase2_reexamination_v1/current_transport_shadow_v3.py)
- [code/astra_phase2_reexamination_v1/full_carrier_autodiff_v1.py](code/astra_phase2_reexamination_v1/full_carrier_autodiff_v1.py)

<details>
<summary>Additional programs and dependencies (19)</summary>

- [code/astra_phase2_reexamination_v1/full_carrier_bank_autodiff_v1.py](code/astra_phase2_reexamination_v1/full_carrier_bank_autodiff_v1.py)
- [code/astra_phase2_reexamination_v1/gaussian_cdf_quadrature_v1.py](code/astra_phase2_reexamination_v1/gaussian_cdf_quadrature_v1.py)
- [code/astra_phase2_reexamination_v1/joint_prefix_final_data_v1.py](code/astra_phase2_reexamination_v1/joint_prefix_final_data_v1.py)
- [code/astra_phase2_reexamination_v1/joint_prefix_final_numpy_v1.py](code/astra_phase2_reexamination_v1/joint_prefix_final_numpy_v1.py)
- [code/astra_phase2_reexamination_v1/judge_joint_prefix_final_cached_v2.py](code/astra_phase2_reexamination_v1/judge_joint_prefix_final_cached_v2.py)
- [code/astra_phase2_reexamination_v1/judge_joint_prefix_final_v1.py](code/astra_phase2_reexamination_v1/judge_joint_prefix_final_v1.py)
- [code/astra_phase2_reexamination_v1/k3_radial_numpy.py](code/astra_phase2_reexamination_v1/k3_radial_numpy.py)
- [code/astra_phase2_reexamination_v1/prepare_joint_numpy_baselines_v1.py](code/astra_phase2_reexamination_v1/prepare_joint_numpy_baselines_v1.py)
- [code/astra_phase2_reexamination_v1/radial_binary_compact.py](code/astra_phase2_reexamination_v1/radial_binary_compact.py)
- [code/astra_phase2_reexamination_v1/radial_compact_feed_fullpair_v1.py](code/astra_phase2_reexamination_v1/radial_compact_feed_fullpair_v1.py)
- [code/astra_phase2_reexamination_v1/radial_compact_radialfeed_v1.py](code/astra_phase2_reexamination_v1/radial_compact_radialfeed_v1.py)
- [code/astra_phase2_reexamination_v1/radial_cubic.py](code/astra_phase2_reexamination_v1/radial_cubic.py)
- [code/astra_phase2_reexamination_v1/radial_features.py](code/astra_phase2_reexamination_v1/radial_features.py)
- [code/astra_phase2_reexamination_v1/radial_gnn_numpy.py](code/astra_phase2_reexamination_v1/radial_gnn_numpy.py)
- [code/astra_phase2_reexamination_v1/radial_mehler.py](code/astra_phase2_reexamination_v1/radial_mehler.py)
- [code/astra_phase2_reexamination_v1/tensor_clamp_min_v1.py](code/astra_phase2_reexamination_v1/tensor_clamp_min_v1.py)
- [code/astra_phase2_reexamination_v1/train_joint_prefix_final_v1.py](code/astra_phase2_reexamination_v1/train_joint_prefix_final_v1.py)
- [code/astra_phase2_reexamination_v1/verify_joint_prefix_final_provenance_v1.py](code/astra_phase2_reexamination_v1/verify_joint_prefix_final_provenance_v1.py)
- [code/codex_resume_20260926_v2/hill_climb_common_energy_capacity_v1/inspect.py](code/codex_resume_20260926_v2/hill_climb_common_energy_capacity_v1/inspect.py)

</details>

## Evidence

- [JOINT_PREFIX_FINAL_FIT_V1_CLOSED.md#L1](JOINT_PREFIX_FINAL_FIT_V1_CLOSED.md#L1)
- [JOINT_PREFIX_FINAL_FIT_V1_CLOSED.md#L14](JOINT_PREFIX_FINAL_FIT_V1_CLOSED.md#L14)
- [JOINT_PREFIX_FINAL_FIT_V1_CLOSED.md#L53](JOINT_PREFIX_FINAL_FIT_V1_CLOSED.md#L53)

<details>
<summary>All result files and records (7)</summary>

- [JOINT_PREFIX_FINAL_FIT_V1_CLOSED.md](JOINT_PREFIX_FINAL_FIT_V1_CLOSED.md) · record
- [receipt.json](receipt.json) · result
- [joint_prefix_final_terminal_integrity_v1.json](joint_prefix_final_terminal_integrity_v1.json) · result
- [receipt-1b885d19.json](receipt-1b885d19.json) · result
- [receipt-ec83d811.json](receipt-ec83d811.json) · result
- [receipt-49686105.json](receipt-49686105.json) · result
- [JOINT_PREFIX_FINAL_FIT_PROTOCOL_V1.md](JOINT_PREFIX_FINAL_FIT_PROTOCOL_V1.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-06-joint-prefix-actual-final-head-fit-closed) · [Copy provenance](../../../PROVENANCE.json)

