# Adaptive Gaussian mixture literature/code audit

2026-09-06T09:39Z · finding · informational · measured

The released full-network propagation uses2q+1 unscented points per Gaussian component in a q-dimensional covariance subspace, evaluates both the network and a linearization, and splits components using a Wasserstein- based criterion. The example permits500 components, with subspace disabled.

Recorded measurement:

    The released full-network propagation uses2q+1 unscented points per Gaussian component in a q-dimensional covariance subspace, evaluates both the network and a linearization, and splits components using a Wasserstein- based criterion. The example permits500 components, with subspace disabled. For this task's full-rank1024-dimensional Gaussian, one such component therefore uses2049 full-network points before other work. This is an input quadrature route, not a demonstrated inexpensive analytic conditional moment recursion. No published1024x16/1e-9 mean-MSE or metered-cost result was identified in these sources. A naive transplant has no demonstrated advantage over the already unsuccessful input rules; do not run it merely because it is called a Gaussian mixture. This is NOT an impossibility result for all mixtures. Full paper PDF was not read in this audit. The prior Zhang/Shin2021 publisher abstract and introduction concern adaptive splitting/reduction and likewise do not establish challenge-scale precision or cost. https://doi.org/10.1016/j.neucom.2021.06.007

## Programs

- [build_shared_newborn_push_v1.py](build_shared_newborn_push_v1.py)
- [../../../experiments/wall-and-residual/full-pair-third-cumulant-closure-rebuilt-independently-compared-submitted/code/astra_phase2_reexamination_v1/radial_binary_cap.py](../../../experiments/wall-and-residual/full-pair-third-cumulant-closure-rebuilt-independently-compared-submitted/code/astra_phase2_reexamination_v1/radial_binary_cap.py)
- [build_binary_paircap_fnp.py](build_binary_paircap_fnp.py)
- [candidate_binary_radial_v1.py](candidate_binary_radial_v1.py)
- [../../../experiments/wall-and-residual/full-pair-third-cumulant-closure-rebuilt-independently-compared-submitted/code/astra_phase2_reexamination_v1/chain_binary_cap_training_v1.py](../../../experiments/wall-and-residual/full-pair-third-cumulant-closure-rebuilt-independently-compared-submitted/code/astra_phase2_reexamination_v1/chain_binary_cap_training_v1.py)
- [../../../experiments/wall-and-residual/exact-kernel-resource-tradeoff/code/astra_phase2_reexamination_v1/chain_radial_training.py](../../../experiments/wall-and-residual/exact-kernel-resource-tradeoff/code/astra_phase2_reexamination_v1/chain_radial_training.py)
- [../../../experiments/wall-and-residual/full-pair-third-cumulant-closure-rebuilt-independently-compared-submitted/code/astra_phase2_reexamination_v1/generate_binary_cap_rows.py](../../../experiments/wall-and-residual/full-pair-third-cumulant-closure-rebuilt-independently-compared-submitted/code/astra_phase2_reexamination_v1/generate_binary_cap_rows.py)
- [../../../experiments/wall-and-residual/exact-kernel-resource-tradeoff/code/astra_phase2_reexamination_v1/generate_radial_rows.py](../../../experiments/wall-and-residual/exact-kernel-resource-tradeoff/code/astra_phase2_reexamination_v1/generate_radial_rows.py)
- [../../../experiments/wall-and-residual/exact-kernel-resource-tradeoff/code/astra_phase2_reexamination_v1/k3_radial_numpy.py](../../../experiments/wall-and-residual/exact-kernel-resource-tradeoff/code/astra_phase2_reexamination_v1/k3_radial_numpy.py)
- [../../../experiments/wall-and-residual/exact-kernel-resource-tradeoff/code/astra_phase2_reexamination_v1/radial_binary_cubic.py](../../../experiments/wall-and-residual/exact-kernel-resource-tradeoff/code/astra_phase2_reexamination_v1/radial_binary_cubic.py)
- [../../../experiments/wall-and-residual/exact-kernel-resource-tradeoff/code/astra_phase2_reexamination_v1/radial_cubic.py](../../../experiments/wall-and-residual/exact-kernel-resource-tradeoff/code/astra_phase2_reexamination_v1/radial_cubic.py)
- [../../../experiments/wall-and-residual/exact-kernel-resource-tradeoff/code/astra_phase2_reexamination_v1/radial_features.py](../../../experiments/wall-and-residual/exact-kernel-resource-tradeoff/code/astra_phase2_reexamination_v1/radial_features.py)

<details>
<summary>Additional programs and dependencies (3)</summary>

- [../../../experiments/learned-correction/global-context-exact-weight-suffix-state-0-102505-gain/code/codex_resume_20260926_v2/hill_climb_common_energy_capacity_v1/inspect.py](../../../experiments/learned-correction/global-context-exact-weight-suffix-state-0-102505-gain/code/codex_resume_20260926_v2/hill_climb_common_energy_capacity_v1/inspect.py)
- [../../../experiments/error-attribution/diagonal-third-cumulant-closure-failed-despite-trained-correction/code/public_g22_operator_features_v1/export_gnn_payload.py](../../../experiments/error-attribution/diagonal-third-cumulant-closure-failed-despite-trained-correction/code/public_g22_operator_features_v1/export_gnn_payload.py)
- [../../../instruments/oracle-recursion-row-generator-online-payload-exporter-were-mechanically/code/public_g22_operator_features_v1/train_gnn_oracle.py](../../../instruments/oracle-recursion-row-generator-online-payload-exporter-were-mechanically/code/public_g22_operator_features_v1/train_gnn_oracle.py)

</details>

## Evidence

- [WGMPROP_READ_ONLY_AUDIT_V1.md#L1](WGMPROP_READ_ONLY_AUDIT_V1.md#L1)
- [WGMPROP_READ_ONLY_AUDIT_V1.md#L27](WGMPROP_READ_ONLY_AUDIT_V1.md#L27)

<details>
<summary>All result files and records (1)</summary>

- [WGMPROP_READ_ONLY_AUDIT_V1.md](WGMPROP_READ_ONLY_AUDIT_V1.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-06-adaptive-gaussian-mixture-literature-code-audit) · [Copy provenance](../../../PROVENANCE.json)

