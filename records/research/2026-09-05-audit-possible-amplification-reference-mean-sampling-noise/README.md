# Audit possible amplification of reference-mean sampling noise

2026-09-05T19:25Z · finding · informational · measured

Several honest teacher-feature probes leave nearly the same~4.8e-9 residual. Before treating that as entirely missing physical information, check whether noisy reference means fed back through covariance contribute appreciably to the teacher residual.

Recorded measurement:

    Several honest teacher-feature probes leave nearly the same~4.8e-9 residual. Before treating that as entirely missing physical information, check whether noisy reference means fed back through covariance contribute appreciably to the teacher residual. This is a data-quality sensitivity audit, NOT reopening sampling/quadrature as an estimator. Ranks35/100 only;8 fixed noise directions each; two nice10 workers with 3BLAS threads. No new training fit, labels, sealed data, sampling estimator, or actual-card parameter changes. Compare layers10..15 mean squared noise response with the existing honest frozen-head teacher residual MSE. If the model estimates<10% on both, close this confounder at that scale; if >20% on both, verify actual label sampler provenance and response linearity before contemplating any changed training protocol. Ambiguous results do not justify claiming that stored truths are bad or noise is the raw floor.

## Programs

- [audit_teacher_reference_noise_v1.py](audit_teacher_reference_noise_v1.py)
- [../../../experiments/wall-and-residual/exact-kernel-resource-tradeoff/code/astra_phase2_reexamination_v1/k3_radial_numpy.py](../../../experiments/wall-and-residual/exact-kernel-resource-tradeoff/code/astra_phase2_reexamination_v1/k3_radial_numpy.py)
- [../../../experiments/wall-and-residual/full-pair-third-cumulant-closure-rebuilt-independently-compared-submitted/code/astra_phase2_reexamination_v1/radial_binary_compact.py](../../../experiments/wall-and-residual/full-pair-third-cumulant-closure-rebuilt-independently-compared-submitted/code/astra_phase2_reexamination_v1/radial_binary_compact.py)
- [../../../experiments/wall-and-residual/full-pair-third-cumulant-closure-rebuilt-independently-compared-submitted/code/astra_phase2_reexamination_v1/radial_compact_feed_fullpair_v1.py](../../../experiments/wall-and-residual/full-pair-third-cumulant-closure-rebuilt-independently-compared-submitted/code/astra_phase2_reexamination_v1/radial_compact_feed_fullpair_v1.py)
- [../../../experiments/wall-and-residual/full-pair-third-cumulant-closure-rebuilt-independently-compared-submitted/code/astra_phase2_reexamination_v1/radial_compact_radialfeed_v1.py](../../../experiments/wall-and-residual/full-pair-third-cumulant-closure-rebuilt-independently-compared-submitted/code/astra_phase2_reexamination_v1/radial_compact_radialfeed_v1.py)
- [../../../experiments/wall-and-residual/exact-kernel-resource-tradeoff/code/astra_phase2_reexamination_v1/radial_cubic.py](../../../experiments/wall-and-residual/exact-kernel-resource-tradeoff/code/astra_phase2_reexamination_v1/radial_cubic.py)
- [../../../experiments/wall-and-residual/exact-kernel-resource-tradeoff/code/astra_phase2_reexamination_v1/radial_features.py](../../../experiments/wall-and-residual/exact-kernel-resource-tradeoff/code/astra_phase2_reexamination_v1/radial_features.py)
- [../../../experiments/wall-and-residual/full-pair-third-cumulant-closure-rebuilt-independently-compared-submitted/code/astra_phase2_reexamination_v1/radial_gnn_numpy.py](../../../experiments/wall-and-residual/full-pair-third-cumulant-closure-rebuilt-independently-compared-submitted/code/astra_phase2_reexamination_v1/radial_gnn_numpy.py)
- [../../../experiments/wall-and-residual/full-pair-third-cumulant-closure-rebuilt-independently-compared-submitted/code/astra_phase2_reexamination_v1/radial_mehler.py](../../../experiments/wall-and-residual/full-pair-third-cumulant-closure-rebuilt-independently-compared-submitted/code/astra_phase2_reexamination_v1/radial_mehler.py)
- [../../../experiments/learned-correction/global-context-exact-weight-suffix-state-0-102505-gain/code/codex_resume_20260926_v2/hill_climb_common_energy_capacity_v1/inspect.py](../../../experiments/learned-correction/global-context-exact-weight-suffix-state-0-102505-gain/code/codex_resume_20260926_v2/hill_climb_common_energy_capacity_v1/inspect.py)

## Evidence

- [TEACHER_REFERENCE_NOISE_AUDIT_V1.md#L1](TEACHER_REFERENCE_NOISE_AUDIT_V1.md#L1)
- [TEACHER_REFERENCE_NOISE_AUDIT_V1.md#L5](TEACHER_REFERENCE_NOISE_AUDIT_V1.md#L5)

<details>
<summary>All result files and records (2)</summary>

- [TEACHER_REFERENCE_NOISE_AUDIT_V1.md](TEACHER_REFERENCE_NOISE_AUDIT_V1.md) · record
- [TEACHER_REFERENCE_NOISE_AUDIT_V1_CLOSED.md](TEACHER_REFERENCE_NOISE_AUDIT_V1_CLOSED.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-05-audit-possible-amplification-reference-mean-sampling-noise) · [Copy provenance](../../../PROVENANCE.json)

