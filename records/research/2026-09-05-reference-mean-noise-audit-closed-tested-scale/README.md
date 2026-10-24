# Reference-mean noise audit: closed at the tested scale

2026-09-05T19:32Z · finding · informational · measured

The two unchanged teacher-forced baselines are byte-identical. Under the explicit iid Gaussian-input N=1e9 reference-mean noise model, eight fixed second-moment probes estimate layers 10–15 noise/residual mean squared error (MSE) fractions of 0.0032950 on rank 35 and 0.0026949 on rank 100 (0.330% and 0.269%). This is far below the predeclared 10%-on-both closing threshold.

Recorded measurement:

    The two unchanged teacher-forced baselines are byte-identical. Under the explicit iid Gaussian-input N=1e9 reference-mean noise model, eight fixed second-moment probes estimate layers 10–15 noise/residual MSE fractions of 0.0032950 on rank 35 and 0.0026949 on rank 100 (0.330% and 0.269%). Final-layer modeled noise MSE is 1.20606e-11 / 1.01182e-11, compared with frozen-head teacher residual MSE 5.43848e-9 / 5.45859e-9, respectively. Mean late nonlinear-even response MSE is 2.72237e-13 / 5.87799e-13. Each two-sided eight-direction audit took approximately 110 seconds. This is far below the predeclared 10%-on-both closing threshold. It does not support changed teacher targets, denoising, more noise probes, or a claim that reference noise explains the failed feature probes. Actual sampler provenance remains unverified: the stored files have no sample count or sampler metadata. These are model-based sensitivity numbers, not measured reference error or a universal noise floor.

## Programs

- [../2026-09-05-audit-possible-amplification-reference-mean-sampling-noise/audit_teacher_reference_noise_v1.py](../2026-09-05-audit-possible-amplification-reference-mean-sampling-noise/audit_teacher_reference_noise_v1.py)
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

- [TEACHER_REFERENCE_NOISE_AUDIT_V1_CLOSED.md#L1](../2026-09-05-audit-possible-amplification-reference-mean-sampling-noise/TEACHER_REFERENCE_NOISE_AUDIT_V1_CLOSED.md#L1)
- [TEACHER_REFERENCE_NOISE_AUDIT_V1_CLOSED.md#L8](../2026-09-05-audit-possible-amplification-reference-mean-sampling-noise/TEACHER_REFERENCE_NOISE_AUDIT_V1_CLOSED.md#L8)
- [TEACHER_REFERENCE_NOISE_AUDIT_V1_CLOSED.md#L12](../2026-09-05-audit-possible-amplification-reference-mean-sampling-noise/TEACHER_REFERENCE_NOISE_AUDIT_V1_CLOSED.md#L12)

<details>
<summary>All result files and records (4)</summary>

- [TEACHER_REFERENCE_NOISE_AUDIT_V1_CLOSED.md](../2026-09-05-audit-possible-amplification-reference-mean-sampling-noise/TEACHER_REFERENCE_NOISE_AUDIT_V1_CLOSED.md) · record
- [teacher_reference_noise_r35_v1.json](teacher_reference_noise_r35_v1.json) · result
- [teacher_reference_noise_r100_v1.json](teacher_reference_noise_r100_v1.json) · result
- [TEACHER_REFERENCE_NOISE_AUDIT_V1.md](../2026-09-05-audit-possible-amplification-reference-mean-sampling-noise/TEACHER_REFERENCE_NOISE_AUDIT_V1.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-05-reference-mean-noise-audit-closed-tested-scale) · [Copy provenance](../../../PROVENANCE.json)

