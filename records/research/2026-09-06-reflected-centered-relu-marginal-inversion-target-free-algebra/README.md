# Reflected centered-ReLU marginal inversion: target-free algebra

2026-09-06T22:57Z · finding · informational · measured

Let Y=(Z+alpha)+, Z standard normal, and let m_j=E[Y^j], v=m_2-m_1^2, k=m_3-3*m_1*m_2+2*m_1^3, gamma=k/v^(3/2). Write g=Phi(alpha), Q=1-g. certify_reflected_relu_marginal_v1.py compares these moments with independently integrated positive tail moments at ten fixed alpha values from -8 to 12, checks the exact derivative identity, and recovers seven fixed synthetic alpha values by 64 bisection steps in [-8,12].

Recorded measurement:

    Let Y=(Z+alpha)+, Z standard normal, and let m_j=E[Y^j], v=m_2-m_1^2, k=m_3-3*m_1*m_2+2*m_1^3, gamma=k/v^(3/2). Write g=Phi(alpha), Q=1-g. Differentiation under the integral gives m_1'=g, m_2'=2*m_1, m_3'=3*m_2, v'=2*m_1*Q, k'=3*Q*(v-m_1^2), gamma'=3*Q*(m_2^2-m_1*m_3)/v^(5/2) < 0. The strict sign follows from Cauchy–Schwarz applied to Y^(1/2) and Y^(3/2); conditional on Y>0 the distribution is not a point mass. As alpha tends to plus infinity gamma tends to zero from above. As alpha tends to minus infinity, the positive tail moments are asymptotic to j!*phi(|alpha|)/|alpha|^(j+1), so gamma tends to infinity. Hence every strictly positive skew has a unique finite alpha. A finite numerical bracket is an implementation domain, not a restriction of this theorem. certify_reflected_relu_marginal_v1.py compares these moments with independently integrated positive tail moments at ten fixed alpha values from -8 to 12, checks the exact derivative identity, and recovers seven fixed synthetic alpha values by 64 bisection steps in [-8,12]. Relative moment tolerance1e-8; alpha recovery tolerance1e-10. The script is inert without --execute and refuses to overwrite its receipt. No real network, target array, fit or production timing is included.

## Programs

- [../2026-09-06-additive-directed-correlation-response-correction/certify_reflected_relu_marginal_v1.py](../2026-09-06-additive-directed-correlation-response-correction/certify_reflected_relu_marginal_v1.py)
- [../../../submissions/329926/estimator.py](../../../submissions/329926/estimator.py)
- [../2026-09-06-additive-directed-correlation-response-correction/capture_reflected_residual_copula_trace_v1.py](../2026-09-06-additive-directed-correlation-response-correction/capture_reflected_residual_copula_trace_v1.py)
- [../../../experiments/wall-and-residual/full-pair-third-cumulant-closure-rebuilt-independently-compared-submitted/code/astra_phase2_reexamination_v1/capture_two_memory_teacher_action_replay_v2.py](../../../experiments/wall-and-residual/full-pair-third-cumulant-closure-rebuilt-independently-compared-submitted/code/astra_phase2_reexamination_v1/capture_two_memory_teacher_action_replay_v2.py)
- [../../../experiments/wall-and-residual/full-pair-third-cumulant-closure-rebuilt-independently-compared-submitted/code/astra_phase2_reexamination_v1/capture_two_memory_teacher_action_v1.py](../../../experiments/wall-and-residual/full-pair-third-cumulant-closure-rebuilt-independently-compared-submitted/code/astra_phase2_reexamination_v1/capture_two_memory_teacher_action_v1.py)
- [../2026-09-06-additive-directed-correlation-response-correction/certify_reflected_copula_tree_action_v1.py](../2026-09-06-additive-directed-correlation-response-correction/certify_reflected_copula_tree_action_v1.py)
- [../../../experiments/wall-and-residual/full-pair-third-cumulant-closure-rebuilt-independently-compared-submitted/code/astra_phase2_reexamination_v1/compact_final_head_v1.py](../../../experiments/wall-and-residual/full-pair-third-cumulant-closure-rebuilt-independently-compared-submitted/code/astra_phase2_reexamination_v1/compact_final_head_v1.py)
- [../../../experiments/wall-and-residual/full-pair-third-cumulant-closure-rebuilt-independently-compared-submitted/code/astra_phase2_reexamination_v1/compact_fullpair_final_reference_v1.py](../../../experiments/wall-and-residual/full-pair-third-cumulant-closure-rebuilt-independently-compared-submitted/code/astra_phase2_reexamination_v1/compact_fullpair_final_reference_v1.py)
- [../../../experiments/wall-and-residual/full-pair-third-cumulant-closure-rebuilt-independently-compared-submitted/code/astra_phase2_reexamination_v1/compact_scalar_final_v1.py](../../../experiments/wall-and-residual/full-pair-third-cumulant-closure-rebuilt-independently-compared-submitted/code/astra_phase2_reexamination_v1/compact_scalar_final_v1.py)
- [../2026-09-06-additive-directed-correlation-response-correction/judge_reflected_residual_copula_tree_v1.py](../2026-09-06-additive-directed-correlation-response-correction/judge_reflected_residual_copula_tree_v1.py)
- [../../../experiments/wall-and-residual/exact-kernel-resource-tradeoff/code/astra_phase2_reexamination_v1/k3_radial_numpy.py](../../../experiments/wall-and-residual/exact-kernel-resource-tradeoff/code/astra_phase2_reexamination_v1/k3_radial_numpy.py)
- [../../../experiments/wall-and-residual/full-pair-third-cumulant-closure-rebuilt-independently-compared-submitted/code/astra_phase2_reexamination_v1/pair_memory_action_physics_v1.py](../../../experiments/wall-and-residual/full-pair-third-cumulant-closure-rebuilt-independently-compared-submitted/code/astra_phase2_reexamination_v1/pair_memory_action_physics_v1.py)

<details>
<summary>Additional programs and dependencies (9)</summary>

- [../../../experiments/wall-and-residual/full-pair-third-cumulant-closure-rebuilt-independently-compared-submitted/code/astra_phase2_reexamination_v1/qualify_two_memory_capture_replay_v2.py](../../../experiments/wall-and-residual/full-pair-third-cumulant-closure-rebuilt-independently-compared-submitted/code/astra_phase2_reexamination_v1/qualify_two_memory_capture_replay_v2.py)
- [../../../experiments/wall-and-residual/full-pair-third-cumulant-closure-rebuilt-independently-compared-submitted/code/astra_phase2_reexamination_v1/radial_binary_compact.py](../../../experiments/wall-and-residual/full-pair-third-cumulant-closure-rebuilt-independently-compared-submitted/code/astra_phase2_reexamination_v1/radial_binary_compact.py)
- [../../../experiments/wall-and-residual/full-pair-third-cumulant-closure-rebuilt-independently-compared-submitted/code/astra_phase2_reexamination_v1/radial_compact_feed_fullpair_v1.py](../../../experiments/wall-and-residual/full-pair-third-cumulant-closure-rebuilt-independently-compared-submitted/code/astra_phase2_reexamination_v1/radial_compact_feed_fullpair_v1.py)
- [../../../experiments/wall-and-residual/full-pair-third-cumulant-closure-rebuilt-independently-compared-submitted/code/astra_phase2_reexamination_v1/radial_compact_radialfeed_v1.py](../../../experiments/wall-and-residual/full-pair-third-cumulant-closure-rebuilt-independently-compared-submitted/code/astra_phase2_reexamination_v1/radial_compact_radialfeed_v1.py)
- [../../../experiments/wall-and-residual/exact-kernel-resource-tradeoff/code/astra_phase2_reexamination_v1/radial_cubic.py](../../../experiments/wall-and-residual/exact-kernel-resource-tradeoff/code/astra_phase2_reexamination_v1/radial_cubic.py)
- [../../../experiments/wall-and-residual/exact-kernel-resource-tradeoff/code/astra_phase2_reexamination_v1/radial_features.py](../../../experiments/wall-and-residual/exact-kernel-resource-tradeoff/code/astra_phase2_reexamination_v1/radial_features.py)
- [../../../experiments/wall-and-residual/full-pair-third-cumulant-closure-rebuilt-independently-compared-submitted/code/astra_phase2_reexamination_v1/radial_gnn_numpy.py](../../../experiments/wall-and-residual/full-pair-third-cumulant-closure-rebuilt-independently-compared-submitted/code/astra_phase2_reexamination_v1/radial_gnn_numpy.py)
- [../../../experiments/wall-and-residual/full-pair-third-cumulant-closure-rebuilt-independently-compared-submitted/code/astra_phase2_reexamination_v1/radial_mehler.py](../../../experiments/wall-and-residual/full-pair-third-cumulant-closure-rebuilt-independently-compared-submitted/code/astra_phase2_reexamination_v1/radial_mehler.py)
- [../../../experiments/learned-correction/global-context-exact-weight-suffix-state-0-102505-gain/code/codex_resume_20260926_v2/hill_climb_common_energy_capacity_v1/inspect.py](../../../experiments/learned-correction/global-context-exact-weight-suffix-state-0-102505-gain/code/codex_resume_20260926_v2/hill_climb_common_energy_capacity_v1/inspect.py)

</details>

## Evidence

- [REFLECTED_RELU_SKEW_INVERSION_ALGEBRA_V1.md#L1](REFLECTED_RELU_SKEW_INVERSION_ALGEBRA_V1.md#L1)

<details>
<summary>All result files and records (2)</summary>

- [REFLECTED_RELU_SKEW_INVERSION_ALGEBRA_V1.md](REFLECTED_RELU_SKEW_INVERSION_ALGEBRA_V1.md) · record
- [reflected_copula_tree_action_certificate_v1.json](reflected_copula_tree_action_certificate_v1.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-09-06-reflected-centered-relu-marginal-inversion-target-free-algebra) · [Copy provenance](../../../PROVENANCE.json)

