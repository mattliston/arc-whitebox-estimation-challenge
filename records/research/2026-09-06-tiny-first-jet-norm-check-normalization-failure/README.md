# Tiny first-jet: norm-check normalization failure

2026-09-06T19:31Z · incident · failed · measured

Fixtures0-6 and8 passed every gate. All nine tensor, repeated-plane and next-layer star-restored identity checks pass the unchanged1e-11 relative-to-reference gate; maximum tensor identity error4.0483541151e-13. Proposed v2 diagnostic change, pending root approval: report both norm normalizations, retain the original norm-error-to-T gate for every eligible pair, and check the norm of huge, ineligible errors relative to that error's own norm.

Recorded measurement:

    Fixtures0-6 and8 passed every gate. All nine tensor, repeated-plane and next-layer star-restored identity checks pass the unchanged1e-11 relative-to-reference gate; maximum tensor identity error4.0483541151e-13. Fixture7 deliberately has pair difference norm1e-5 and transverse norm.1. Its solved cube is a very BAD approximation: ||E||/||T||=2462.335006. It would unambiguously fail the fixed <=.1 eligibility criterion. The Frobenius error-norm formula's floating-point discrepancy, when normalized by the much smaller ||T||^2, is2.1875847410e-9 and fails1e-11. Relative to ||E||^2 this discrepancy is about3.61e-16, i.e. ordinary floating-point roundoff. The tensor identity on that fixture still passes at4.05e-13 relative to T.

## Programs

- [../../../experiments/wall-and-residual/repaired-native-metal-implementation-forward-local-vector-jacobian-product/code/astra_phase2_reexamination_v1/certify_pair_jet_coalescence_v1.py](../../../experiments/wall-and-residual/repaired-native-metal-implementation-forward-local-vector-jacobian-product/code/astra_phase2_reexamination_v1/certify_pair_jet_coalescence_v1.py)
- [../../../submissions/329926/estimator.py](../../../submissions/329926/estimator.py)
- [../../../experiments/wall-and-residual/repaired-native-metal-implementation-forward-local-vector-jacobian-product/code/astra_phase2_reexamination_v1/capture_pair_jet_opportunity_stage_a_v1.py](../../../experiments/wall-and-residual/repaired-native-metal-implementation-forward-local-vector-jacobian-product/code/astra_phase2_reexamination_v1/capture_pair_jet_opportunity_stage_a_v1.py)
- [../../../experiments/wall-and-residual/full-pair-third-cumulant-closure-rebuilt-independently-compared-submitted/code/astra_phase2_reexamination_v1/centered_bank_autodiff_v1.py](../../../experiments/wall-and-residual/full-pair-third-cumulant-closure-rebuilt-independently-compared-submitted/code/astra_phase2_reexamination_v1/centered_bank_autodiff_v1.py)
- [../../../experiments/wall-and-residual/full-pair-third-cumulant-closure-rebuilt-independently-compared-submitted/code/astra_phase2_reexamination_v1/centered_bank_autodiff_v3.py](../../../experiments/wall-and-residual/full-pair-third-cumulant-closure-rebuilt-independently-compared-submitted/code/astra_phase2_reexamination_v1/centered_bank_autodiff_v3.py)
- [../../../experiments/wall-and-residual/full-pair-third-cumulant-closure-rebuilt-independently-compared-submitted/code/astra_phase2_reexamination_v1/centered_torch_numerics_v1.py](../../../experiments/wall-and-residual/full-pair-third-cumulant-closure-rebuilt-independently-compared-submitted/code/astra_phase2_reexamination_v1/centered_torch_numerics_v1.py)
- [../../../experiments/wall-and-residual/full-pair-third-cumulant-closure-rebuilt-independently-compared-submitted/code/astra_phase2_reexamination_v1/centered_torch_numerics_v3.py](../../../experiments/wall-and-residual/full-pair-third-cumulant-closure-rebuilt-independently-compared-submitted/code/astra_phase2_reexamination_v1/centered_torch_numerics_v3.py)
- [../../../experiments/wall-and-residual/full-pair-third-cumulant-closure-rebuilt-independently-compared-submitted/code/astra_phase2_reexamination_v1/current_transport_shadow_v1.py](../../../experiments/wall-and-residual/full-pair-third-cumulant-closure-rebuilt-independently-compared-submitted/code/astra_phase2_reexamination_v1/current_transport_shadow_v1.py)
- [../../../experiments/wall-and-residual/full-pair-third-cumulant-closure-rebuilt-independently-compared-submitted/code/astra_phase2_reexamination_v1/current_transport_shadow_v3.py](../../../experiments/wall-and-residual/full-pair-third-cumulant-closure-rebuilt-independently-compared-submitted/code/astra_phase2_reexamination_v1/current_transport_shadow_v3.py)
- [../../../experiments/wall-and-residual/full-pair-third-cumulant-closure-rebuilt-independently-compared-submitted/code/astra_phase2_reexamination_v1/full_carrier_autodiff_v1.py](../../../experiments/wall-and-residual/full-pair-third-cumulant-closure-rebuilt-independently-compared-submitted/code/astra_phase2_reexamination_v1/full_carrier_autodiff_v1.py)
- [../../../experiments/wall-and-residual/full-pair-third-cumulant-closure-rebuilt-independently-compared-submitted/code/astra_phase2_reexamination_v1/full_carrier_bank_autodiff_v1.py](../../../experiments/wall-and-residual/full-pair-third-cumulant-closure-rebuilt-independently-compared-submitted/code/astra_phase2_reexamination_v1/full_carrier_bank_autodiff_v1.py)
- [../../../experiments/wall-and-residual/full-pair-third-cumulant-closure-rebuilt-independently-compared-submitted/code/astra_phase2_reexamination_v1/gaussian_cdf_quadrature_v1.py](../../../experiments/wall-and-residual/full-pair-third-cumulant-closure-rebuilt-independently-compared-submitted/code/astra_phase2_reexamination_v1/gaussian_cdf_quadrature_v1.py)

<details>
<summary>Additional programs and dependencies (2)</summary>

- [../../../experiments/wall-and-residual/full-pair-third-cumulant-closure-rebuilt-independently-compared-submitted/code/astra_phase2_reexamination_v1/tensor_clamp_min_v1.py](../../../experiments/wall-and-residual/full-pair-third-cumulant-closure-rebuilt-independently-compared-submitted/code/astra_phase2_reexamination_v1/tensor_clamp_min_v1.py)
- [../../../experiments/learned-correction/global-context-exact-weight-suffix-state-0-102505-gain/code/codex_resume_20260926_v2/hill_climb_common_energy_capacity_v1/inspect.py](../../../experiments/learned-correction/global-context-exact-weight-suffix-state-0-102505-gain/code/codex_resume_20260926_v2/hill_climb_common_energy_capacity_v1/inspect.py)

</details>

## Evidence

- [PAIR_JET_TINY_V1_NORMALIZATION_FAILURE.md#L1](PAIR_JET_TINY_V1_NORMALIZATION_FAILURE.md#L1)
- [PAIR_JET_TINY_V1_NORMALIZATION_FAILURE.md#L13](PAIR_JET_TINY_V1_NORMALIZATION_FAILURE.md#L13)

<details>
<summary>All result files and records (1)</summary>

- [PAIR_JET_TINY_V1_NORMALIZATION_FAILURE.md](PAIR_JET_TINY_V1_NORMALIZATION_FAILURE.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-06-tiny-first-jet-norm-check-normalization-failure) · [Copy provenance](../../../PROVENANCE.json)

