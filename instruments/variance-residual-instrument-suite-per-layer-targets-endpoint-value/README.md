# Variance-residual instrument suite with per-layer targets and endpoint value tests

2026-09-01 · tool · informational · measured

The supervised target system for the variance channel: R-fields (truth-post-cov minus closure on card state) banked per layer for both nets at shared-draw 1e-5 truth; oracle depth curve priced (full v-channel = raw 1.25-1.46e-8; +4-6 pts/layer beyond fourth-cumulant Edgeworth operator (L4)); value metric = pushed-v; W-conventions nailed; truth pre-planes located (mps_planes l>=2); third-cumulant repeated-index plane (K21)-Frobenius shown to be a null instrument at N=4e5.

Recorded measurement:

    The supervised target system for the variance channel: R-fields (truth-post-cov minus closure on card state) banked per layer for both nets at shared-draw 1e-5 truth; oracle depth curve priced (full v-channel = raw 1.25-1.46e-8; +4-6 pts/layer beyond L4); value metric = pushed-v; W-conventions nailed; truth pre-planes located (mps_planes l>=2); K21-Frobenius shown to be a null instrument at N=4e5.

## Programs

- [code/p2_width1024_depth16/public_g19_attribution_v1/residue_analysis.py](code/p2_width1024_depth16/public_g19_attribution_v1/residue_analysis.py)
- [code/p2_width1024_depth16/public_g19_attribution_v1/run_attribution.py](code/p2_width1024_depth16/public_g19_attribution_v1/run_attribution.py)
- [code/p2_width1024_depth16/public_g19_attribution_v1/run_c1_adjoint_rank.py](code/p2_width1024_depth16/public_g19_attribution_v1/run_c1_adjoint_rank.py)
- [code/p2_width1024_depth16/public_g19_attribution_v1/run_c4v1_closed_form.py](code/p2_width1024_depth16/public_g19_attribution_v1/run_c4v1_closed_form.py)
- [code/p2_width1024_depth16/public_g19_attribution_v1/run_chaos1024_shallow.py](code/p2_width1024_depth16/public_g19_attribution_v1/run_chaos1024_shallow.py)
- [code/p2_width1024_depth16/public_g19_attribution_v1/run_chaos_supply_shallow.py](code/p2_width1024_depth16/public_g19_attribution_v1/run_chaos_supply_shallow.py)
- [code/p2_width1024_depth16/public_g19_attribution_v1/run_collective_patch.py](code/p2_width1024_depth16/public_g19_attribution_v1/run_collective_patch.py)
- [code/p2_width1024_depth16/public_g19_attribution_v1/run_engine1024_probe.py](code/p2_width1024_depth16/public_g19_attribution_v1/run_engine1024_probe.py)
- [code/p2_width1024_depth16/public_g19_attribution_v1/run_g3_dv1024.py](code/p2_width1024_depth16/public_g19_attribution_v1/run_g3_dv1024.py)
- [code/p2_width1024_depth16/public_g19_attribution_v1/run_g3_engine_delta.py](code/p2_width1024_depth16/public_g19_attribution_v1/run_g3_engine_delta.py)
- [code/p2_width1024_depth16/public_g19_attribution_v1/run_g3_exact_shallow_delta.py](code/p2_width1024_depth16/public_g19_attribution_v1/run_g3_exact_shallow_delta.py)
- [code/p2_width1024_depth16/public_g19_attribution_v1/run_gram_split.py](code/p2_width1024_depth16/public_g19_attribution_v1/run_gram_split.py)

<details>
<summary>Additional programs and dependencies (20)</summary>

- [code/p2_width1024_depth16/public_g19_attribution_v1/run_gram_verdict.py](code/p2_width1024_depth16/public_g19_attribution_v1/run_gram_verdict.py)
- [code/p2_width1024_depth16/public_g19_attribution_v1/run_k3push_hermite.py](code/p2_width1024_depth16/public_g19_attribution_v1/run_k3push_hermite.py)
- [code/p2_width1024_depth16/public_g19_attribution_v1/run_ncqc_corefit.py](code/p2_width1024_depth16/public_g19_attribution_v1/run_ncqc_corefit.py)
- [code/p2_width1024_depth16/public_g19_attribution_v1/run_pair_exact_probe.py](code/p2_width1024_depth16/public_g19_attribution_v1/run_pair_exact_probe.py)
- [code/p2_width1024_depth16/public_g19_attribution_v1/run_perfect_ceiling_m.py](code/p2_width1024_depth16/public_g19_attribution_v1/run_perfect_ceiling_m.py)
- [code/p2_width1024_depth16/public_g19_attribution_v1/run_plane_recursion.py](code/p2_width1024_depth16/public_g19_attribution_v1/run_plane_recursion.py)
- [code/p2_width1024_depth16/public_g19_attribution_v1/run_tangent_v2.py](code/p2_width1024_depth16/public_g19_attribution_v1/run_tangent_v2.py)
- [code/p2_width1024_depth16/public_g19_attribution_v1/run_tensor_recursion_v4.py](code/p2_width1024_depth16/public_g19_attribution_v1/run_tensor_recursion_v4.py)
- [code/p2_width1024_depth16/public_g19_attribution_v1/run_vresid_field.py](code/p2_width1024_depth16/public_g19_attribution_v1/run_vresid_field.py)
- [code/p2_width1024_depth16/public_g19_attribution_v1/valdecomp.py](code/p2_width1024_depth16/public_g19_attribution_v1/valdecomp.py)
- [code/p2_width1024_depth16/public_g19_math_acceleration_v1/exact_recurrence_v3.py](code/p2_width1024_depth16/public_g19_math_acceleration_v1/exact_recurrence_v3.py)
- [code/p2_width1024_depth16/public_g19_math_acceleration_v1/run_chaos_ceiling.py](code/p2_width1024_depth16/public_g19_math_acceleration_v1/run_chaos_ceiling.py)
- [code/p2_width1024_depth16/public_g19_math_acceleration_v1/run_g3_birth_words.py](code/p2_width1024_depth16/public_g19_math_acceleration_v1/run_g3_birth_words.py)
- [code/p2_width1024_depth16/public_g19_math_acceleration_v1/run_g3_exact_emission.py](code/p2_width1024_depth16/public_g19_math_acceleration_v1/run_g3_exact_emission.py)
- [code/p2_width1024_depth16/public_g19_math_acceleration_v1/run_g3_transport.py](code/p2_width1024_depth16/public_g19_math_acceleration_v1/run_g3_transport.py)
- [code/p2_width1024_depth16/public_g19_math_acceleration_v1/run_g3_value_tiny.py](code/p2_width1024_depth16/public_g19_math_acceleration_v1/run_g3_value_tiny.py)
- [code/p2_width1024_depth16/public_g19_math_acceleration_v1/run_shared_rank.py](code/p2_width1024_depth16/public_g19_math_acceleration_v1/run_shared_rank.py)
- [code/p2_width1024_depth16/public_g19_math_acceleration_v1/run_word_closure.py](code/p2_width1024_depth16/public_g19_math_acceleration_v1/run_word_closure.py)
- [code/p2_width1024_depth16/public_q24_native_transport_energy_final_workspace_v4/estimator.py](code/p2_width1024_depth16/public_q24_native_transport_energy_final_workspace_v4/estimator.py)
- [code/x3_r0c1_kappa3_ceiling/tensor_crosscheck_v1/second_order_triple.py](code/x3_r0c1_kappa3_ceiling/tensor_crosscheck_v1/second_order_triple.py)

</details>

## Evidence

- [VRESID_FIELD_rank0035.json#L1](VRESID_FIELD_rank0035.json#L1)
- [VRESID_FIELD_rank0100.json#L1](VRESID_FIELD_rank0100.json#L1)
- [VRESID_FIELD_rank0035_so2.json#L1](VRESID_FIELD_rank0035_so2.json#L1)
- [LANE_h2_h3_endpoint_adjoint.md#L1](../../experiments/fourth-order-terms/chaos-state-engines-scalar-cumulant-resolvent-identities-certified-two/LANE_h2_h3_endpoint_adjoint.md#L1)

<details>
<summary>All result files and records (4)</summary>

- [VRESID_FIELD_rank0035.json](VRESID_FIELD_rank0035.json) · result
- [VRESID_FIELD_rank0100.json](VRESID_FIELD_rank0100.json) · result
- [VRESID_FIELD_rank0035_so2.json](VRESID_FIELD_rank0035_so2.json) · result
- [LANE_h2_h3_endpoint_adjoint.md](../../experiments/fourth-order-terms/chaos-state-engines-scalar-cumulant-resolvent-identities-certified-two/LANE_h2_h3_endpoint_adjoint.md) · record

</details>

[Research graph](../../graph/index.html#node=2026-09-01-variance-residual-instrument-suite-per-layer-targets-endpoint-value) · [Copy provenance](../../PROVENANCE.json)

