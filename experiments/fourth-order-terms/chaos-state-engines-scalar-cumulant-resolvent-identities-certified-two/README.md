# Chaos-state engines: scalar, cumulant and resolvent identities certified on two networks

2026-09-01 · experiment · rejected · measured

The chaos-order campaign: required state order = 4 (ceiling ladder, errors saturate); Q2 PROVEN (cumulant endpoints are Jordan-word Grams, machine-exact); three numerics laws certified (CF denominator factor-2; kappa4 16x convention; trapezoid endpoints ; the last closed the engine beta deficit 0.988->0.9994); resolvent-exact emission (no Stein ladder) with certified Wick assemblies K21post (1,1,1,1/2,1) and fourth-cumulant two-pair plane (K22)/fourth-cumulant triple-single plane (K31) integer formulas; engine layer certified on rank35+rank100 at 0.1%-class, parameter-free.

Recorded measurement:

    The chaos-order campaign: required state order = 4 (ceiling ladder, errors saturate); Q2 PROVEN (cumulant endpoints are Jordan-word Grams, machine-exact); three numerics laws certified (CF denominator factor-2; kappa4 16x convention; trapezoid endpoints — the last closed the engine beta deficit 0.988->0.9994); resolvent-exact emission (no Stein ladder) with certified Wick assemblies K21post (1,1,1,1/2,1) and K22/K31 integer formulas; engine layer certified on rank35+rank100 at 0.1%-class, parameter-free.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

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

- [LANE_h2_h3_endpoint_adjoint.md#L1298](LANE_h2_h3_endpoint_adjoint.md#L1298)
- [ENGINE1024_PROBE_rank0035.json#L5](ENGINE1024_PROBE_rank0035.json#L5)
- [ENGINE1024_PROBE_rank0100.json#L9](ENGINE1024_PROBE_rank0100.json#L9)

<details>
<summary>All result files and records (3)</summary>

- [LANE_h2_h3_endpoint_adjoint.md](LANE_h2_h3_endpoint_adjoint.md) · record
- [ENGINE1024_PROBE_rank0035.json](ENGINE1024_PROBE_rank0035.json) · result
- [ENGINE1024_PROBE_rank0100.json](ENGINE1024_PROBE_rank0100.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-09-01-chaos-state-engines-scalar-cumulant-resolvent-identities-certified-two) · [Copy provenance](../../../PROVENANCE.json)

