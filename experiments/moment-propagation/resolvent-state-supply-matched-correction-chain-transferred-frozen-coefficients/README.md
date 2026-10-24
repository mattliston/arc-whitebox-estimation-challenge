# Resolvent state supply matched the correction chain and transferred with frozen coefficients

2026-09-01 · experiment · rejected · measured

The first new raw mechanism since the certified chain: resolvent post-state + certified Wick assembly minus the engine on matched stats; per-layer corr 0.16-0.26 = exact parity with the chain's own per-layer windows; 3-layer stack +1.1% raw in-sample (r35), +0.6% with FROZEN gammas on r100. Value-hump law (solo fourth-cumulant Edgeworth operator (L4) +0.5/L5 +0.3/L8 +0.2/L11 ~0/L2 null); identification a/M-split degeneracy decoded (4th-order pressure steers it; a-share steerable 0.11<->0.65).

Recorded measurement:

    The first new raw mechanism since the certified chain: resolvent post-state + certified Wick assembly minus the engine on matched stats; per-layer corr 0.16-0.26 = exact parity with the chain's own per-layer windows; 3-layer stack +1.1% raw in-sample (r35), +0.6% with FROZEN gammas on r100. Value-hump law (solo L4 +0.5/L5 +0.3/L8 +0.2/L11 ~0/L2 null); identification a/M-split degeneracy decoded (4th-order pressure steers it; a-share steerable 0.11<->0.65).

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

- [G3_ENGINE_DELTA_rank0035_L4.json#L1](G3_ENGINE_DELTA_rank0035_L4.json#L1)
- [G3_ENGINE_DELTA_rank0100_L8.json#L1](G3_ENGINE_DELTA_rank0100_L8.json#L1)
- [G3_ENGINE_DELTA_rank0035_L8.json#L1](G3_ENGINE_DELTA_rank0035_L8.json#L1)
- [G3_ENGINE_DELTA_rank0035_L5.json#L1](G3_ENGINE_DELTA_rank0035_L5.json#L1)
- [G3_ENGINE_DELTA_rank0100_L5.json#L1](G3_ENGINE_DELTA_rank0100_L5.json#L1)
- [G3_ENGINE_DELTA_rank0035_L6.json#L1](G3_ENGINE_DELTA_rank0035_L6.json#L1)
- [G3_ENGINE_DELTA_rank0035_L11.json#L1](G3_ENGINE_DELTA_rank0035_L11.json#L1)
- [LANE_h2_h3_endpoint_adjoint.md#L1](../../fourth-order-terms/chaos-state-engines-scalar-cumulant-resolvent-identities-certified-two/LANE_h2_h3_endpoint_adjoint.md#L1)

<details>
<summary>All result files and records (8)</summary>

- [G3_ENGINE_DELTA_rank0035_L4.json](G3_ENGINE_DELTA_rank0035_L4.json) · result
- [G3_ENGINE_DELTA_rank0100_L8.json](G3_ENGINE_DELTA_rank0100_L8.json) · result
- [G3_ENGINE_DELTA_rank0035_L8.json](G3_ENGINE_DELTA_rank0035_L8.json) · result
- [G3_ENGINE_DELTA_rank0035_L5.json](G3_ENGINE_DELTA_rank0035_L5.json) · result
- [G3_ENGINE_DELTA_rank0100_L5.json](G3_ENGINE_DELTA_rank0100_L5.json) · result
- [G3_ENGINE_DELTA_rank0035_L6.json](G3_ENGINE_DELTA_rank0035_L6.json) · result
- [G3_ENGINE_DELTA_rank0035_L11.json](G3_ENGINE_DELTA_rank0035_L11.json) · result
- [LANE_h2_h3_endpoint_adjoint.md](../../fourth-order-terms/chaos-state-engines-scalar-cumulant-resolvent-identities-certified-two/LANE_h2_h3_endpoint_adjoint.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-01-resolvent-state-supply-matched-correction-chain-transferred-frozen-coefficients) · [Copy provenance](../../../PROVENANCE.json)

