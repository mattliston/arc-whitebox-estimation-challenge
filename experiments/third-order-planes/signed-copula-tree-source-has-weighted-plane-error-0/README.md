# Signed copula-tree source has weighted plane error 0.33959289170093016 versus the .03 gate

2026-09-08 · experiment · rejected · measured

The fixed inverse and tensor-action fixtures passed, but approximate endpoint clipping and substantial source discrepancy prevented a full rollout.

Recorded measurement:

    VERDICT_2026_09_08_PDT.md:
    standardized target discrepancy0.001778370798694; aggregate relative
    covariance residual6.410132609855e-5. These are numerical summaries, not
    0,32,...,992, weighted relative error is0.33959289170093016; unweighted
    gate fails. This is a proxy, NOT current409f error, final raw or adjusted

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_phase2_reexamination_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/signed_copula_tree_action_v1/judge.py](code/signed_copula_tree_action_v1/judge.py)
- [code/candidate_fullpair_runtime_rectd4_t8192_bulkcopies_v1.py](code/candidate_fullpair_runtime_rectd4_t8192_bulkcopies_v1.py)
- [code/capture_reflected_residual_copula_trace_v1.py](code/capture_reflected_residual_copula_trace_v1.py)
- [code/capture_two_memory_teacher_action_replay_v2.py](code/capture_two_memory_teacher_action_replay_v2.py)
- [code/capture_two_memory_teacher_action_v1.py](code/capture_two_memory_teacher_action_v1.py)
- [code/certify_reflected_copula_tree_action_v1.py](code/certify_reflected_copula_tree_action_v1.py)
- [code/certify_reflected_relu_marginal_v1.py](code/certify_reflected_relu_marginal_v1.py)
- [code/compact_final_head_v1.py](code/compact_final_head_v1.py)
- [code/compact_fullpair_final_reference_v1.py](code/compact_fullpair_final_reference_v1.py)
- [code/compact_scalar_final_v1.py](code/compact_scalar_final_v1.py)
- [code/current_response_scalar_certificate_v1/scalar_interval.py](code/current_response_scalar_certificate_v1/scalar_interval.py)
- [code/judge_reflected_residual_copula_tree_v1.py](code/judge_reflected_residual_copula_tree_v1.py)

<details>
<summary>Additional programs and dependencies (10)</summary>

- [code/k3_radial_numpy.py](code/k3_radial_numpy.py)
- [code/pair_memory_action_physics_v1.py](code/pair_memory_action_physics_v1.py)
- [code/qualify_two_memory_capture_replay_v2.py](code/qualify_two_memory_capture_replay_v2.py)
- [code/radial_binary_compact.py](code/radial_binary_compact.py)
- [code/radial_compact_feed_fullpair_v1.py](code/radial_compact_feed_fullpair_v1.py)
- [code/radial_compact_radialfeed_v1.py](code/radial_compact_radialfeed_v1.py)
- [code/radial_cubic.py](code/radial_cubic.py)
- [code/radial_features.py](code/radial_features.py)
- [code/radial_gnn_numpy.py](code/radial_gnn_numpy.py)
- [code/radial_mehler.py](code/radial_mehler.py)

</details>

## Evidence

- [VERDICT_2026_09_08_PDT.md#L3](VERDICT_2026_09_08_PDT.md#L3)
- [VERDICT_2026_09_08_PDT.md#L11](VERDICT_2026_09_08_PDT.md#L11)
- [VERDICT_2026_09_08_PDT.md#L12](VERDICT_2026_09_08_PDT.md#L12)
- [VERDICT_2026_09_08_PDT.md#L18](VERDICT_2026_09_08_PDT.md#L18)
- [VERDICT_2026_09_08_PDT.md#L20](VERDICT_2026_09_08_PDT.md#L20)

<details>
<summary>All result files and records (2)</summary>

- [VERDICT_2026_09_08_PDT.md](VERDICT_2026_09_08_PDT.md) · record
- [PROTOCOL.md](PROTOCOL.md) · config

</details>

[Research graph](../../../graph/index.html#node=2026-09-08-signed-copula-tree-source-has-weighted-plane-error-0) · [Copy provenance](../../../PROVENANCE.json)

