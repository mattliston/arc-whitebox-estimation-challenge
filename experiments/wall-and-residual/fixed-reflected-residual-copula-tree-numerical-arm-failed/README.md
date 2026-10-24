# Fixed reflected residual-copula tree: numerical arm failed

2026-09-06T23:22Z · experiment · rejected · measured

The separate observation-only capture passed all original-bank prefix and final byte checks, and every post0..12 physical third-cumulant repeated-index plane (K21) normalized difference was0. It directly captured actual post C/b/B and next Bpre/bpre, without symmetrization of bank state. This is a failed fixed degree12 numerical arm, not evidence that an exact copula is infeasible or that the tree has poor/good raw accuracy.

Recorded measurement:

    The separate observation-only capture passed all original-bank prefix and final byte checks, and every post0..12 physical K21 normalized difference was0. It directly captured actual post C/b/B and next Bpre/bpre, without symmetrization of bank state. Capture wall7.815979292s; peakRSS5,080,186,880 bytes, below6GiB. Two threads were requested; empty backend introspection is not a proof of the effective count. Receipt SHA dfe88ab975fdae184c7636ca197f04e136d6c64734f75840435e8cbfcc6e02fe. Causal NPZ SHA 09c00424bfccd50715f77f01181d7b7786bbdb75b22f870d29e1579c163b78a5. - Existing post12 support:832 live,192 cold. The auxiliary symmetrized live covariance passed strict FP64 Cholesky without repair. Relative change from actual live C was9.90255307e-7. Seventy-one cold variances were negative; they were reported and excluded only by the original bank mask. - Trace-subtracted marginal standardized skew ranged -0.00889901 to 28.81963369, with one reflection and no exactly Gaussian coordinates. All832 marginals fit the fixed [-8,12] bracket. Recovered alpha ranged -2.61610839 to3.07780381. Maximum relative scalar skew/variance residuals were2.41170e-13 /4.39806e-16, passing the prescribed numerical checks. - The degree12 Hermite covariance polynomial failed its endpoint sign bracket on810 of345,696 unordered pairs. Its retained marginal variance fraction ranged .9546654985 to .9999906324. The judge stopped at this first pair-inverse gate; it did not choose alternative roots or orders. - Exact endpoint/Price-quadrature checks, latent R Cholesky, tree/star action, predictor seal and the <=.03 teacher-fidelity test were NOT run. No numerical or source-fidelity success is inferred for those stages. Judge numerical wall.353734125s; peakRSS269,697,024 bytes, below2GiB; process exit1 and passed=false. The result and marginal diagnostic artifact remain in reflected_residual_copula_tree_information_v1. No files were overwritten or repaired and no retry was launched. This is a failed fixed degree12 numerical arm, not evidence that an exact copula is infeasible or that the tree has poor/good raw accuracy. Its prospective no-repair/no-order-sweep stop is binding for this arm. Changing the inverse after seeing these failures is not authorized as a hidden technical replay. The official score bank remains unchanged at4.98223e-9; the less-than1e-9 goal has not been achieved and is not externally blocked.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/astra_phase2_reexamination_v1/candidate_fullpair_runtime_rectd4_t8192_bulkcopies_v1.py](code/astra_phase2_reexamination_v1/candidate_fullpair_runtime_rectd4_t8192_bulkcopies_v1.py)
- [code/astra_phase2_reexamination_v1/capture_reflected_residual_copula_trace_v1.py](code/astra_phase2_reexamination_v1/capture_reflected_residual_copula_trace_v1.py)
- [code/astra_phase2_reexamination_v1/capture_two_memory_teacher_action_replay_v2.py](code/astra_phase2_reexamination_v1/capture_two_memory_teacher_action_replay_v2.py)
- [code/astra_phase2_reexamination_v1/capture_two_memory_teacher_action_v1.py](code/astra_phase2_reexamination_v1/capture_two_memory_teacher_action_v1.py)
- [code/astra_phase2_reexamination_v1/certify_reflected_copula_tree_action_v1.py](code/astra_phase2_reexamination_v1/certify_reflected_copula_tree_action_v1.py)
- [code/astra_phase2_reexamination_v1/certify_reflected_relu_marginal_v1.py](code/astra_phase2_reexamination_v1/certify_reflected_relu_marginal_v1.py)
- [code/astra_phase2_reexamination_v1/check_covariance_metric_completion_wide_v1.py](code/astra_phase2_reexamination_v1/check_covariance_metric_completion_wide_v1.py)
- [code/astra_phase2_reexamination_v1/compact_final_head_v1.py](code/astra_phase2_reexamination_v1/compact_final_head_v1.py)
- [code/astra_phase2_reexamination_v1/compact_fullpair_final_reference_v1.py](code/astra_phase2_reexamination_v1/compact_fullpair_final_reference_v1.py)
- [code/astra_phase2_reexamination_v1/compact_scalar_final_v1.py](code/astra_phase2_reexamination_v1/compact_scalar_final_v1.py)
- [code/astra_phase2_reexamination_v1/covariance_metric_completion_v1.py](code/astra_phase2_reexamination_v1/covariance_metric_completion_v1.py)
- [code/astra_phase2_reexamination_v1/judge_reflected_residual_copula_tree_v1.py](code/astra_phase2_reexamination_v1/judge_reflected_residual_copula_tree_v1.py)

<details>
<summary>Additional programs and dependencies (11)</summary>

- [code/astra_phase2_reexamination_v1/k3_radial_numpy.py](code/astra_phase2_reexamination_v1/k3_radial_numpy.py)
- [code/astra_phase2_reexamination_v1/pair_memory_action_physics_v1.py](code/astra_phase2_reexamination_v1/pair_memory_action_physics_v1.py)
- [code/astra_phase2_reexamination_v1/qualify_two_memory_capture_replay_v2.py](code/astra_phase2_reexamination_v1/qualify_two_memory_capture_replay_v2.py)
- [code/astra_phase2_reexamination_v1/radial_binary_compact.py](code/astra_phase2_reexamination_v1/radial_binary_compact.py)
- [code/astra_phase2_reexamination_v1/radial_compact_feed_fullpair_v1.py](code/astra_phase2_reexamination_v1/radial_compact_feed_fullpair_v1.py)
- [code/astra_phase2_reexamination_v1/radial_compact_radialfeed_v1.py](code/astra_phase2_reexamination_v1/radial_compact_radialfeed_v1.py)
- [code/astra_phase2_reexamination_v1/radial_cubic.py](code/astra_phase2_reexamination_v1/radial_cubic.py)
- [code/astra_phase2_reexamination_v1/radial_features.py](code/astra_phase2_reexamination_v1/radial_features.py)
- [code/astra_phase2_reexamination_v1/radial_gnn_numpy.py](code/astra_phase2_reexamination_v1/radial_gnn_numpy.py)
- [code/astra_phase2_reexamination_v1/radial_mehler.py](code/astra_phase2_reexamination_v1/radial_mehler.py)
- [code/codex_resume_20260926_v2/hill_climb_common_energy_capacity_v1/inspect.py](code/codex_resume_20260926_v2/hill_climb_common_energy_capacity_v1/inspect.py)

</details>

## Evidence

- [REFLECTED_RESIDUAL_COPULA_TREE_V1_CLOSED.md#L1](REFLECTED_RESIDUAL_COPULA_TREE_V1_CLOSED.md#L1)
- [REFLECTED_RESIDUAL_COPULA_TREE_V1_CLOSED.md#L34](REFLECTED_RESIDUAL_COPULA_TREE_V1_CLOSED.md#L34)

<details>
<summary>All result files and records (2)</summary>

- [REFLECTED_RESIDUAL_COPULA_TREE_V1_CLOSED.md](REFLECTED_RESIDUAL_COPULA_TREE_V1_CLOSED.md) · record
- [REFLECTED_RESIDUAL_COPULA_TREE_PROTOCOL_V1.md](../../../records/research/2026-09-06-reflected-residual-copula-tree-fixed-root-protocol-accepted-numerical/REFLECTED_RESIDUAL_COPULA_TREE_PROTOCOL_V1.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-06-fixed-reflected-residual-copula-tree-numerical-arm-failed) · [Copy provenance](../../../PROVENANCE.json)

