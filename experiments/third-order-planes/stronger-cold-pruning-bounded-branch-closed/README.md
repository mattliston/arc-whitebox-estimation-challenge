# Stronger cold pruning: bounded branch closed

2026-09-05T01:34Z · experiment · informational · measured

- Gate.03:6.39236e-8 /6.33339e-8; exact next-layer covariance/third-moment diagonal shadows improve to5.78605e-8 /5.82215e-8, still far worse. - Gate.05:1.43146e-7 /1.74671e-7; with shadows1.33226e-7 /1.16228e-7. These results do not justify a port or matched retraining of this form.

Recorded measurement:

    - Gate.03:6.39236e-8 /6.33339e-8; exact next-layer covariance/third-moment diagonal shadows improve to5.78605e-8 /5.82215e-8, still far worse. - Gate.05:1.43146e-7 /1.74671e-7; with shadows1.33226e-7 /1.16228e-7.

Program source is in [`code/`](code/), preserving the relative layout below `.`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/estimator.py](code/estimator.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_phase2_reexamination_v1/build_compact_scalar_final_v1.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_phase2_reexamination_v1/build_compact_scalar_final_v1.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_phase2_reexamination_v1/build_fullpair_scalar_final_v1.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_phase2_reexamination_v1/build_fullpair_scalar_final_v1.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_phase2_reexamination_v1/candidate_binary_compact_g01_learned_d6_v1.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_phase2_reexamination_v1/candidate_binary_compact_g01_learned_d6_v1.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_phase2_reexamination_v1/candidate_binary_compact_g01_learned_direct_v1.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_phase2_reexamination_v1/candidate_binary_compact_g01_learned_direct_v1.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_phase2_reexamination_v1/candidate_compact_radialfeed_unlearned_d6_v1.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_phase2_reexamination_v1/candidate_compact_radialfeed_unlearned_d6_v1.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_phase2_reexamination_v1/candidate_compact_scalar_final_d6_v1.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_phase2_reexamination_v1/candidate_compact_scalar_final_d6_v1.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_phase2_reexamination_v1/chain_compact_final_head_v1.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_phase2_reexamination_v1/chain_compact_final_head_v1.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_phase2_reexamination_v1/chain_compact_radialfeed_training_v1.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_phase2_reexamination_v1/chain_compact_radialfeed_training_v1.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_phase2_reexamination_v1/chain_radial_training.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_phase2_reexamination_v1/chain_radial_training.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_phase2_reexamination_v1/compact_final_head_v1.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_phase2_reexamination_v1/compact_final_head_v1.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_phase2_reexamination_v1/compact_final_reference_v1.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_phase2_reexamination_v1/compact_final_reference_v1.py)

<details>
<summary>Additional programs and dependencies (22)</summary>

- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_phase2_reexamination_v1/compact_fullpair_final_reference_v1.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_phase2_reexamination_v1/compact_fullpair_final_reference_v1.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_phase2_reexamination_v1/compact_scalar_final_v1.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_phase2_reexamination_v1/compact_scalar_final_v1.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_phase2_reexamination_v1/generate_compact_final_rows_v1.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_phase2_reexamination_v1/generate_compact_final_rows_v1.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_phase2_reexamination_v1/generate_compact_radialfeed_rows_v1.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_phase2_reexamination_v1/generate_compact_radialfeed_rows_v1.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_phase2_reexamination_v1/generate_radial_rows.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_phase2_reexamination_v1/generate_radial_rows.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_phase2_reexamination_v1/k3_radial_numpy.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_phase2_reexamination_v1/k3_radial_numpy.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_phase2_reexamination_v1/radial_binary_compact.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_phase2_reexamination_v1/radial_binary_compact.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_phase2_reexamination_v1/radial_binary_compact_asymmetric_v1.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_phase2_reexamination_v1/radial_binary_compact_asymmetric_v1.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_phase2_reexamination_v1/radial_compact_feed_fullpair_v1.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_phase2_reexamination_v1/radial_compact_feed_fullpair_v1.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_phase2_reexamination_v1/radial_compact_radialfeed_v1.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_phase2_reexamination_v1/radial_compact_radialfeed_v1.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_phase2_reexamination_v1/radial_cubic.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_phase2_reexamination_v1/radial_cubic.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_phase2_reexamination_v1/radial_features.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_phase2_reexamination_v1/radial_features.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_phase2_reexamination_v1/radial_gnn_numpy.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_phase2_reexamination_v1/radial_gnn_numpy.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_phase2_reexamination_v1/radial_mehler.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_phase2_reexamination_v1/radial_mehler.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_phase2_reexamination_v1/screen_binary_asymmetric_v1.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_phase2_reexamination_v1/screen_binary_asymmetric_v1.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_phase2_reexamination_v1/screen_compact_final_v1.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_phase2_reexamination_v1/screen_compact_final_v1.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_phase2_reexamination_v1/screen_fullpair_final_v1.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_phase2_reexamination_v1/screen_fullpair_final_v1.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_phase2_reexamination_v1/train_compact_final_head_v1.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_phase2_reexamination_v1/train_compact_final_head_v1.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_phase2_reexamination_v1/train_fullpair_final_head_v1.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_phase2_reexamination_v1/train_fullpair_final_head_v1.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260926_v2/hill_climb_common_energy_capacity_v1/inspect.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/codex_resume_20260926_v2/hill_climb_common_energy_capacity_v1/inspect.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_g22_operator_features_v1/export_gnn_payload.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_g22_operator_features_v1/export_gnn_payload.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_g22_operator_features_v1/train_gnn_oracle.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_g22_operator_features_v1/train_gnn_oracle.py)

</details>

## Evidence

- [RESEARCH_RECEIPTS_2026_09_04_EVENING.md#L43](../../arithmetic-kernels/compact-closure-cost-tests-factor-thinning-contraction-scheduling/RESEARCH_RECEIPTS_2026_09_04_EVENING.md#L43)
- [RESEARCH_RECEIPTS_2026_09_04_EVENING.md#L50](../../arithmetic-kernels/compact-closure-cost-tests-factor-thinning-contraction-scheduling/RESEARCH_RECEIPTS_2026_09_04_EVENING.md#L50)

<details>
<summary>All result files and records (2)</summary>

- [RESEARCH_RECEIPTS_2026_09_04_EVENING.md](../../arithmetic-kernels/compact-closure-cost-tests-factor-thinning-contraction-scheduling/RESEARCH_RECEIPTS_2026_09_04_EVENING.md) · record
- [binary_asymmetric_v1.json](binary_asymmetric_v1.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-09-05-stronger-cold-pruning-bounded-branch-closed) · [Copy provenance](../../../PROVENANCE.json)

