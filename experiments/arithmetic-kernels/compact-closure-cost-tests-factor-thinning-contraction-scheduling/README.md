# Compact closure cost tests: factor thinning and contraction scheduling

2026-09-05T01:34Z · experiment · informational · measured

chain_compact_final_head_v1.py runs three nice10 CPU row workers then one MPS fixed40epoch final-only fit,160/40 same net split. Every final input is captured before current correction; no truth feedback anywhere.

Recorded measurement:

    1. Audited direct-write D6/4096 rectangular products with exact zero padding for active counts that are only multiples of32. All masks and learned parameters frozen. candidate_binary_compact_g01_learned_d6_v1.py: native rank35 raw2.2937990e-8, utilization.222519884 versus D5~.2307. Actual client/server resource check running at this entry. No full panel verdict yet for this exact version. 2. Scalar-only final read. Layers1..14 and their learned correction remain frozen. Refit only the final head on its own prefix's states, using scalar features plus the existing recurrent messages. This permits omitting the final full CP plane, off-diagonal covariance output, and star plane reads. Not a reopened iterative DAgger program: the older program's flat result was checked in T-116j/T-117e before this cost-motivated design. chain_compact_final_head_v1.py runs three nice10 CPU row workers then one MPS fixed40epoch final-only fit,160/40 same net split. Every final input is captured before current correction; no truth feedback anywhere. Targets are read only after all inputs and recurrent states have been frozen. Live exec session24634; inspect its handle before acting. Reference and FlopScope builders are versioned separately for independent parity checks.

Program source is in [`code/`](code/), preserving the relative layout below `.`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_phase2_reexamination_v1/candidate_binary_compact_g01_learned_d6_v1.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_phase2_reexamination_v1/candidate_binary_compact_g01_learned_d6_v1.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_phase2_reexamination_v1/chain_compact_final_head_v1.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_phase2_reexamination_v1/chain_compact_final_head_v1.py)
- [code/estimator.py](code/estimator.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_phase2_reexamination_v1/build_compact_radialfeed_v1.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_phase2_reexamination_v1/build_compact_radialfeed_v1.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_phase2_reexamination_v1/build_compact_scalar_final_v1.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_phase2_reexamination_v1/build_compact_scalar_final_v1.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_phase2_reexamination_v1/build_compact_tensorfeat_v1.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_phase2_reexamination_v1/build_compact_tensorfeat_v1.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_phase2_reexamination_v1/build_fullpair_scalar_final_v1.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_phase2_reexamination_v1/build_fullpair_scalar_final_v1.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_phase2_reexamination_v1/candidate_binary_compact_g01_learned_direct_v1.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_phase2_reexamination_v1/candidate_binary_compact_g01_learned_direct_v1.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_phase2_reexamination_v1/candidate_compact_radialfeed_unlearned_d6_v1.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_phase2_reexamination_v1/candidate_compact_radialfeed_unlearned_d6_v1.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_phase2_reexamination_v1/candidate_compact_scalar_final_d6_v1.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_phase2_reexamination_v1/candidate_compact_scalar_final_d6_v1.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_phase2_reexamination_v1/chain_compact_radialfeed_training_v1.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_phase2_reexamination_v1/chain_compact_radialfeed_training_v1.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_phase2_reexamination_v1/chain_radial_training.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_phase2_reexamination_v1/chain_radial_training.py)

<details>
<summary>Additional programs and dependencies (25)</summary>

- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_phase2_reexamination_v1/compact_final_head_v1.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_phase2_reexamination_v1/compact_final_head_v1.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_phase2_reexamination_v1/compact_final_reference_v1.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_phase2_reexamination_v1/compact_final_reference_v1.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_phase2_reexamination_v1/compact_fullpair_final_reference_v1.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_phase2_reexamination_v1/compact_fullpair_final_reference_v1.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_phase2_reexamination_v1/compact_scalar_final_v1.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_phase2_reexamination_v1/compact_scalar_final_v1.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_phase2_reexamination_v1/compact_tensor_directions_v1.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_phase2_reexamination_v1/compact_tensor_directions_v1.py)
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

- [RESEARCH_RECEIPTS_2026_09_04_EVENING.md#L56](RESEARCH_RECEIPTS_2026_09_04_EVENING.md#L56)
- [RESEARCH_RECEIPTS_2026_09_04_EVENING.md#L1](RESEARCH_RECEIPTS_2026_09_04_EVENING.md#L1)

<details>
<summary>All result files and records (1)</summary>

- [RESEARCH_RECEIPTS_2026_09_04_EVENING.md](RESEARCH_RECEIPTS_2026_09_04_EVENING.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-05-compact-closure-cost-tests-factor-thinning-contraction-scheduling) · [Copy provenance](../../../PROVENANCE.json)

