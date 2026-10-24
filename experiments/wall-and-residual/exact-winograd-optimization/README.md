# Exact Winograd optimization

2026-09-05T00:52Z · experiment · informational · measured

Actual numpy-free client/server, full learned D6/4096: - New candidate candidate_binary_learned_d6_direct_v1.py. - Utilization .298482952 versus .312744994, a4.56% saving. - Wall65.78s under concurrent load, residual.286452587s (passes.4s). - Output SHA256 1aaefe9b72de8b60f8aacb2e776daa566cae57bf13f68b4456de9022b7e500ad is IDENTICAL to the old D6/4096 remote output. - Source SHA256 31777abfd8b0b12b0bb192960d74513c1be8e085ab296e01bee1dca7b262ac6d. - No14-net run or new peak-resident-set size (RSS) audit for this exact version yet.

Recorded measurement:

    Actual numpy-free client/server, full learned D6/4096: - New candidate candidate_binary_learned_d6_direct_v1.py. - Utilization .298482952 versus .312744994, a4.56% saving. - Wall65.78s under concurrent load, residual.286452587s (passes.4s). - Output SHA256 1aaefe9b72de8b60f8aacb2e776daa566cae57bf13f68b4456de9022b7e500ad is IDENTICAL to the old D6/4096 remote output. - Source SHA256 31777abfd8b0b12b0bb192960d74513c1be8e085ab296e01bee1dca7b262ac6d. - No14-net run or new peak-RSS audit for this exact version yet. Composed with compact-neuron .01, unlearned D5: candidate_binary_compact_g01_unlearned_direct_v1.py, rank35 raw2.93826267e-8 EXACTLY unchanged, utilization.228172300 versus.233989185. Not a submission by itself; it needs a matched learned head and all resource/panel gates.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/astra_phase2_reexamination_v1/build_direct_transform_v1.py](code/astra_phase2_reexamination_v1/build_direct_transform_v1.py)
- [code/astra_phase2_reexamination_v1/candidate_binary_learned_d6_direct_v1.py](code/astra_phase2_reexamination_v1/candidate_binary_learned_d6_direct_v1.py)
- [code/astra_phase2_reexamination_v1/candidate_binary_compact_g01_unlearned_direct_v1.py](code/astra_phase2_reexamination_v1/candidate_binary_compact_g01_unlearned_direct_v1.py)
- [code/astra_phase2_reexamination_v1/build_compact_d6_v1.py](code/astra_phase2_reexamination_v1/build_compact_d6_v1.py)
- [code/astra_phase2_reexamination_v1/candidate_binary_compact_g01_unlearned_v2.py](code/astra_phase2_reexamination_v1/candidate_binary_compact_g01_unlearned_v2.py)
- [code/astra_phase2_reexamination_v1/candidate_binary_learned_d6_v1.py](code/astra_phase2_reexamination_v1/candidate_binary_learned_d6_v1.py)
- [code/astra_phase2_reexamination_v1/chain_binary_compact_training_v1.py](code/astra_phase2_reexamination_v1/chain_binary_compact_training_v1.py)
- [code/astra_phase2_reexamination_v1/chain_compact_final_head_v1.py](code/astra_phase2_reexamination_v1/chain_compact_final_head_v1.py)
- [code/astra_phase2_reexamination_v1/chain_radial_training.py](code/astra_phase2_reexamination_v1/chain_radial_training.py)
- [code/astra_phase2_reexamination_v1/check_direct_transform_v1.py](code/astra_phase2_reexamination_v1/check_direct_transform_v1.py)
- [code/astra_phase2_reexamination_v1/check_packed_deep_kernel.py](code/astra_phase2_reexamination_v1/check_packed_deep_kernel.py)
- [code/astra_phase2_reexamination_v1/compact_final_head_v1.py](code/astra_phase2_reexamination_v1/compact_final_head_v1.py)

<details>
<summary>Additional programs and dependencies (26)</summary>

- [code/astra_phase2_reexamination_v1/compact_final_reference_v1.py](code/astra_phase2_reexamination_v1/compact_final_reference_v1.py)
- [code/astra_phase2_reexamination_v1/compact_fullpair_final_reference_v1.py](code/astra_phase2_reexamination_v1/compact_fullpair_final_reference_v1.py)
- [code/astra_phase2_reexamination_v1/compact_scalar_final_v1.py](code/astra_phase2_reexamination_v1/compact_scalar_final_v1.py)
- [code/astra_phase2_reexamination_v1/generate_binary_compact_rows.py](code/astra_phase2_reexamination_v1/generate_binary_compact_rows.py)
- [code/astra_phase2_reexamination_v1/generate_compact_final_rows_v1.py](code/astra_phase2_reexamination_v1/generate_compact_final_rows_v1.py)
- [code/astra_phase2_reexamination_v1/generate_fullpair_final_rows_v1.py](code/astra_phase2_reexamination_v1/generate_fullpair_final_rows_v1.py)
- [code/astra_phase2_reexamination_v1/generate_radial_rows.py](code/astra_phase2_reexamination_v1/generate_radial_rows.py)
- [code/astra_phase2_reexamination_v1/k3_radial_numpy.py](code/astra_phase2_reexamination_v1/k3_radial_numpy.py)
- [code/astra_phase2_reexamination_v1/packed_arena_generic.py](code/astra_phase2_reexamination_v1/packed_arena_generic.py)
- [code/astra_phase2_reexamination_v1/radial_binary_compact.py](code/astra_phase2_reexamination_v1/radial_binary_compact.py)
- [code/astra_phase2_reexamination_v1/radial_binary_cubic.py](code/astra_phase2_reexamination_v1/radial_binary_cubic.py)
- [code/astra_phase2_reexamination_v1/radial_binary_gateprune.py](code/astra_phase2_reexamination_v1/radial_binary_gateprune.py)
- [code/astra_phase2_reexamination_v1/radial_binary_sparsecarry.py](code/astra_phase2_reexamination_v1/radial_binary_sparsecarry.py)
- [code/astra_phase2_reexamination_v1/radial_compact_feed_fullpair_v1.py](code/astra_phase2_reexamination_v1/radial_compact_feed_fullpair_v1.py)
- [code/astra_phase2_reexamination_v1/radial_compact_radialfeed_v1.py](code/astra_phase2_reexamination_v1/radial_compact_radialfeed_v1.py)
- [code/astra_phase2_reexamination_v1/radial_cubic.py](code/astra_phase2_reexamination_v1/radial_cubic.py)
- [code/astra_phase2_reexamination_v1/radial_features.py](code/astra_phase2_reexamination_v1/radial_features.py)
- [code/astra_phase2_reexamination_v1/radial_gnn_numpy.py](code/astra_phase2_reexamination_v1/radial_gnn_numpy.py)
- [code/astra_phase2_reexamination_v1/radial_mehler.py](code/astra_phase2_reexamination_v1/radial_mehler.py)
- [code/astra_phase2_reexamination_v1/screen_compact_final_v1.py](code/astra_phase2_reexamination_v1/screen_compact_final_v1.py)
- [code/astra_phase2_reexamination_v1/screen_fullpair_final_v1.py](code/astra_phase2_reexamination_v1/screen_fullpair_final_v1.py)
- [code/astra_phase2_reexamination_v1/train_compact_final_head_v1.py](code/astra_phase2_reexamination_v1/train_compact_final_head_v1.py)
- [code/astra_phase2_reexamination_v1/train_fullpair_final_head_v1.py](code/astra_phase2_reexamination_v1/train_fullpair_final_head_v1.py)
- [code/codex_resume_20260926_v2/hill_climb_common_energy_capacity_v1/inspect.py](code/codex_resume_20260926_v2/hill_climb_common_energy_capacity_v1/inspect.py)
- [code/public_g22_operator_features_v1/export_gnn_payload.py](code/public_g22_operator_features_v1/export_gnn_payload.py)
- [code/public_g22_operator_features_v1/train_gnn_oracle.py](code/public_g22_operator_features_v1/train_gnn_oracle.py)

</details>

## Evidence

- [RESEARCH_RECEIPTS_2026_09_04_LATE.md#L65](../../arithmetic-kernels/bounded-radial-factor-thinning-variants-failed-their-local-accuracy/RESEARCH_RECEIPTS_2026_09_04_LATE.md#L65)
- [RESEARCH_RECEIPTS_2026_09_04_LATE.md#L1](../../arithmetic-kernels/bounded-radial-factor-thinning-variants-failed-their-local-accuracy/RESEARCH_RECEIPTS_2026_09_04_LATE.md#L1)

<details>
<summary>All result files and records (1)</summary>

- [RESEARCH_RECEIPTS_2026_09_04_LATE.md](../../arithmetic-kernels/bounded-radial-factor-thinning-variants-failed-their-local-accuracy/RESEARCH_RECEIPTS_2026_09_04_LATE.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-05-exact-winograd-optimization) · [Copy provenance](../../../PROVENANCE.json)

