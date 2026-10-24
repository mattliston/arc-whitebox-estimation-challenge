# Response-row matched round: closed at the declared gate

2026-09-05T07:09Z · experiment · rejected · measured

- Rank35:2.10823951686e-8 versus2.04444474536e-8, +3.1204%. - Rank100:1.99700304672e-8 versus2.00406278849e-8, -0.3523%. Rank35 exceeds the predeclared maximum3% raw worsening. Payload:payload_response_rows_v1.npz; model tag ASTRA_RESPONSE_ROWS_V1_C64_E40; training receipt chain_response_rows_training_v1.json; full-loop receipts response_rows_closedloop_r35_v1.json and response_rows_closedloop_r100_v1.json.

Recorded measurement:

    - Rank35:2.10823951686e-8 versus2.04444474536e-8, +3.1204%. - Rank100:1.99700304672e-8 versus2.00406278849e-8, -0.3523%. Rank35 exceeds the predeclared maximum3% raw worsening. The separate unlearned rank35 metered cost saving5.85% does not establish a competitive learned adjusted result. No threshold sweep, final-head refit, candidate port,14-net panel, or submission. Preserve rows, model, payload and receipts.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/astra_phase2_reexamination_v1/build_response_rows_cost_v1.py](code/astra_phase2_reexamination_v1/build_response_rows_cost_v1.py)
- [code/astra_phase2_reexamination_v1/build_response_rows_cost_v2.py](code/astra_phase2_reexamination_v1/build_response_rows_cost_v2.py)
- [code/astra_phase2_reexamination_v1/build_response_rows_current_runtime_v1.py](code/astra_phase2_reexamination_v1/build_response_rows_current_runtime_v1.py)
- [code/astra_phase2_reexamination_v1/candidate_fullpair_runtime_d5_t8192_v1.py](code/astra_phase2_reexamination_v1/candidate_fullpair_runtime_d5_t8192_v1.py)
- [code/astra_phase2_reexamination_v1/candidate_fullpair_runtime_rectd4_t8192_bulkcopies_v1.py](code/astra_phase2_reexamination_v1/candidate_fullpair_runtime_rectd4_t8192_bulkcopies_v1.py)
- [code/astra_phase2_reexamination_v1/candidate_response_pair_polynomial_f32_v1.py](code/astra_phase2_reexamination_v1/candidate_response_pair_polynomial_f32_v1.py)
- [code/astra_phase2_reexamination_v1/candidate_response_rows_cost_v1.py](code/astra_phase2_reexamination_v1/candidate_response_rows_cost_v1.py)
- [code/astra_phase2_reexamination_v1/candidate_response_rows_cost_v2.py](code/astra_phase2_reexamination_v1/candidate_response_rows_cost_v2.py)
- [code/astra_phase2_reexamination_v1/candidate_response_rows_current_runtime_v1.py](code/astra_phase2_reexamination_v1/candidate_response_rows_current_runtime_v1.py)
- [code/astra_phase2_reexamination_v1/candidate_response_rows_parent_cost_v1.py](code/astra_phase2_reexamination_v1/candidate_response_rows_parent_cost_v1.py)
- [code/astra_phase2_reexamination_v1/candidate_whole_pair_kernel_post7_k512_v1.py](code/astra_phase2_reexamination_v1/candidate_whole_pair_kernel_post7_k512_v1.py)
- [code/astra_phase2_reexamination_v1/chain_radial_training.py](code/astra_phase2_reexamination_v1/chain_radial_training.py)

<details>
<summary>Additional programs and dependencies (20)</summary>

- [code/astra_phase2_reexamination_v1/chain_response_rows_training_v1.py](code/astra_phase2_reexamination_v1/chain_response_rows_training_v1.py)
- [code/astra_phase2_reexamination_v1/check_response_rows_adjusted_v1.py](code/astra_phase2_reexamination_v1/check_response_rows_adjusted_v1.py)
- [code/astra_phase2_reexamination_v1/generate_radial_rows.py](code/astra_phase2_reexamination_v1/generate_radial_rows.py)
- [code/astra_phase2_reexamination_v1/generate_response_rows_v1.py](code/astra_phase2_reexamination_v1/generate_response_rows_v1.py)
- [code/astra_phase2_reexamination_v1/k3_radial_numpy.py](code/astra_phase2_reexamination_v1/k3_radial_numpy.py)
- [code/astra_phase2_reexamination_v1/radial_binary_compact.py](code/astra_phase2_reexamination_v1/radial_binary_compact.py)
- [code/astra_phase2_reexamination_v1/radial_compact_feed_fullpair_v1.py](code/astra_phase2_reexamination_v1/radial_compact_feed_fullpair_v1.py)
- [code/astra_phase2_reexamination_v1/radial_compact_radialfeed_v1.py](code/astra_phase2_reexamination_v1/radial_compact_radialfeed_v1.py)
- [code/astra_phase2_reexamination_v1/radial_cubic.py](code/astra_phase2_reexamination_v1/radial_cubic.py)
- [code/astra_phase2_reexamination_v1/radial_features.py](code/astra_phase2_reexamination_v1/radial_features.py)
- [code/astra_phase2_reexamination_v1/radial_mehler.py](code/astra_phase2_reexamination_v1/radial_mehler.py)
- [code/astra_phase2_reexamination_v1/radial_response_rows_v1.py](code/astra_phase2_reexamination_v1/radial_response_rows_v1.py)
- [code/codex_resume_20260919_v1/current_p_dual_terminal_information_v1/capture_candidate.py](code/codex_resume_20260919_v1/current_p_dual_terminal_information_v1/capture_candidate.py)
- [code/codex_resume_20260919_v1/current_p_dual_terminal_information_v1/common.py](code/codex_resume_20260919_v1/current_p_dual_terminal_information_v1/common.py)
- [code/codex_resume_20260919_v1/current_p_dual_terminal_information_v1/guard.py](code/codex_resume_20260919_v1/current_p_dual_terminal_information_v1/guard.py)
- [code/codex_resume_20260919_v1/current_p_dual_terminal_information_v1/seal.py](code/codex_resume_20260919_v1/current_p_dual_terminal_information_v1/seal.py)
- [code/codex_resume_20260919_v1/current_p_release_v1/candidate.py](code/codex_resume_20260919_v1/current_p_release_v1/candidate.py)
- [code/codex_resume_20260926_v2/hill_climb_common_energy_capacity_v1/inspect.py](code/codex_resume_20260926_v2/hill_climb_common_energy_capacity_v1/inspect.py)
- [code/public_g22_operator_features_v1/export_gnn_payload.py](code/public_g22_operator_features_v1/export_gnn_payload.py)
- [code/public_g22_operator_features_v1/train_gnn_oracle.py](code/public_g22_operator_features_v1/train_gnn_oracle.py)

</details>

## Evidence

- [RESPONSE_ROWS_MATCHED_ROUND_V1_CLOSED.md#L1](RESPONSE_ROWS_MATCHED_ROUND_V1_CLOSED.md#L1)
- [RESPONSE_ROWS_MATCHED_ROUND_V1_CLOSED.md#L12](RESPONSE_ROWS_MATCHED_ROUND_V1_CLOSED.md#L12)

<details>
<summary>All result files and records (5)</summary>

- [RESPONSE_ROWS_MATCHED_ROUND_V1_CLOSED.md](RESPONSE_ROWS_MATCHED_ROUND_V1_CLOSED.md) · record
- [chain_response_rows_training_v1.json](chain_response_rows_training_v1.json) · result
- [response_rows_closedloop_r35_v1.json](response_rows_closedloop_r35_v1.json) · result
- [response_rows_closedloop_r100_v1.json](response_rows_closedloop_r100_v1.json) · result
- [RESPONSE_ROWS_MATCHED_ROUND_V1.md](RESPONSE_ROWS_MATCHED_ROUND_V1.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-05-response-row-matched-round-closed-declared-gate) · [Copy provenance](../../../PROVENANCE.json)

