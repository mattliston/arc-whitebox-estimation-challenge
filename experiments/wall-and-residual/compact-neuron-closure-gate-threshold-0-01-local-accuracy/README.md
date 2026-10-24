# Compact-neuron closure with gate threshold 0.01: local accuracy and resource validation passed

2026-09-05T01:34Z · experiment · informational · measured

Matched 200 honest rows, fixed160/40 net split, C64,40epochs, mean head only. Held-out one-step final MSE6.39339e-9 versus host8.095e-9 (21.0% removal). Closed-loop development raw against stored1e9 post means, rank35/rank100: 2.2929837e-8 /2.3413942e-8.

Recorded measurement:

    Matched 200 honest rows, fixed160/40 net split, C64,40epochs, mean head only. Held-out one-step final MSE6.39339e-9 versus host8.095e-9 (21.0% removal). The completed training payload's actual mtime is September4,17:57:11 PDT. Closed-loop development raw against stored1e9 post means, rank35/rank100: 2.2929837e-8 /2.3413942e-8. Full-bank learned reference: 2.2123083e-8 /2.1138533e-8. This modest raw loss buys a large compute saving. | Group | Raw MSE | Mean utilization | Adjusted | | --- | ---: | ---: | ---: | | Public8 | 2.6961976e-8 | .23010 | 6.2055707e-9 | | Virgin6 | 2.3396607e-8 | .22818 | 5.3523087e-9 | Versus the previous fully validated D5 full-bank card on identical panels, adjusted decreases23.08% /25.60%. Versus the official-best329790 card's identical local panels, decreases61.33% /58.88%. These are LOCAL scores, not an official graded submission. The official best remains1.4369934e-8. Native official-version35/100/35 gate passes: maximum mean deviations 1.2611e-6 /1.1714e-6 against independent NumPy; learned delta deviations 3.45e-8 /2.45e-8; repeated rank35 byte-identical. PeakRSS6,798,016,512bytes. Native residual time under diagnostic copying is not the official gate. Actual numpy-free local client/server: utilization.230770826, wall35.13s, residual.174879970s, max childRSS6,624,051,200bytes. All resource limits pass. Receipts binary_compact_g01_learned_native_gate_v1.json and binary_compact_g01_learned_direct_remote_v1{,_resources}.json.

Program source is in [`code/`](code/), preserving the relative layout below `.`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_phase2_reexamination_v1/candidate_binary_compact_g01_learned_direct_v1.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_phase2_reexamination_v1/candidate_binary_compact_g01_learned_direct_v1.py)
- [code/estimator.py](code/estimator.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_phase2_reexamination_v1/build_compact_scalar_final_v1.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_phase2_reexamination_v1/build_compact_scalar_final_v1.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_phase2_reexamination_v1/build_fullpair_scalar_final_v1.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_phase2_reexamination_v1/build_fullpair_scalar_final_v1.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_phase2_reexamination_v1/candidate_binary_compact_g01_learned_d6_v1.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_phase2_reexamination_v1/candidate_binary_compact_g01_learned_d6_v1.py)
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

- [RESEARCH_RECEIPTS_2026_09_04_EVENING.md#L8](../../arithmetic-kernels/compact-closure-cost-tests-factor-thinning-contraction-scheduling/RESEARCH_RECEIPTS_2026_09_04_EVENING.md#L8)
- [RESEARCH_RECEIPTS_2026_09_04_EVENING.md#L1](../../arithmetic-kernels/compact-closure-cost-tests-factor-thinning-contraction-scheduling/RESEARCH_RECEIPTS_2026_09_04_EVENING.md#L1)
- [RESEARCH_RECEIPTS_2026_09_04_EVENING.md#L19](../../arithmetic-kernels/compact-closure-cost-tests-factor-thinning-contraction-scheduling/RESEARCH_RECEIPTS_2026_09_04_EVENING.md#L19)
- [RESEARCH_RECEIPTS_2026_09_04_EVENING.md#L26](../../arithmetic-kernels/compact-closure-cost-tests-factor-thinning-contraction-scheduling/RESEARCH_RECEIPTS_2026_09_04_EVENING.md#L26)

<details>
<summary>All result files and records (2)</summary>

- [RESEARCH_RECEIPTS_2026_09_04_EVENING.md](../../arithmetic-kernels/compact-closure-cost-tests-factor-thinning-contraction-scheduling/RESEARCH_RECEIPTS_2026_09_04_EVENING.md) · record
- [binary_compact_g01_learned_native_gate_v1.json](binary_compact_g01_learned_native_gate_v1.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-09-05-compact-neuron-closure-gate-threshold-0-01-local-accuracy) · [Copy provenance](../../../PROVENANCE.json)

