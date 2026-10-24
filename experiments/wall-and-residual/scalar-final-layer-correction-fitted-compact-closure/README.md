# Scalar final-layer correction fitted on the compact closure

2026-09-05T01:34Z · experiment · informational · measured

The200 on-policy final-only rows completed in356.6s. The original trainer failed before epoch1 because its composed output bias inferred float64, unsupported on MPS. The previous compact depth-5 third-cumulant arena (D5) package was locally built and inspected: only estimator.py and manifest.json.

Recorded measurement:

    The200 on-policy final-only rows completed in356.6s. The original trainer failed before epoch1 because its composed output bias inferred float64, unsupported on MPS. Original script preserved; the versioned resume wrapper adds explicit float32 only. It completed the same fixed40epoch fit and wrote payload_compact_final_head_v1.npz (mtime18:16:04PDT). Held-out40 raw MSE2.5778075e-8 versus original final head2.5609225e-8, a0.66% loss. Clamped prefix and original-final predictions match the frozen reference BYTE-for-BYTE. The initial unclamped comparison differed only at negative outputs that the estimator clips; this was checked explicitly. Independent scalar-final reference development raw2.3340556e-8 /2.3494745e-8. It reproduces frozen full-matrix host means within1.041e-7 and its final inputs within8.53e-6. FNP D6 rank35 raw2.3350245e-8, utilization.214039197. | Group | Raw MSE | Mean utilization | Adjusted | | --- | ---: | ---: | ---: | | Public8 | 2.7222063e-8 | .21485 | 5.8500076e-9 | | Virgin6 | 2.3517122e-8 | .21327 | 5.0310490e-9 | Native35/100/35, INCLUDING both prefix and separate final-head parity, passes: max mean1.5382e-6, final correction4.406e-8, final input.002009; repeated rank35 byte-identical; peakRSS7,147,962,368bytes. Actual numpy-free client/server utilization.214135089, wall70.03s under concurrent load, residual.272378087, max childRSS6,383,304,704bytes. All resource limits pass. Source SHA25669e1ef2cf0828a5bd3567007b6981a97bfcca95302a760b95d0e0b6d362d62c1. This is the current best fully validated LOCAL result, not a graded score. Early single-cube schedule, restoring current repeated entries into star: first layer only raw3.39770e-8 /3.34568e-8; first two4.78951e-8 /4.69640e-8, versus compact full two-cube2.93709e-8 /3.32566e-8. Not worth a port or coadaptation at this cost/accuracy tradeoff. No global-family reopening.

Program source is in [`code/`](code/), preserving the relative layout below `.`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_phase2_reexamination_v1/candidate_compact_scalar_final_d6_v1.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_phase2_reexamination_v1/candidate_compact_scalar_final_d6_v1.py)
- [code/estimator.py](code/estimator.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_phase2_reexamination_v1/build_compact_scalar_final_v1.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_phase2_reexamination_v1/build_compact_scalar_final_v1.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_phase2_reexamination_v1/build_fullpair_scalar_final_v1.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_phase2_reexamination_v1/build_fullpair_scalar_final_v1.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_phase2_reexamination_v1/candidate_binary_compact_g01_learned_d6_v1.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_phase2_reexamination_v1/candidate_binary_compact_g01_learned_d6_v1.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_phase2_reexamination_v1/candidate_binary_compact_g01_learned_direct_v1.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_phase2_reexamination_v1/candidate_binary_compact_g01_learned_direct_v1.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_phase2_reexamination_v1/candidate_compact_radialfeed_unlearned_d6_v1.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_phase2_reexamination_v1/candidate_compact_radialfeed_unlearned_d6_v1.py)
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

- [RESEARCH_RECEIPTS_2026_09_04_EVENING.md#L78](../../arithmetic-kernels/compact-closure-cost-tests-factor-thinning-contraction-scheduling/RESEARCH_RECEIPTS_2026_09_04_EVENING.md#L78)
- [RESEARCH_RECEIPTS_2026_09_04_EVENING.md#L1](../../arithmetic-kernels/compact-closure-cost-tests-factor-thinning-contraction-scheduling/RESEARCH_RECEIPTS_2026_09_04_EVENING.md#L1)
- [RESEARCH_RECEIPTS_2026_09_04_EVENING.md#L90](../../arithmetic-kernels/compact-closure-cost-tests-factor-thinning-contraction-scheduling/RESEARCH_RECEIPTS_2026_09_04_EVENING.md#L90)
- [RESEARCH_RECEIPTS_2026_09_04_EVENING.md#L99](../../arithmetic-kernels/compact-closure-cost-tests-factor-thinning-contraction-scheduling/RESEARCH_RECEIPTS_2026_09_04_EVENING.md#L99)

<details>
<summary>All result files and records (1)</summary>

- [RESEARCH_RECEIPTS_2026_09_04_EVENING.md](../../arithmetic-kernels/compact-closure-cost-tests-factor-thinning-contraction-scheduling/RESEARCH_RECEIPTS_2026_09_04_EVENING.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-05-scalar-final-layer-correction-fitted-compact-closure) · [Copy provenance](../../../PROVENANCE.json)

