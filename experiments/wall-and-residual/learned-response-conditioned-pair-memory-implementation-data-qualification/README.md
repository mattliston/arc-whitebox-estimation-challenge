# Learned response-conditioned pair memory: implementation and data qualification

2026-09-06T23:56Z · experiment · informational · measured

Synthetic n7/history13 certificate finished23:44:56UTC, wall1.68653s, peakRSS259,751,936B. CPU64/NumPy64 outputs agree to3.13e-17 relative; gate-gradient finite-difference disagreement <=1.18413e-10. All remaining23 fixed rows36..58 passed preparation, each about.32--.42s; maximum processRSS1,526,759,424B.

Recorded measurement:

    Synthetic n7/history13 certificate finished23:44:56UTC, wall1.68653s, peakRSS259,751,936B. CPU64/NumPy64 outputs agree to3.13e-17 relative; gate-gradient finite-difference disagreement <=1.18413e-10. MPS32 output disagreement <=1.92304e-8, gate-gradient disagreement <=1.21390e-7. Response-unit identity error4.82594e-16; query-sign parity and bounds pass. Only the single fixed fixture/vector/probe was used, no sweep or retry. Receipt SHA df4dfa5a0cd4e9410c07aa34e6426538c2acfd585e1eddbaafb42ac0b0939b56. The inputs-only rank35 full-width certificate finished23:50:21UTC, wall3.45221s, peakRSS2,590,736,384B. Both NumPy64/CPU64 outputs agreed to 4.21328e-16 relative. MPS32 output <=2.22371e-7, full gradient <=2.36845e-7, gate-gradient <=4.90093e-7. No loss.npz was opened or hashed by this certificate; its probe was synthetic, not a target. Receipt SHA e1ea3d1f919504ec886436a48b6a50a6db34b5f443fdb09eb1cccd48b6f81663. The comparison is on preparedFP32 causal inputs, not an independently regenerated native-FP32 moment carrier or a deployment parity claim. Numerical programs ran nice10, requested twoCPU threads, and used only one MPS worker at a time with fallback0. AP never imported threadpoolctl. Last memory-pressure check was816.75MiB swap in use. No thresholds were altered after observing data. No fit has been scored by this note; the paired100-epoch fit and its .50/.50/.80/3-of-4 gates are the next stage.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/astra_phase2_reexamination_v1/candidate_fullpair_runtime_rectd4_t8192_bulkcopies_v1.py](code/astra_phase2_reexamination_v1/candidate_fullpair_runtime_rectd4_t8192_bulkcopies_v1.py)
- [code/astra_phase2_reexamination_v1/capture_two_memory_teacher_action_replay_v2.py](code/astra_phase2_reexamination_v1/capture_two_memory_teacher_action_replay_v2.py)
- [code/astra_phase2_reexamination_v1/capture_two_memory_teacher_action_v1.py](code/astra_phase2_reexamination_v1/capture_two_memory_teacher_action_v1.py)
- [code/astra_phase2_reexamination_v1/certify_learned_response_pair_memory_v1.py](code/astra_phase2_reexamination_v1/certify_learned_response_pair_memory_v1.py)
- [code/astra_phase2_reexamination_v1/certify_learned_response_pair_memory_wide_v1.py](code/astra_phase2_reexamination_v1/certify_learned_response_pair_memory_wide_v1.py)
- [code/astra_phase2_reexamination_v1/certify_pair_memory_response_v1.py](code/astra_phase2_reexamination_v1/certify_pair_memory_response_v1.py)
- [code/astra_phase2_reexamination_v1/certify_pair_memory_response_v2.py](code/astra_phase2_reexamination_v1/certify_pair_memory_response_v2.py)
- [code/astra_phase2_reexamination_v1/compact_final_head_v1.py](code/astra_phase2_reexamination_v1/compact_final_head_v1.py)
- [code/astra_phase2_reexamination_v1/compact_fullpair_final_reference_v1.py](code/astra_phase2_reexamination_v1/compact_fullpair_final_reference_v1.py)
- [code/astra_phase2_reexamination_v1/compact_scalar_final_v1.py](code/astra_phase2_reexamination_v1/compact_scalar_final_v1.py)
- [code/astra_phase2_reexamination_v1/judge_nonlinear_pair_memory_information_v1.py](code/astra_phase2_reexamination_v1/judge_nonlinear_pair_memory_information_v1.py)
- [code/astra_phase2_reexamination_v1/k3_radial_numpy.py](code/astra_phase2_reexamination_v1/k3_radial_numpy.py)

<details>
<summary>Additional programs and dependencies (14)</summary>

- [code/astra_phase2_reexamination_v1/learned_response_pair_memory_numpy_v1.py](code/astra_phase2_reexamination_v1/learned_response_pair_memory_numpy_v1.py)
- [code/astra_phase2_reexamination_v1/learned_response_pair_memory_torch_v1.py](code/astra_phase2_reexamination_v1/learned_response_pair_memory_torch_v1.py)
- [code/astra_phase2_reexamination_v1/pair_memory_action_physics_v1.py](code/astra_phase2_reexamination_v1/pair_memory_action_physics_v1.py)
- [code/astra_phase2_reexamination_v1/pair_memory_action_physics_v2.py](code/astra_phase2_reexamination_v1/pair_memory_action_physics_v2.py)
- [code/astra_phase2_reexamination_v1/prepare_learned_response_pair_memory_v1.py](code/astra_phase2_reexamination_v1/prepare_learned_response_pair_memory_v1.py)
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

- [LEARNED_RESPONSE_PAIR_MEMORY_NUMERICAL_QUALIFICATION_V1.md#L1](LEARNED_RESPONSE_PAIR_MEMORY_NUMERICAL_QUALIFICATION_V1.md#L1)
- [LEARNED_RESPONSE_PAIR_MEMORY_NUMERICAL_QUALIFICATION_V1.md#L11](LEARNED_RESPONSE_PAIR_MEMORY_NUMERICAL_QUALIFICATION_V1.md#L11)
- [LEARNED_RESPONSE_PAIR_MEMORY_NUMERICAL_QUALIFICATION_V1.md#L23](LEARNED_RESPONSE_PAIR_MEMORY_NUMERICAL_QUALIFICATION_V1.md#L23)

<details>
<summary>All result files and records (1)</summary>

- [LEARNED_RESPONSE_PAIR_MEMORY_NUMERICAL_QUALIFICATION_V1.md](LEARNED_RESPONSE_PAIR_MEMORY_NUMERICAL_QUALIFICATION_V1.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-06-learned-response-conditioned-pair-memory-implementation-data-qualification) · [Copy provenance](../../../PROVENANCE.json)

