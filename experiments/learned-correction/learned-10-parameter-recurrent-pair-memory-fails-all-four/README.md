# Learned 10-parameter recurrent pair memory fails all four held-out gates against zero and reset controls

2026-09-07T00:04Z · experiment · rejected · measured

A 10-parameter recurrent pair field (a learned 'pair memory' whose latent history H is model-generated while the lower-order trajectory and past post-ReLU third-order K21 planes are teacher-forced) was fitted for 100 epochs on 20 TRAIN networks (ranks 35..58 minus the VALID ranks 37, 39, 42, 48) against an independently trained no-history RESET control, supervising the bank's missing directed pair action from post-layer-12 to pre-layer-13. All four fixed held-out information gates failed, so the state law, source, readout and fit were closed without resume, sweep or rollout.

Recorded measurement:

    History covariance error / zero correction 0.9036964070131149 (gate <= 0.50); history next-mean proxy error / zero 0.9911990674534291 (gate <= 0.50); history / RESET next-mean 1.000236776947614 (gate <= 0.80); history beats RESET 1 of 4 (gate >= 3 of 4). Pooled covariance MSE history 6.826706406415215e-8, RESET 6.826881298375888e-8, zero 7.554203329167537e-8; pooled next-mean proxy MSE history 2.2013042803480336e-8, RESET 2.2007831856229816e-8, zero 2.2208498298970208e-8. Fit wall 107.7840105 s; peak RSS 930,906,112 bytes.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/astra_phase2_reexamination_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/train_learned_response_pair_memory_v1.py](code/train_learned_response_pair_memory_v1.py)
- [code/prepare_learned_response_pair_memory_v1.py](code/prepare_learned_response_pair_memory_v1.py)
- [code/certify_learned_response_pair_memory_v1.py](code/certify_learned_response_pair_memory_v1.py)
- [code/certify_learned_response_pair_memory_wide_v1.py](code/certify_learned_response_pair_memory_wide_v1.py)
- [code/candidate_fullpair_runtime_rectd4_t8192_bulkcopies_v1.py](code/candidate_fullpair_runtime_rectd4_t8192_bulkcopies_v1.py)
- [code/capture_two_memory_teacher_action_replay_v2.py](code/capture_two_memory_teacher_action_replay_v2.py)
- [code/capture_two_memory_teacher_action_v1.py](code/capture_two_memory_teacher_action_v1.py)
- [code/certify_pair_memory_response_v1.py](code/certify_pair_memory_response_v1.py)
- [code/certify_pair_memory_response_v2.py](code/certify_pair_memory_response_v2.py)
- [code/compact_final_head_v1.py](code/compact_final_head_v1.py)
- [code/compact_fullpair_final_reference_v1.py](code/compact_fullpair_final_reference_v1.py)
- [code/compact_scalar_final_v1.py](code/compact_scalar_final_v1.py)

<details>
<summary>Additional programs and dependencies (14)</summary>

- [code/judge_nonlinear_pair_memory_information_v1.py](code/judge_nonlinear_pair_memory_information_v1.py)
- [code/k3_radial_numpy.py](code/k3_radial_numpy.py)
- [code/learned_response_pair_memory_numpy_v1.py](code/learned_response_pair_memory_numpy_v1.py)
- [code/learned_response_pair_memory_torch_v1.py](code/learned_response_pair_memory_torch_v1.py)
- [code/pair_memory_action_physics_v1.py](code/pair_memory_action_physics_v1.py)
- [code/pair_memory_action_physics_v2.py](code/pair_memory_action_physics_v2.py)
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

- [LEARNED_RESPONSE_PAIR_MEMORY_V1_CLOSED.md#L36](LEARNED_RESPONSE_PAIR_MEMORY_V1_CLOSED.md#L36)
- [LEARNED_RESPONSE_PAIR_MEMORY_V1_CLOSED.md#L43](LEARNED_RESPONSE_PAIR_MEMORY_V1_CLOSED.md#L43)
- [CONTINUATION_2026_09_06_1704_PDT.md#L41](CONTINUATION_2026_09_06_1704_PDT.md#L41)

<details>
<summary>All result files and records (2)</summary>

- [LEARNED_RESPONSE_PAIR_MEMORY_V1_CLOSED.md](LEARNED_RESPONSE_PAIR_MEMORY_V1_CLOSED.md) · result
- [CONTINUATION_2026_09_06_1704_PDT.md](CONTINUATION_2026_09_06_1704_PDT.md) · record

</details>

## Related work

- measured by: [Source-only arithmetic census of the closed pair-memory law: 278.775B envelope, 11.225B under the 290B ceiling](../../../records/research/2026-09-07-source-only-arithmetic-census-closed-pair-memory-law-278/README.md)

[Research graph](../../../graph/index.html#node=2026-09-07-learned-10-parameter-recurrent-pair-memory-fails-all-four) · [Copy provenance](../../../PROVENANCE.json)

