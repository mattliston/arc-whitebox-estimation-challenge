# Nonlinear historical pair memory: fixed information test failed

2026-09-06T21:41Z · experiment · rejected · measured

History / RESET pooled next-mean error is 1.0020445794. History loses on all four networks; per-net ratios are37:1.00295073,39:1.00007517, 42:1.00219598,48:1.00275726. The fixed test is complete and closed.

Recorded measurement:

    | Consumer | Zero-correction MSE | History MSE | History / zero | RESET / zero | | --- | ---: | ---: | ---: | ---: | | Covariance | 7.55420332890e-8 | 6.83127896854e-8 | .9043017074 | .9044486819 | | Next leading mean | 2.22084983478e-8 | 2.20466193732e-8 | .9927109446 | .9906854096 | History / RESET pooled next-mean error is 1.0020445794. History loses on all four networks; per-net ratios are37:1.00295073,39:1.00007517, 42:1.00219598,48:1.00275726. All four fixed gates fail: covariance <=.50, next mean <=.50, history/RESET next <=.80, and >=3/4 next-mean wins. History coefficients are[-.00287137328019,-.000228368440528]; RESET [-.00308812824208,+.000182416869501]. Normal-equation condition numbers are3.65577 and2.41839 respectively, with finite positive TRAIN energies. There is no singular-solve or near-zero-denominator ambiguity. Original capture v1 rows35–37 passed numerical parity, but37 exceeded the fixed6GiB cap:6,459,342,848 bytes versus6,442,450,944. The v1 panel driver correctly stopped before38; all sources, rows and failure logs remain. An additive v2 resource-only replay accumulated the OFFLINE physical CP read in fixed2048-column float64 chunks. Bank state arithmetic was unchanged. All24 v2 captures passed at4,955,635,712–5,534,760,960 bytes peak RSS. The three existing v1 rows were compared with v2: all original bank arrays byte-identical; worst physical-post-plane relativeL2 difference3.9588e-15 against the fixed1e-10 limit. No source or target-dependent parameter changed. The response v2 tiny certificate passed ten symmetric/asymmetric/active-floor fixtures, worst error1.14926e-13 against2e-11. The v1 symmetric formula's asymmetric witness was1.25808e-4; its frozen sources/certificate are preserved. The final judge passed all provenance, chronology and finite-value checks; wall20.9817s, peak1,326,891,008 bytes. One nice10 NumPy-only worker, no Torch or MPS. Two BLAS threads were requested; backend instrumentation was empty, so the effective thread count is not independently certified. Swap remained 840.75MiB. Floored cold post-variance units are the explicit protocol choice, not a PSD repair or grounds to relabel this negative as inconclusive. The fixed test is complete and closed. The <1e-9 official adjusted goal remains active and unmet. No threshold bank has been earned by this result.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/astra_phase2_reexamination_v1/candidate_fullpair_runtime_rectd4_t8192_bulkcopies_v1.py](code/astra_phase2_reexamination_v1/candidate_fullpair_runtime_rectd4_t8192_bulkcopies_v1.py)
- [code/astra_phase2_reexamination_v1/capture_reflected_residual_copula_trace_v1.py](code/astra_phase2_reexamination_v1/capture_reflected_residual_copula_trace_v1.py)
- [code/astra_phase2_reexamination_v1/capture_two_memory_teacher_action_replay_v2.py](code/astra_phase2_reexamination_v1/capture_two_memory_teacher_action_replay_v2.py)
- [code/astra_phase2_reexamination_v1/capture_two_memory_teacher_action_v1.py](code/astra_phase2_reexamination_v1/capture_two_memory_teacher_action_v1.py)
- [code/astra_phase2_reexamination_v1/certify_pair_memory_response_v1.py](code/astra_phase2_reexamination_v1/certify_pair_memory_response_v1.py)
- [code/astra_phase2_reexamination_v1/certify_pair_memory_response_v2.py](code/astra_phase2_reexamination_v1/certify_pair_memory_response_v2.py)
- [code/astra_phase2_reexamination_v1/certify_reflected_copula_tree_action_v1.py](code/astra_phase2_reexamination_v1/certify_reflected_copula_tree_action_v1.py)
- [code/astra_phase2_reexamination_v1/certify_reflected_relu_marginal_v1.py](code/astra_phase2_reexamination_v1/certify_reflected_relu_marginal_v1.py)
- [code/astra_phase2_reexamination_v1/compact_final_head_v1.py](code/astra_phase2_reexamination_v1/compact_final_head_v1.py)
- [code/astra_phase2_reexamination_v1/compact_fullpair_final_reference_v1.py](code/astra_phase2_reexamination_v1/compact_fullpair_final_reference_v1.py)
- [code/astra_phase2_reexamination_v1/compact_scalar_final_v1.py](code/astra_phase2_reexamination_v1/compact_scalar_final_v1.py)
- [code/astra_phase2_reexamination_v1/judge_nonlinear_pair_memory_information_v1.py](code/astra_phase2_reexamination_v1/judge_nonlinear_pair_memory_information_v1.py)

<details>
<summary>Additional programs and dependencies (16)</summary>

- [code/astra_phase2_reexamination_v1/judge_reflected_residual_copula_tree_v1.py](code/astra_phase2_reexamination_v1/judge_reflected_residual_copula_tree_v1.py)
- [code/astra_phase2_reexamination_v1/k3_radial_numpy.py](code/astra_phase2_reexamination_v1/k3_radial_numpy.py)
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

- [NONLINEAR_PAIR_MEMORY_INFORMATION_V1_CLOSED.md#L1](NONLINEAR_PAIR_MEMORY_INFORMATION_V1_CLOSED.md#L1)
- [NONLINEAR_PAIR_MEMORY_INFORMATION_V1_CLOSED.md#L38](NONLINEAR_PAIR_MEMORY_INFORMATION_V1_CLOSED.md#L38)
- [NONLINEAR_PAIR_MEMORY_INFORMATION_V1_CLOSED.md#L42](NONLINEAR_PAIR_MEMORY_INFORMATION_V1_CLOSED.md#L42)

<details>
<summary>All result files and records (4)</summary>

- [NONLINEAR_PAIR_MEMORY_INFORMATION_V1_CLOSED.md](NONLINEAR_PAIR_MEMORY_INFORMATION_V1_CLOSED.md) · record
- [NONLINEAR_PAIR_MEMORY_INFORMATION_PROTOCOL_V1.md](NONLINEAR_PAIR_MEMORY_INFORMATION_PROTOCOL_V1.md) · record
- [PAIR_MEMORY_RESPONSE_EXACTNESS_V2.md](../../../records/research/2026-09-06-additive-directed-correlation-response-correction/PAIR_MEMORY_RESPONSE_EXACTNESS_V2.md) · record
- [result.json](result.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-09-06-nonlinear-historical-pair-memory-fixed-information-test-failed) · [Copy provenance](../../../PROVENANCE.json)

