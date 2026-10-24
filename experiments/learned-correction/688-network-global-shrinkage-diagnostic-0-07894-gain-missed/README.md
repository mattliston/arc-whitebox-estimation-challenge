# 688-network global shrinkage diagnostic: 0.07894% gain missed the acceptance threshold

2026-08-27 · experiment · rejected · measured

Held-fold beta shrink reaches out-of-fold (OOF) mean squared error (MSE) 2.971331865239914e-8 versus deployment-shaped native 2.973679246649156e-8: a 0.07894% gain, all four folds, with 58.866% network wins. Final beta is 0.144121509. It misses the stricter 2.967252855e-8 gate, and optimistic depth-3 third-cumulant arena (D3) projection 1.898516960e-8 is worse than actual depth-4 third-cumulant arena (D4) bank #328622.

Recorded measurement:

    Held-fold beta shrink reaches OOF MSE 2.971331865239914e-8 versus deployment-shaped native 2.973679246649156e-8: a 0.07894% gain, all four folds, with 58.866% network wins. Final beta is 0.144121509. It misses the stricter 2.967252855e-8 gate, and optimistic D3 projection 1.898516960e-8 is worse than actual D4 bank #328622.

Program source is in [`code/`](code/), preserving the relative layout below `.`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/estimator.py](code/estimator.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d10_cp3_arena_depth3_full_estimator_v1/build_candidate.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d10_cp3_arena_depth3_full_estimator_v1/build_candidate.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d10_cp3_arena_depth3_full_estimator_v1/estimator.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d10_cp3_arena_depth3_full_estimator_v1/estimator.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d10_cp3_arena_depth3_full_estimator_v1/run_candidate_gate.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d10_cp3_arena_depth3_full_estimator_v1/run_candidate_gate.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d11_cp3_arena_depth4_full_estimator_v1/build_candidate.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d11_cp3_arena_depth4_full_estimator_v1/build_candidate.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d11_cp3_arena_depth4_full_estimator_v1/estimator.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d11_cp3_arena_depth4_full_estimator_v1/estimator.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d11_cp3_arena_depth4_full_estimator_v1/run_candidate_gate.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d11_cp3_arena_depth4_full_estimator_v1/run_candidate_gate.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d9_cp3_arena_depth1_full_estimator_v1/build_candidate.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d9_cp3_arena_depth1_full_estimator_v1/build_candidate.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d9_cp3_arena_depth1_full_estimator_v1/estimator.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d9_cp3_arena_depth1_full_estimator_v1/estimator.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d9_cp3_arena_depth1_full_estimator_v1/run_candidate_gate.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d9_cp3_arena_depth1_full_estimator_v1/run_candidate_gate.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_full688_local_innovation_transport_v1/dataset.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_full688_local_innovation_transport_v1/dataset.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_full688_transport_energy_coherence_cv_v1/run_experiment.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_full688_transport_energy_coherence_cv_v1/run_experiment.py)

<details>
<summary>Additional programs and dependencies (5)</summary>

- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_noeigh_card_v1/estimator.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_noeigh_card_v1/estimator.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_full688_vs_native_bank_shrink_diagnostic_v1/run_diagnostic.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_full688_vs_native_bank_shrink_diagnostic_v1/run_diagnostic.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_g11b_all688_shrink_diagnostic_v1/run_diagnostic.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_g11b_all688_shrink_diagnostic_v1/run_diagnostic.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_native_j_q10q15_residual_head_v1/run_discriminator.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_native_j_q10q15_residual_head_v1/run_discriminator.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_qhead_full688_refit_v1/run_refit.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_qhead_full688_refit_v1/run_refit.py)

</details>

## Evidence

- [PREREGISTRATION.json#L1](PREREGISTRATION.json#L1)
- [PREREGISTRATION-57882c7f.json#L1](PREREGISTRATION-57882c7f.json#L1)
- [RESULT.json#L1](RESULT.json#L1)
- [RESULT.json#L1](../../wall-and-residual/688-network-endpoint-action-head-depth-3-arena-measured/RESULT.json#L1)
- [SUBMIT_WATCH.log#L1](../../wall-and-residual/688-network-endpoint-action-head-depth-3-arena-measured/SUBMIT_WATCH.log#L1)

<details>
<summary>All result files and records (5)</summary>

- [PREREGISTRATION.json](PREREGISTRATION.json) · result
- [PREREGISTRATION-57882c7f.json](PREREGISTRATION-57882c7f.json) · result
- [RESULT.json](RESULT.json) · result
- [RESULT.json](../../wall-and-residual/688-network-endpoint-action-head-depth-3-arena-measured/RESULT.json) · result
- [SUBMIT_WATCH.log](../../wall-and-residual/688-network-endpoint-action-head-depth-3-arena-measured/SUBMIT_WATCH.log) · log

</details>

[Research graph](../../../graph/index.html#node=2026-08-27-688-network-global-shrinkage-diagnostic-0-07894-gain-missed) · [Copy provenance](../../../PROVENANCE.json)

