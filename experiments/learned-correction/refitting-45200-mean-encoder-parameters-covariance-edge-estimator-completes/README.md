# Refitting 45200 mean-encoder parameters on the covariance-edge estimator completes 160 graphs and 40 updates

2026-09-21T12:58Z · experiment · adopted · measured

The learned mean-encoder P (message weights and biases, three encoder layers, head, head bias and gain: 11 keys, 45,200 parameters) had last been fitted on the older birth-direction carrier; it was now refit in place on the current bank #331684 through an explicit CPU re-implementation of the whole estimator (full covariance, coordinate-star and trace ownership, shared rank-384 transport with ten joins and four young cohorts, covariance edge at layers 8 to 14, the 0.01 selectors and the scalar terminal shortcut) in which ordinary Torch products replace the native fast products. One fixed pass over the 160 TRAIN networks (four networks per Adam update, 40 updates, cosine learning rate 5e-4 to 5e-5, gradient-norm cap 5, loss = final served MSE plus 0.05 times the earlier 15 served MSEs, divided by 2e-8) completed with everything else frozen and produced the single candidate 8260ddb1... that became #331760.

Recorded measurement:

    160/160 graphs, 40/40 updates, peak 6.420 GB, zero added swap; resource admission: worst qualified graph 30.11640950012952 s, projected 160 graphs 4818.625520020723 s, ceiling 14400 s, RSS trigger 7500000000 bytes

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/codex_resume_20260919_v1/current_full_lower_p_fit_v1/run_fit.py](code/codex_resume_20260919_v1/current_full_lower_p_fit_v1/run_fit.py)
- [code/codex_resume_20260919_v1/current_full_lower_p_fit_v1/train.py](code/codex_resume_20260919_v1/current_full_lower_p_fit_v1/train.py)
- [code/codex_resume_20260919_v1/current_full_lower_p_fit_v1/build.py](code/codex_resume_20260919_v1/current_full_lower_p_fit_v1/build.py)
- [code/codex_resume_20260919_v1/current_p_fit_pipeline_v1/run.py](code/codex_resume_20260919_v1/current_p_fit_pipeline_v1/run.py)
- [code/astra_phase2_reexamination_v1/centered_bank_autodiff_v1.py](code/astra_phase2_reexamination_v1/centered_bank_autodiff_v1.py)
- [code/astra_phase2_reexamination_v1/centered_bank_autodiff_v3.py](code/astra_phase2_reexamination_v1/centered_bank_autodiff_v3.py)
- [code/astra_phase2_reexamination_v1/centered_torch_numerics_v1.py](code/astra_phase2_reexamination_v1/centered_torch_numerics_v1.py)
- [code/astra_phase2_reexamination_v1/centered_torch_numerics_v3.py](code/astra_phase2_reexamination_v1/centered_torch_numerics_v3.py)
- [code/astra_phase2_reexamination_v1/current_shared384_causal_p_v1/shared_operator.py](code/astra_phase2_reexamination_v1/current_shared384_causal_p_v1/shared_operator.py)
- [code/astra_phase2_reexamination_v1/current_transport_shadow_v1.py](code/astra_phase2_reexamination_v1/current_transport_shadow_v1.py)
- [code/astra_phase2_reexamination_v1/current_transport_shadow_v3.py](code/astra_phase2_reexamination_v1/current_transport_shadow_v3.py)
- [code/astra_phase2_reexamination_v1/full_carrier_autodiff_v1.py](code/astra_phase2_reexamination_v1/full_carrier_autodiff_v1.py)

<details>
<summary>Additional programs and dependencies (21)</summary>

- [code/astra_phase2_reexamination_v1/full_carrier_bank_autodiff_v1.py](code/astra_phase2_reexamination_v1/full_carrier_bank_autodiff_v1.py)
- [code/astra_phase2_reexamination_v1/gaussian_cdf_quadrature_v1.py](code/astra_phase2_reexamination_v1/gaussian_cdf_quadrature_v1.py)
- [code/astra_phase2_reexamination_v1/joint_prefix_final_data_v1.py](code/astra_phase2_reexamination_v1/joint_prefix_final_data_v1.py)
- [code/astra_phase2_reexamination_v1/tensor_clamp_min_v1.py](code/astra_phase2_reexamination_v1/tensor_clamp_min_v1.py)
- [code/codex_resume_20260919_v1/current_full_lower_p_allocator_v1/alloc_common.py](code/codex_resume_20260919_v1/current_full_lower_p_allocator_v1/alloc_common.py)
- [code/codex_resume_20260919_v1/current_full_lower_p_allocator_v1/allocator.py](code/codex_resume_20260919_v1/current_full_lower_p_allocator_v1/allocator.py)
- [code/codex_resume_20260919_v1/current_full_lower_p_allocator_v1/guard.py](code/codex_resume_20260919_v1/current_full_lower_p_allocator_v1/guard.py)
- [code/codex_resume_20260919_v1/current_full_lower_p_fit_v1/candidate.py](code/codex_resume_20260919_v1/current_full_lower_p_fit_v1/candidate.py)
- [code/codex_resume_20260919_v1/current_full_lower_p_fit_v1/endpoint_common.py](code/codex_resume_20260919_v1/current_full_lower_p_fit_v1/endpoint_common.py)
- [code/codex_resume_20260919_v1/current_full_lower_p_fit_v1/fit_common.py](code/codex_resume_20260919_v1/current_full_lower_p_fit_v1/fit_common.py)
- [code/codex_resume_20260919_v1/current_full_lower_p_fit_v1/guard.py](code/codex_resume_20260919_v1/current_full_lower_p_fit_v1/guard.py)
- [code/codex_resume_20260919_v1/current_full_lower_p_fit_v1/native.py](code/codex_resume_20260919_v1/current_full_lower_p_fit_v1/native.py)
- [code/codex_resume_20260919_v1/current_full_lower_p_fit_v1/null_candidate.py](code/codex_resume_20260919_v1/current_full_lower_p_fit_v1/null_candidate.py)
- [code/codex_resume_20260919_v1/current_full_lower_p_fit_v1/qualify_null.py](code/codex_resume_20260919_v1/current_full_lower_p_fit_v1/qualify_null.py)
- [code/codex_resume_20260919_v1/current_full_lower_p_fit_v1/run_endpoint.py](code/codex_resume_20260919_v1/current_full_lower_p_fit_v1/run_endpoint.py)
- [code/codex_resume_20260919_v1/current_full_lower_p_fit_v1/score_native40.py](code/codex_resume_20260919_v1/current_full_lower_p_fit_v1/score_native40.py)
- [code/codex_resume_20260919_v1/current_full_lower_p_fit_v1/seal_predictions.py](code/codex_resume_20260919_v1/current_full_lower_p_fit_v1/seal_predictions.py)
- [code/codex_resume_20260919_v1/current_full_lower_p_proposal_v1/common.py](code/codex_resume_20260919_v1/current_full_lower_p_proposal_v1/common.py)
- [code/codex_resume_20260919_v1/current_full_lower_p_proposal_v1/model_v2.py](code/codex_resume_20260919_v1/current_full_lower_p_proposal_v1/model_v2.py)
- [code/codex_resume_20260919_v1/full_lower_covariance_edge_packed_v1/candidate.py](code/codex_resume_20260919_v1/full_lower_covariance_edge_packed_v1/candidate.py)
- [code/codex_resume_20260919_v1/run_guarded.py](code/codex_resume_20260919_v1/run_guarded.py)

</details>

## Evidence

- [PROTOCOL.md#L30](../../../records/research/2026-09-21-cpu-estimator-mean-encoder-refit-first-backward-pass-exceeded/PROTOCOL.md#L30)
- [PROTOCOL.md#L88](../../../records/research/2026-09-21-cpu-estimator-mean-encoder-refit-first-backward-pass-exceeded/PROTOCOL.md#L88)
- [REEXAMINATION_INDEX_20260919.md#L473](../../birth-terms/birth-only-threshold-0-05-raises-pooled-adjusted-error/REEXAMINATION_INDEX_20260919.md#L473)
- [REEXAMINATION_INDEX_20260919.md#L501](../../birth-terms/birth-only-threshold-0-05-raises-pooled-adjusted-error/REEXAMINATION_INDEX_20260919.md#L501)
- [RESULT.md#L3](RESULT.md#L3)

<details>
<summary>All result files and records (6)</summary>

- [PROTOCOL.md](../../../records/research/2026-09-21-cpu-estimator-mean-encoder-refit-first-backward-pass-exceeded/PROTOCOL.md) · config
- [ENDPOINT.md](ENDPOINT.md) · config
- [resource_admission.json](resource_admission.json) · record
- [PROTOCOL.md](PROTOCOL.md) · config
- [RESULT.md](RESULT.md) · record
- [REEXAMINATION_INDEX_20260919.md](../../birth-terms/birth-only-threshold-0-05-raises-pooled-adjusted-error/REEXAMINATION_INDEX_20260919.md) · record

</details>

## Related work

- ← enables: [Allocator relief at graph boundaries cuts the plain forward high-water from 6,793,330,688 to 2,965,045,248 bytes](../../../records/research/2026-09-21-allocator-relief-graph-boundaries-cuts-plain-forward-high-water/README.md)
- enables: [Allocator relief at graph boundaries cuts the plain forward high-water from 6,793,330,688 to 2,965,045,248 bytes](../../../records/research/2026-09-21-allocator-relief-graph-boundaries-cuts-plain-forward-high-water/README.md)
- measured by: [Refit mean encoder on the 40-network validation split: adjusted ratio 0.9848676992236052, 32/40 wins at equal compute](../../evaluation-methodology/refit-mean-encoder-40-network-validation-split-adjusted-ratio/README.md)
- shipped in: [Submission #331760: 45200 mean-encoder parameters refit on the covariance-edge estimator; score 3.507738856975084e-9](../../../submissions/331760/README.md)
- ← shipped in: [Submission #331760: 45200 mean-encoder parameters refit on the covariance-edge estimator; score 3.507738856975084e-9](../../../submissions/331760/README.md)

[Research graph](../../../graph/index.html#node=2026-09-21-refitting-45200-mean-encoder-parameters-covariance-edge-estimator-completes) · [Copy provenance](../../../PROVENANCE.json)

