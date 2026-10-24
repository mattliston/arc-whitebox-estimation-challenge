# 688-network endpoint-action head refit: accepted relative to its parent but worse than the native observer

2026-08-27 · experiment · informational · measured

The frozen one-shot 22->64->64 refit passes every preregistered gate. out-of-fold (OOF) mean squared error (MSE) is 3.066199675740282e-8 versus 3.117186304030849e-8 for the incumbent on the same rows (+1.63566%); all four fold means improve, network wins are 83.866%, paired z=25.716, and protected40 improves to 2.993270275e-8 from 3.061465198e-8. However, identical-population native model MSE is 2.973679246e-8, so this is not an absolute base win.

Recorded measurement:

    The frozen one-shot 22->64->64 refit passes every preregistered gate. OOF MSE is 3.066199675740282e-8 versus 3.117186304030849e-8 for the incumbent on the same rows (+1.63566%); all four fold means improve, network wins are 83.866%, paired z=25.716, and protected40 improves to 2.993270275e-8 from 3.061465198e-8. However, identical-population native model MSE is 2.973679246e-8, so this is not an absolute base win.

Program source is in [`code/`](code/), preserving the relative layout below `.`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/estimator.py](code/estimator.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d9_cp3_arena_depth1_full_estimator_v1/estimator.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0c1d9_cp3_arena_depth1_full_estimator_v1/estimator.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_full688_local_innovation_transport_v1/dataset.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_full688_local_innovation_transport_v1/dataset.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_full688_transport_energy_coherence_cv_v1/run_experiment.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_full688_transport_energy_coherence_cv_v1/run_experiment.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_d1_alpha075_qhead_bank_v1/build_candidate.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_d1_alpha075_qhead_bank_v1/build_candidate.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_d1_alpha075_qhead_bank_v1/estimator.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_d1_alpha075_qhead_bank_v1/estimator.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_d1_alpha075_qhead_bank_v1/generate_expected_outputs.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_d1_alpha075_qhead_bank_v1/generate_expected_outputs.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_d1_alpha075_qhead_bank_v1/generate_torch_authority.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_d1_alpha075_qhead_bank_v1/generate_torch_authority.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_d1_alpha075_qhead_bank_v1/run_full_gate.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_d1_alpha075_qhead_bank_v1/run_full_gate.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_d1_alpha075_qhead_bank_v1/verify_transport_parity.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_d1_alpha075_qhead_bank_v1/verify_transport_parity.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_native_j_q10q15_residual_head_v1/run_discriminator.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_native_j_q10q15_residual_head_v1/run_discriminator.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_native_transport_energy_pilot_v1/deployment_prep/export_literal_payload.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_native_transport_energy_pilot_v1/deployment_prep/export_literal_payload.py)

<details>
<summary>Additional programs and dependencies (1)</summary>

- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_qhead_full688_refit_v1/run_refit.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_qhead_full688_refit_v1/run_refit.py)

</details>

## Evidence

- [PREREGISTRATION.json#L1](PREREGISTRATION.json#L1)
- [RUN_SEAL.json#L1](RUN_SEAL.json#L1)
- [RESULT.json#L838](../../wall-and-residual/688-network-endpoint-action-head-depth-3-arena-measured/RESULT.json#L838)
- [SUBMIT_WATCH.log#L1](SUBMIT_WATCH.log#L1)

<details>
<summary>All result files and records (4)</summary>

- [PREREGISTRATION.json](PREREGISTRATION.json) · result
- [RUN_SEAL.json](RUN_SEAL.json) · result
- [RESULT.json](../../wall-and-residual/688-network-endpoint-action-head-depth-3-arena-measured/RESULT.json) · result
- [SUBMIT_WATCH.log](SUBMIT_WATCH.log) · log

</details>

[Research graph](../../../graph/index.html#node=2026-08-27-688-network-endpoint-action-head-refit-accepted-relative-parent) · [Copy provenance](../../../PROVENANCE.json)

