# Depth-1 arena with correction strength 0.75 and frozen endpoint-action head submitted as #328616

2026-08-27 · experiment · rejected · measured

The exact [35,279,35] gate passed with zero offline-authority drift, exact +96,699,371 FLOPs per call, byte-identical repeat, maximum residual 0.341496 s, and maximum resident-set size (RSS) 4.289 GB. Its protected projection did not transfer: submission #328616 graded raw 3.065038697513956e-8 and adjusted 2.2155806519289345e-8, both worse than #328586.

Recorded measurement:

    The exact [35,279,35] gate passed with zero offline-authority drift, exact +96,699,371 FLOPs per call, byte-identical repeat, maximum residual 0.341496 s, and maximum RSS 4.289 GB. Its protected projection did not transfer: submission #328616 graded raw 3.065038697513956e-8 and adjusted 2.2155806519289345e-8, both worse than #328586.

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
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_native_j_checkpoint_interpolation_v1/run_control.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_native_j_checkpoint_interpolation_v1/run_control.py)
- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_native_j_q10q15_residual_head_v1/run_discriminator.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_native_j_q10q15_residual_head_v1/run_discriminator.py)

<details>
<summary>Additional programs and dependencies (1)</summary>

- [code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_native_transport_energy_pilot_v1/deployment_prep/export_literal_payload.py](code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_native_transport_energy_pilot_v1/deployment_prep/export_literal_payload.py)

</details>

## Evidence

- [RESULT.json#L1](RESULT.json#L1)
- [RESULT-97a7f1b9.json#L1](RESULT-97a7f1b9.json#L1)
- [FULL_GATE_RESULT.json#L1](FULL_GATE_RESULT.json#L1)
- [SUBMIT_WATCH.log#L1](../../learned-correction/688-network-endpoint-action-head-refit-accepted-relative-parent/SUBMIT_WATCH.log#L1)

<details>
<summary>All result files and records (4)</summary>

- [RESULT.json](RESULT.json) · result
- [RESULT-97a7f1b9.json](RESULT-97a7f1b9.json) · result
- [FULL_GATE_RESULT.json](FULL_GATE_RESULT.json) · result
- [SUBMIT_WATCH.log](../../learned-correction/688-network-endpoint-action-head-refit-accepted-relative-parent/SUBMIT_WATCH.log) · log

</details>

[Research graph](../../../graph/index.html#node=2026-08-27-depth-1-arena-correction-strength-0-75-frozen-endpoint) · [Copy provenance](../../../PROVENANCE.json)

