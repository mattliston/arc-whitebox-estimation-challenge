# Exact-weight autonomous suffix state: all four held-out folds regressed

2026-08-27 · experiment · rejected · measured

The first C8 exact-W state failed all four out-of-fold (OOF) folds: pooled candidate mean squared error (MSE) 3.1214875126e-8 versus nonlinear-Z 3.0538458575e-8 and native 2.9736792466e-8, with 0/4 folds improving. It is terminal without serialization or retry.

Recorded measurement:

    The first C8 exact-W state failed all four OOF folds: pooled candidate MSE 3.1214875126e-8 versus nonlinear-Z 3.0538458575e-8 and native 2.9736792466e-8, with 0/4 folds improving. It is terminal without serialization or retry.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/public_q24_c8_autonomous_correction_state_v1/run_c8_oof.py](code/public_q24_c8_autonomous_correction_state_v1/run_c8_oof.py)
- [code/public_full688_local_innovation_transport_v1/dataset.py](code/public_full688_local_innovation_transport_v1/dataset.py)
- [code/public_full688_transport_energy_coherence_cv_v1/run_experiment.py](code/public_full688_transport_energy_coherence_cv_v1/run_experiment.py)
- [code/public_q24_native_j_q10q15_residual_head_v1/run_discriminator.py](code/public_q24_native_j_q10q15_residual_head_v1/run_discriminator.py)
- [code/public_q24_nonlinear_transported_raw20_head_v1/run_nonlinear_z.py](code/public_q24_nonlinear_transported_raw20_head_v1/run_nonlinear_z.py)
- [code/public_q24_qhead_full688_refit_v1/run_refit.py](code/public_q24_qhead_full688_refit_v1/run_refit.py)

## Evidence

- [PREREGISTRATION.json#L1](PREREGISTRATION.json#L1)
- [RUN_STATE.json#L1](RUN_STATE.json#L1)
- [RESULT.json#L1](RESULT.json#L1)
- [FINAL_FIT_RESULT.json#L1](FINAL_FIT_RESULT.json#L1)
- [FULL_GATE_RESULT.json#L1](../../wall-and-residual/688-network-endpoint-action-head-depth-3-arena-measured/FULL_GATE_RESULT.json#L1)
- [SUBMIT_WATCH.log#L1](../../wall-and-residual/688-network-endpoint-action-head-depth-3-arena-measured/SUBMIT_WATCH.log#L1)
- [PROCEDURAL_TERMINAL_RECEIPT.json#L1](PROCEDURAL_TERMINAL_RECEIPT.json#L1)

<details>
<summary>All result files and records (7)</summary>

- [PREREGISTRATION.json](PREREGISTRATION.json) · result
- [RUN_STATE.json](RUN_STATE.json) · result
- [RESULT.json](RESULT.json) · result
- [FINAL_FIT_RESULT.json](FINAL_FIT_RESULT.json) · result
- [FULL_GATE_RESULT.json](../../wall-and-residual/688-network-endpoint-action-head-depth-3-arena-measured/FULL_GATE_RESULT.json) · result
- [SUBMIT_WATCH.log](../../wall-and-residual/688-network-endpoint-action-head-depth-3-arena-measured/SUBMIT_WATCH.log) · log
- [PROCEDURAL_TERMINAL_RECEIPT.json](PROCEDURAL_TERMINAL_RECEIPT.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-08-27-exact-weight-autonomous-suffix-state-all-four-held-out) · [Copy provenance](../../../PROVENANCE.json)

