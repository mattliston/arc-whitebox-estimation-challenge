# 22-input late-layer action head: 1.968315% endpoint gain missed the required oracle-gap recovery

2026-08-27 · experiment · rejected · measured

The immutable exactly-once four-fold independent256 primary arm is a valid terminal failure. Its local q mean squared error (MSE) is 5.298375774407814e-9 and its endpoint MSE is 2.988906304823771e-8 versus fold-J base 3.048918630419429e-8: a 1.968315% base improvement that closes only 2.464820% of the base-to-exact-q-oracle gap. The absolute local gate <=1.375544984492742e-9 and strict endpoint gate <1.2427319908150922e-8 both fail.

Recorded measurement:

    The immutable exactly-once four-fold independent256 primary arm is a valid terminal failure. Its local q MSE is 5.298375774407814e-9 and its endpoint MSE is 2.988906304823771e-8 versus fold-J base 3.048918630419429e-8: a 1.968315% base improvement that closes only 2.464820% of the base-to-exact-q-oracle gap. The absolute local gate <=1.375544984492742e-9 and strict endpoint gate <1.2427319908150922e-8 both fail. The shared256 diagnostic is marginally better but is frozen diagnostic-only, also fails the absolute gates, and is nonselectable after results. This is a closed_measured missing-information boundary for native22, with no retry. A post-terminal observational trace, explicitly not a new gate, also shows normalized training q loss ending near 0.969-0.974, so even fit rows explain only about 2-3% of normalized q energy.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/public_q24_signed_pair_qinnovation_distillation_v1/native22_capacity_runner.py](code/public_q24_signed_pair_qinnovation_distillation_v1/native22_capacity_runner.py)
- [code/public_full688_local_innovation_transport_v1/dataset.py](code/public_full688_local_innovation_transport_v1/dataset.py)
- [code/public_full688_transport_energy_coherence_cv_v1/run_experiment.py](code/public_full688_transport_energy_coherence_cv_v1/run_experiment.py)
- [code/public_q24_native_j_q10q15_residual_head_v1/run_discriminator.py](code/public_q24_native_j_q10q15_residual_head_v1/run_discriminator.py)
- [code/public_q24_qhead_full688_refit_v1/run_refit.py](code/public_q24_qhead_full688_refit_v1/run_refit.py)

## Evidence

- [NATIVE22_CAPACITY_PREREGISTRATION_V1.json#L108](NATIVE22_CAPACITY_PREREGISTRATION_V1.json#L108)
- [NATIVE22_EXECUTION_RELEASE_V1.json#L1](NATIVE22_EXECUTION_RELEASE_V1.json#L1)
- [RESULT.json#L1](RESULT.json#L1)
- [NATIVE22_TERMINAL_AUDIT_V1.json#L57](NATIVE22_TERMINAL_AUDIT_V1.json#L57)

<details>
<summary>All result files and records (4)</summary>

- [NATIVE22_CAPACITY_PREREGISTRATION_V1.json](NATIVE22_CAPACITY_PREREGISTRATION_V1.json) · result
- [NATIVE22_EXECUTION_RELEASE_V1.json](NATIVE22_EXECUTION_RELEASE_V1.json) · result
- [RESULT.json](RESULT.json) · result
- [NATIVE22_TERMINAL_AUDIT_V1.json](NATIVE22_TERMINAL_AUDIT_V1.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-08-27-22-input-late-layer-action-head-1-968315-endpoint) · [Copy provenance](../../../PROVENANCE.json)

