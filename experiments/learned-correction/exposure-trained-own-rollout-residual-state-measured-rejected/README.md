# Exposure-trained own-rollout residual state: measured and rejected

2026-08-24 · experiment · rejected · measured

The frozen 6,688-parameter H64 own-rollout model is finite at autonomous development final mean squared error (MSE) 3.4709697955756664e-8: 9.14% better than the established ~3.82e-8 family, but 0.90% above its preregistered <3.44e-8 primary gate and above the <1e-8 stretch gate. Same-checkpoint teacher-forced MSE is 8.207963477637236e-9, leaving a 4.23x autonomous gap. Configuration was frozen before loading observer_fit 0..199; ranks 160..199 were already development-open and materialized at startup, metrics were read only after epoch 16, and no observer_selection or rank>=200 target was accessed.

Recorded measurement:

    The frozen 6,688-parameter H64 own-rollout model is finite at autonomous development final MSE 3.4709697955756664e-8: 9.14% better than the established ~3.82e-8 family, but 0.90% above its preregistered <3.44e-8 primary gate and above the <1e-8 stretch gate. Same-checkpoint teacher-forced MSE is 8.207963477637236e-9, leaving a 4.23x autonomous gap. Configuration was frozen before loading observer_fit 0..199; ranks 160..199 were already development-open and materialized at startup, metrics were read only after epoch 16, and no observer_selection or rank>=200 target was accessed.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_exposure_trained_residual_state_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/run_experiment.py](code/run_experiment.py)

## Evidence

- [PREREGISTRATION.json#L23](PREREGISTRATION.json#L23)
- [PREREGISTRATION_SEAL.json#L1](PREREGISTRATION_SEAL.json#L1)
- [source_README.md#L34](source_README.md#L34)
- [RESULT.json#L1](RESULT.json#L1)
- [RESULT_RECONCILIATION.json#L1](RESULT_RECONCILIATION.json#L1)

<details>
<summary>All result files and records (5)</summary>

- [PREREGISTRATION.json](PREREGISTRATION.json) · result
- [PREREGISTRATION_SEAL.json](PREREGISTRATION_SEAL.json) · result
- [source_README.md](source_README.md) · record
- [RESULT.json](RESULT.json) · result
- [RESULT_RECONCILIATION.json](RESULT_RECONCILIATION.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-08-24-exposure-trained-own-rollout-residual-state-measured-rejected) · [Copy provenance](../../../PROVENANCE.json)

