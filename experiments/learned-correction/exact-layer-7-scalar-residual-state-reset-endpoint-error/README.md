# Exact layer-7 scalar residual-state reset: endpoint error 3.3897455206496917e-8 missed the target

2026-08-27 · experiment · rejected · measured

On observer-fit validation ranks 160:199, a single perfect post-ReLU h7 reset followed by the frozen autonomous scalar suffix yields raw mean squared error (MSE) 3.3897455206496917e-8. It closes only 0.0306486637669077 of the autonomous-to-teacher MSE gap and remains 3.3897x above 1e-8. No target, observer-selection, name, or deployment truth was opened.

Recorded measurement:

    On observer-fit validation ranks 160:199, a single perfect post-ReLU h7 reset followed by the frozen autonomous scalar suffix yields raw MSE 3.3897455206496917e-8. It closes only 0.0306486637669077 of the autonomous-to-teacher MSE gap and remains 3.3897x above 1e-8. No target, observer-selection, name, or deployment truth was opened. RESULT SHA-256 4f4e2798021796ab735f3ae631b61cd67494e2bb279fc8630ef9e7de905f661e.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/p2r0g1_kerdock_h7_anchor_ceiling_v1/run_discriminator.py](code/p2r0g1_kerdock_h7_anchor_ceiling_v1/run_discriminator.py)
- [code/public_exposure_trained_residual_state_v1/run_experiment.py](code/public_exposure_trained_residual_state_v1/run_experiment.py)

## Evidence

- [PREREGISTRATION.json#L1](PREREGISTRATION.json#L1)
- [RESULT.json#L48](RESULT.json#L48)
- [source_README.md#L1](source_README.md#L1)

<details>
<summary>All result files and records (3)</summary>

- [PREREGISTRATION.json](PREREGISTRATION.json) · result
- [RESULT.json](RESULT.json) · result
- [source_README.md](source_README.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-08-27-exact-layer-7-scalar-residual-state-reset-endpoint-error) · [Copy provenance](../../../PROVENANCE.json)

