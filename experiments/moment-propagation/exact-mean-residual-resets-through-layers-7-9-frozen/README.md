# Exact mean-residual resets through layers 7 and 9: frozen suffix remained above 1e-8

2026-08-27 · experiment · rejected · measured

Perfect full-width mean-residual resets at h2:h7 end at 3.3897455206496917e-8, identical to the last h7 reset because the old recurrence is Markov. Extending exact resets through h9 reaches 3.16038750480864e-8 and closes 0.1171931939109314 of the autonomous-to-teacher mean squared error (MSE) gap, still 3.1604x above 1e-8. RESULT SHA-256 a96828887522e8e412cfb53a9d39d2dd39d3ca561a43c1e66d52f27f73f78ca5.

Recorded measurement:

    Perfect full-width mean-residual resets at h2:h7 end at 3.3897455206496917e-8, identical to the last h7 reset because the old recurrence is Markov. Extending exact resets through h9 reaches 3.16038750480864e-8 and closes 0.1171931939109314 of the autonomous-to-teacher MSE gap, still 3.1604x above 1e-8. RESULT SHA-256 a96828887522e8e412cfb53a9d39d2dd39d3ca561a43c1e66d52f27f73f78ca5.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/p2r0g1_kerdock_multilayer_anchor_ceiling_v1/run_discriminator.py](code/p2r0g1_kerdock_multilayer_anchor_ceiling_v1/run_discriminator.py)
- [code/public_exposure_trained_residual_state_v1/run_experiment.py](code/public_exposure_trained_residual_state_v1/run_experiment.py)

## Evidence

- [PREREGISTRATION.json#L1](PREREGISTRATION.json#L1)
- [RESULT.json#L101](RESULT.json#L101)

<details>
<summary>All result files and records (2)</summary>

- [PREREGISTRATION.json](PREREGISTRATION.json) · result
- [RESULT.json](RESULT.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-08-27-exact-mean-residual-resets-through-layers-7-9-frozen) · [Copy provenance](../../../PROVENANCE.json)

