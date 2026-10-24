# Mean-and-variance residual state: endpoint error 6.65959% worse than the dynamic-state observer

2026-08-24 · experiment · rejected · measured

The verified fourfold full-688 typed-state result is finite at out-of-fold (OOF) final mean squared error (MSE) 3.5165106535091496e-8. It improves the state-independent Duhamel form by 10.01995% but remains 6.65959% worse than P2R0F2D, 18.51065% above its frozen mean gate, and reaches maximum per-network MSE 3.42368130350359e-7 at rank643/mlp364. Diagonal prediction-error variance therefore does not arrest the late-depth accumulation in this exact form; observer_selection remained sealed.

Recorded measurement:

    The verified fourfold full-688 typed-state result is finite at OOF final MSE 3.5165106535091496e-8. It improves the state-independent Duhamel form by 10.01995% but remains 6.65959% worse than P2R0F2D, 18.51065% above its frozen mean gate, and reaches maximum per-network MSE 3.42368130350359e-7 at rank643/mlp364. Diagonal prediction-error variance therefore does not arrest the late-depth accumulation in this exact form; observer_selection remained sealed.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/public_full688_typed_diagonal_error_variance_cv_v1/run_experiment.py](code/public_full688_typed_diagonal_error_variance_cv_v1/run_experiment.py)
- [code/public_full688_typed_diagonal_error_variance_cv_v1/verify_results.py](code/public_full688_typed_diagonal_error_variance_cv_v1/verify_results.py)
- [code/public_full688_local_innovation_transport_v1/dataset.py](code/public_full688_local_innovation_transport_v1/dataset.py)

## Evidence

- [PREREGISTRATION.json#L1](PREREGISTRATION.json#L1)
- [RUN_MANIFEST.json#L1](RUN_MANIFEST.json#L1)
- [PREREGISTRATION_SEAL.json#L1](PREREGISTRATION_SEAL.json#L1)
- [CV_RESULTS.json#L1](CV_RESULTS.json#L1)
- [VERIFICATION.json#L57](VERIFICATION.json#L57)

<details>
<summary>All result files and records (5)</summary>

- [PREREGISTRATION.json](PREREGISTRATION.json) · result
- [RUN_MANIFEST.json](RUN_MANIFEST.json) · result
- [PREREGISTRATION_SEAL.json](PREREGISTRATION_SEAL.json) · result
- [CV_RESULTS.json](CV_RESULTS.json) · result
- [VERIFICATION.json](VERIFICATION.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-08-24-mean-variance-residual-state-endpoint-error-6-65959-worse) · [Copy provenance](../../../PROVENANCE.json)

