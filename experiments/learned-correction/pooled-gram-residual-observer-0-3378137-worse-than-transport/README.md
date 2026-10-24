# Pooled-Gram residual observer: 0.3378137% worse than the transport-energy observer

2026-08-24 · experiment · rejected · measured

The independently verified full-688 full-Gram pooled-context result is finite at out-of-fold (OOF) final mean squared error (MSE) 3.2444190155738505e-8. It regresses 0.3378137% versus J, remains 1.5932495% better than D, misses its frozen mean gate by 9.3408339%, and reaches maximum per-network MSE 3.391656718774529e-7 at rank643/mlp364. Both accuracy conjuncts fail, no literal promotion meter was run, and observer_selection remained sealed.

Recorded measurement:

    The independently verified full-688 full-Gram pooled-context result is finite at OOF final MSE 3.2444190155738505e-8. It regresses 0.3378137% versus J, remains 1.5932495% better than D, misses its frozen mean gate by 9.3408339%, and reaches maximum per-network MSE 3.391656718774529e-7 at rank643/mlp364. Both accuracy conjuncts fail, no literal promotion meter was run, and observer_selection remained sealed.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/public_full688_full_gram_pooled_context_cv_v1/run_experiment.py](code/public_full688_full_gram_pooled_context_cv_v1/run_experiment.py)
- [code/public_full688_full_gram_pooled_context_cv_v1/verify_results.py](code/public_full688_full_gram_pooled_context_cv_v1/verify_results.py)
- [code/public_full688_local_innovation_transport_v1/dataset.py](code/public_full688_local_innovation_transport_v1/dataset.py)
- [code/public_full688_transport_energy_coherence_cv_v1/run_experiment.py](code/public_full688_transport_energy_coherence_cv_v1/run_experiment.py)

## Evidence

- [OOF_DECOMPOSITION_AND_SUCCESSOR.json#L1](OOF_DECOMPOSITION_AND_SUCCESSOR.json#L1)
- [PREREGISTRATION.json#L1](PREREGISTRATION.json#L1)
- [RUN_MANIFEST.json#L1](RUN_MANIFEST.json#L1)
- [PREREGISTRATION_SEAL.json#L1](PREREGISTRATION_SEAL.json#L1)
- [CV_RESULTS-2a8a5695.json#L1](../../error-attribution/pooled-gram-residual-observer-failure-endpoint-adjoint-structure-separated/CV_RESULTS-2a8a5695.json#L1)
- [VERIFICATION.json#L42](../../error-attribution/pooled-gram-residual-observer-failure-endpoint-adjoint-structure-separated/VERIFICATION.json#L42)

<details>
<summary>All result files and records (6)</summary>

- [OOF_DECOMPOSITION_AND_SUCCESSOR.json](OOF_DECOMPOSITION_AND_SUCCESSOR.json) · result
- [PREREGISTRATION.json](PREREGISTRATION.json) · result
- [RUN_MANIFEST.json](RUN_MANIFEST.json) · result
- [PREREGISTRATION_SEAL.json](PREREGISTRATION_SEAL.json) · result
- [CV_RESULTS-2a8a5695.json](../../error-attribution/pooled-gram-residual-observer-failure-endpoint-adjoint-structure-separated/CV_RESULTS-2a8a5695.json) · result
- [VERIFICATION.json](../../error-attribution/pooled-gram-residual-observer-failure-endpoint-adjoint-structure-separated/VERIFICATION.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-08-24-pooled-gram-residual-observer-0-3378137-worse-than-transport) · [Copy provenance](../../../PROVENANCE.json)

