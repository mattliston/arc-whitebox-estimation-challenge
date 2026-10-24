# State-independent Duhamel innovation cross-validation: endpoint error 18.5369% worse

2026-08-24 · experiment · rejected · measured

The verified state-independent source-only rollout is finite but regresses to aggregate out-of-fold (OOF) final mean squared error (MSE) 3.908100560611721e-8, 18.5369% worse than P2R0F2D. Its maximum per-network MSE is 3.4112928798161795e-7 at role rank 643. It fails both frozen conjuncts;OOF <2.9672528546945005e-8 and every network <1e-7;showing that raw-local innovation without learned state context is insufficient.

Recorded measurement:

    The verified state-independent source-only rollout is finite but regresses to aggregate OOF final MSE 3.908100560611721e-8, 18.5369% worse than P2R0F2D. Its maximum per-network MSE is 3.4112928798161795e-7 at role rank 643. It fails both frozen conjuncts—OOF <2.9672528546945005e-8 and every network <1e-7—showing that raw-local innovation without learned state context is insufficient.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/public_full688_state_independent_duhamel_cv_v1/run_experiment.py](code/public_full688_state_independent_duhamel_cv_v1/run_experiment.py)
- [code/public_full688_state_independent_duhamel_cv_v1/verify_results.py](code/public_full688_state_independent_duhamel_cv_v1/verify_results.py)
- [code/public_full688_local_innovation_transport_v1/dataset.py](code/public_full688_local_innovation_transport_v1/dataset.py)

## Evidence

- [CV_RESULTS.json#L1](../../flop-accounting/dynamic-multistate-observer-full-688-out-fold-error-3/CV_RESULTS.json#L1)
- [VERIFICATION.json#L1](../../flop-accounting/dynamic-multistate-observer-full-688-out-fold-error-3/VERIFICATION.json#L1)
- [PREREGISTRATION.json#L64](PREREGISTRATION.json#L64)
- [RUN_MANIFEST.json#L44](RUN_MANIFEST.json#L44)
- [PREREGISTRATION_SEAL.json#L1](PREREGISTRATION_SEAL.json#L1)
- [CV_RESULTS.json#L1](CV_RESULTS.json#L1)
- [VERIFICATION.json#L55](VERIFICATION.json#L55)

<details>
<summary>All result files and records (7)</summary>

- [CV_RESULTS.json](../../flop-accounting/dynamic-multistate-observer-full-688-out-fold-error-3/CV_RESULTS.json) · result
- [VERIFICATION.json](../../flop-accounting/dynamic-multistate-observer-full-688-out-fold-error-3/VERIFICATION.json) · result
- [PREREGISTRATION.json](PREREGISTRATION.json) · result
- [RUN_MANIFEST.json](RUN_MANIFEST.json) · result
- [PREREGISTRATION_SEAL.json](PREREGISTRATION_SEAL.json) · result
- [CV_RESULTS.json](CV_RESULTS.json) · result
- [VERIFICATION.json](VERIFICATION.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-08-24-state-independent-duhamel-innovation-cross-validation-endpoint-error-18) · [Copy provenance](../../../PROVENANCE.json)

