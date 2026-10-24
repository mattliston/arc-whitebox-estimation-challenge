# Transport-energy residual observer: out-of-fold error 3.2334958229470984e-8, above the acceptance threshold

2026-08-24 · experiment · rejected · measured

The verified fourfold full-688 per-channel transport-energy result is finite at out-of-fold (OOF) final mean squared error (MSE) 3.2334958229470984e-8, 1.92456% better than P2R0F2D and the best autonomous capacity result so far. It remains 8.97271% above its frozen mean gate, and maximum per-network MSE is 3.408533352646559e-7 at rank643/mlp364. Its separate CP1792 feature pass makes exact order-24 Hermite pair-series closure (Q24)+CP1792+J cost 2,342,509,746,011 FLOPs, 143,486,490,459 above budget, so this exact OOF result is not submission-ready.

Recorded measurement:

    The verified fourfold full-688 per-channel transport-energy result is finite at OOF final MSE 3.2334958229470984e-8, 1.92456% better than P2R0F2D and the best autonomous capacity result so far. It remains 8.97271% above its frozen mean gate, and maximum per-network MSE is 3.408533352646559e-7 at rank643/mlp364. Its separate CP1792 feature pass makes exact Q24+CP1792+J cost 2,342,509,746,011 FLOPs, 143,486,490,459 above budget, so this exact OOF result is not submission-ready. Observer_selection remained sealed.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/public_full688_transport_energy_coherence_cv_v1/run_experiment.py](code/public_full688_transport_energy_coherence_cv_v1/run_experiment.py)
- [code/public_full688_transport_energy_coherence_cv_v1/verify_results.py](code/public_full688_transport_energy_coherence_cv_v1/verify_results.py)
- [code/cp3_cap16384_mean_recurrence_ablation/estimator.py](code/cp3_cap16384_mean_recurrence_ablation/estimator.py)
- [code/public_cp16384_dynamic_multistate_pilot_v1/generate_host_row.py](code/public_cp16384_dynamic_multistate_pilot_v1/generate_host_row.py)
- [code/public_cp16384_dynamic_multistate_pilot_v1/instrumented_estimator.py](code/public_cp16384_dynamic_multistate_pilot_v1/instrumented_estimator.py)
- [code/public_full688_local_innovation_transport_v1/dataset.py](code/public_full688_local_innovation_transport_v1/dataset.py)

## Evidence

- [OOF_DECOMPOSITION_AND_SUCCESSOR.json#L1](OOF_DECOMPOSITION_AND_SUCCESSOR.json#L1)
- [PREREGISTRATION.json#L1](PREREGISTRATION.json#L1)
- [RUN_MANIFEST.json#L1](RUN_MANIFEST.json#L1)
- [PREREGISTRATION_SEAL.json#L1](PREREGISTRATION_SEAL.json#L1)
- [CV_RESULTS.json#L1](../../error-attribution/pooled-gram-residual-observer-failure-endpoint-adjoint-structure-separated/CV_RESULTS.json#L1)
- [VERIFICATION.json#L57](VERIFICATION.json#L57)
- [ERRATUM_001_EXECUTABILITY.json#L1](../rank-16384-third-cumulant-carrier-recursive-mean-plus-dynamic/ERRATUM_001_EXECUTABILITY.json#L1)

<details>
<summary>All result files and records (7)</summary>

- [OOF_DECOMPOSITION_AND_SUCCESSOR.json](OOF_DECOMPOSITION_AND_SUCCESSOR.json) · result
- [PREREGISTRATION.json](PREREGISTRATION.json) · result
- [RUN_MANIFEST.json](RUN_MANIFEST.json) · result
- [PREREGISTRATION_SEAL.json](PREREGISTRATION_SEAL.json) · result
- [CV_RESULTS.json](../../error-attribution/pooled-gram-residual-observer-failure-endpoint-adjoint-structure-separated/CV_RESULTS.json) · result
- [VERIFICATION.json](VERIFICATION.json) · result
- [ERRATUM_001_EXECUTABILITY.json](../rank-16384-third-cumulant-carrier-recursive-mean-plus-dynamic/ERRATUM_001_EXECUTABILITY.json) · result

</details>

## Related work

- ← shipped in: [Submission #328317: native-trajectory residual observer on the order-24 closure](../../../submissions/328317/README.md)

[Research graph](../../../graph/index.html#node=2026-08-24-transport-energy-residual-observer-out-fold-error-3-2334958229470984e) · [Copy provenance](../../../PROVENANCE.json)

