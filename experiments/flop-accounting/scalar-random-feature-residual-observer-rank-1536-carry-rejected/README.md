# Scalar random-feature residual observer on rank-1536 carry: rejected by the accuracy gate

2026-08-24 · experiment · rejected · measured

The verified fourfold observer_fit result is finite but decisively misses the compute-route accuracy gate: CP1536 host raw mean squared error (MSE) is 3.7127081767379208e-6, teacher-forced scalar-RFF MSE is 1.2587467593815147e-6, and autonomous MSE is 1.2744674766113143e-6. Teacher and autonomous differ by only 1.25%, so recurrence drift is not the limiting factor in this spelling; local fixed-RFF capacity is. Its projected deployment total is 199,548,139,522 FLOPs (9.0744% of budget), yielding projected floor-adjusted 1.2744674766113142e-7.

Recorded measurement:

    The verified fourfold observer_fit result is finite but decisively misses the compute-route accuracy gate: CP1536 host raw MSE is 3.7127081767379208e-6, teacher-forced scalar-RFF MSE is 1.2587467593815147e-6, and autonomous MSE is 1.2744674766113143e-6. Teacher and autonomous differ by only 1.25%, so recurrence drift is not the limiting factor in this spelling; local fixed-RFF capacity is. Its projected deployment total is 199,548,139,522 FLOPs (9.0744% of budget), yielding projected floor-adjusted 1.2744674766113142e-7.

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_cp1536_scalar_transport_rff_cv_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/run_experiment.py](code/run_experiment.py)
- [code/run_experiment_v2.py](code/run_experiment_v2.py)

## Evidence

- [PREREGISTRATION.json#L1](PREREGISTRATION.json#L1)
- [RUN_MANIFEST.json#L1](RUN_MANIFEST.json#L1)
- [PREREGISTRATION_SEAL.json#L1](PREREGISTRATION_SEAL.json#L1)
- [MECHANICAL_AMENDMENT_V2.json#L1](MECHANICAL_AMENDMENT_V2.json#L1)
- [RUN_MANIFEST_V2.json#L1](RUN_MANIFEST_V2.json#L1)
- [MECHANICAL_SEAL_V2.json#L1](MECHANICAL_SEAL_V2.json#L1)
- [CV_RESULTS_V2.json#L1](CV_RESULTS_V2.json#L1)
- [verification.json#L1](verification.json#L1)
- [EXECUTION_RECONCILIATION_V2.json#L1](EXECUTION_RECONCILIATION_V2.json#L1)
- [graph_delta.json#L14](graph_delta.json#L14)

<details>
<summary>All result files and records (10)</summary>

- [PREREGISTRATION.json](PREREGISTRATION.json) · result
- [RUN_MANIFEST.json](RUN_MANIFEST.json) · result
- [PREREGISTRATION_SEAL.json](PREREGISTRATION_SEAL.json) · result
- [MECHANICAL_AMENDMENT_V2.json](MECHANICAL_AMENDMENT_V2.json) · result
- [RUN_MANIFEST_V2.json](RUN_MANIFEST_V2.json) · result
- [MECHANICAL_SEAL_V2.json](MECHANICAL_SEAL_V2.json) · result
- [CV_RESULTS_V2.json](CV_RESULTS_V2.json) · result
- [verification.json](verification.json) · result
- [EXECUTION_RECONCILIATION_V2.json](EXECUTION_RECONCILIATION_V2.json) · result
- [graph_delta.json](graph_delta.json) · result

</details>

[Research graph](../../../graph/index.html#node=2026-08-24-scalar-random-feature-residual-observer-rank-1536-carry-rejected) · [Copy provenance](../../../PROVENANCE.json)

