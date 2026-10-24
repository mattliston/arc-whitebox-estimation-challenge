# Matched-cost random, control-variate and lattice sampling comparators: metered without accuracy scoring

2026-08-23T21:07Z · experiment · inconclusive · measured

Three sampling lanes were built to spend exactly the cap-1792 closure's bill (each comparator bill at or below 214,122,341,378 FLOPs with no dummy work): plain iid sampling with 6372 samples, an affine Gaussian control variate with 3183 samples, and a randomised-shift lattice (RQMC). A cp1792 + RQMC nonlinear-residual hybrid was also derived and metered. Only cross-prediction disagreements were recorded target-blind; the accuracy opening never happened because the full-budget program superseded the equal-budget frame the same day.

Recorded measurement:

    iid: 214,122,102,784 FLOPs (margin 238,594 below cp1792), 6372 samples; affine_cv: 214,114,123,776 FLOPs (margin 8,217,602), 3183 samples; cross-prediction layer-16 RMS disagreements: iid vs cp1792 0.003552135692967976, rqmc vs cp1792 0.0031106551048402237, affine_cv vs cp1792 0.004893214735729589; hybrid cp1792+RQMC 219,870,793,378 FLOPs (+5,748,452,000 over cp1792)

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/equal_budget_sampling_comparator/estimator.py](code/equal_budget_sampling_comparator/estimator.py)
- [code/cp1792_rqmc_nonlinear_residual/estimator.py](code/cp1792_rqmc_nonlinear_residual/estimator.py)

## Evidence

- [RESOURCE_RESULT.json#L1](RESOURCE_RESULT.json#L1)
- [TARGET_BLIND_DISCRIMINATOR_RECEIPT.json#L1](TARGET_BLIND_DISCRIMINATOR_RECEIPT.json#L1)
- [VERIFICATION.json#L1](VERIFICATION.json#L1)

<details>
<summary>All result files and records (5)</summary>

- [RESOURCE_RESULT.json](RESOURCE_RESULT.json) · result
- [TARGET_BLIND_DISCRIMINATOR_RECEIPT.json](TARGET_BLIND_DISCRIMINATOR_RECEIPT.json) · result
- [PREREGISTRATION.md](PREREGISTRATION.md) · record
- [VERIFICATION.json](VERIFICATION.json) · result
- [DERIVATION_AND_BOUNDARY.md](DERIVATION_AND_BOUNDARY.md) · record

</details>

## Related work

- supersedes: [Full-budget sampling: whitened paired draws and Sobol improved final-layer error but worsened all-layer error](../full-budget-sampling-whitened-paired-draws-sobol-improved-final/README.md)

[Research graph](../../../graph/index.html#node=2026-08-23-matched-cost-random-control-variate-lattice-sampling-comparators-metered) · [Copy provenance](../../../PROVENANCE.json)

