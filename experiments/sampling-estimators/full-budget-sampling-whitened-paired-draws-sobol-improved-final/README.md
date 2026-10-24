# Full-budget sampling: whitened paired draws and Sobol improved final-layer error but worsened all-layer error

2026-08-24T02:52Z · experiment · informational · measured

Two sampling estimators spending the whole budget were sealed target-blind and scored once in the same mini100 opening as the closure variants: raw Sobol with 64512 samples and paired whitened iid sampling with 64000 samples. Both beat the recursive-mean-carry closure on final-layer MSE at 99% of budget, but the closure wins on all-layer MSE at 28% of budget. The comparison fixed the sampling frontier the closure had to pass and was not promoted, per the directive to rank closures by raw MSE and keep samplers as comparators.

Recorded measurement:

    iid_paired_whitened_n64000: final 6.648608832965987e-7, all-layer 9.300978554221737e-7, 2,187,012,965,375 FLOPs (0.9945383523586315); sobol_raw_n64512: final 7.555354468651681e-7, all-layer 1.147282471386777e-6, 2,183,394,289,696 FLOPs (0.992892769179889); closure cap16384_recursive_mean_carry final 9.340196551564291e-7, all-layer 5.293963582531012e-7 at 620,090,560,514 FLOPs; final ranking best to worst: iid whitened, Sobol, recursive mean carry, layer-2 mean carry, cap16384, incumbent cp1792

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/fullbudget_sampling_primary_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/estimator.py](code/estimator.py)

## Evidence

- [fullbudget-joint-mini100-v2.joint-raw-mse.json#L1](../../../instruments/target-custody-tool-one-use-sealed-scoring-target-blind/fullbudget-joint-mini100-v2.joint-raw-mse.json#L1)
- [ROOT_PREDICTION_SEAL_RELEASE_V1.json#L1](ROOT_PREDICTION_SEAL_RELEASE_V1.json#L1)

<details>
<summary>All result files and records (3)</summary>

- [PREREGISTRATION.md](PREREGISTRATION.md) · record
- [ROOT_PREDICTION_SEAL_RELEASE_V1.json](ROOT_PREDICTION_SEAL_RELEASE_V1.json) · record
- [fullbudget-joint-mini100-v2.joint-raw-mse.json](../../../instruments/target-custody-tool-one-use-sealed-scoring-target-blind/fullbudget-joint-mini100-v2.joint-raw-mse.json) · result

</details>

## Related work

- ← supersedes: [Matched-cost random, control-variate and lattice sampling comparators: metered without accuracy scoring](../matched-cost-random-control-variate-lattice-sampling-comparators-metered/README.md)
- measured by: [Full-birth third-cumulant carrier with recursive mean feedback: final raw error 9.340196551564291e-7](../../third-order-planes/full-birth-third-cumulant-carrier-recursive-mean-feedback-final/README.md)

[Research graph](../../../graph/index.html#node=2026-08-24-full-budget-sampling-whitened-paired-draws-sobol-improved-final) · [Copy provenance](../../../PROVENANCE.json)

