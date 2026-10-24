# Full-birth third-cumulant carrier with recursive mean feedback: final raw error 9.340196551564291e-7

2026-08-24T02:52Z · experiment · adopted · measured

The third-cumulant canonical-polyadic factor carrier (CP3) retained every generated factor up to cap 16384 and was compared with two mean-feedback variants on 100 paired networks. Recursively carrying the reported mean reduced endpoint error to a quarter of the cap-1792 reference; the remaining discrepancy could not be attributed to factor-rank truncation.

Recorded measurement:

    resource rungs: cap 2048 229,196,233,730 FLOPs (0.10422638012187235), cap 4096 336,972,195,842 (0.15323721338154428), cap 8192 500,555,071,490 (0.22762609273286216), cap 16384 620,090,560,514 (0.28198453970344417); mini100 final raw MSE: cap16384 3.5989996088739297e-6 (ratio 0.9668486121097386 vs cp1792), layer-2 mean carry 3.2909202609834e-6 (0.884085032699173), recursive mean carry 9.340196551564291e-7 (0.2509185066440621; all-layer 5.293963582531012e-7, ratio 0.27566791853998335); paired candidate wins 100/100 for all three; incumbent cp1792 3.722402415224689e-6

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/cp3_full_birth_cap_ladder/estimator.py](code/cp3_full_birth_cap_ladder/estimator.py)
- [code/cp3_cap16384_mean_recurrence_ablation/estimator.py](code/cp3_cap16384_mean_recurrence_ablation/estimator.py)
- [code/fullbudget_joint_score_v2/score_fullbudget_joint_once_v2.py](code/fullbudget_joint_score_v2/score_fullbudget_joint_once_v2.py)

## Evidence

- [fullbudget-joint-mini100-v2.joint-raw-mse.json#L1](../../../instruments/target-custody-tool-one-use-sealed-scoring-target-blind/fullbudget-joint-mini100-v2.joint-raw-mse.json#L1)
- [SYNTHETIC_RESOURCE_LADDER.json#L1](SYNTHETIC_RESOURCE_LADDER.json#L1)
- [FULL_BUDGET_CLOSURE_DIRECTIVE.md#L31](../../../records/research/2026-08-23-width-equivariant-residual-observer-32-channels-over-four-weight/FULL_BUDGET_CLOSURE_DIRECTIVE.md#L31)

<details>
<summary>All result files and records (6)</summary>

- [SYNTHETIC_RESOURCE_LADDER.json](SYNTHETIC_RESOURCE_LADDER.json) · result
- [PREREGISTRATION.md](PREREGISTRATION.md) · record
- [SYNTHETIC_RESOURCE_RESULT.json](SYNTHETIC_RESOURCE_RESULT.json) · result
- [fullbudget-joint-mini100-v2.joint-raw-mse.json](../../../instruments/target-custody-tool-one-use-sealed-scoring-target-blind/fullbudget-joint-mini100-v2.joint-raw-mse.json) · result
- [INDEPENDENT_POST_OPEN_AUDIT_V2.json](INDEPENDENT_POST_OPEN_AUDIT_V2.json) · result
- [FULL_BUDGET_CLOSURE_DIRECTIVE.md](../../../records/research/2026-08-23-width-equivariant-residual-observer-32-channels-over-four-weight/FULL_BUDGET_CLOSURE_DIRECTIVE.md) · record

</details>

## Related work

- supersedes: [Third-cumulant carrier caps 512–1792: small endpoint gains, layer-2 error saturated at cap 1024](../third-cumulant-carrier-caps-5121792-small-endpoint-gains-layer/README.md)
- ← measured by: [Full-budget sampling: whitened paired draws and Sobol improved final-layer error but worsened all-layer error](../../sampling-estimators/full-budget-sampling-whitened-paired-draws-sobol-improved-final/README.md)
- ← enables: [Sampling-free closure accuracy target below 1e-8 raw error using up to the full 2^41 budget](../../../records/research/2026-08-24-sampling-free-closure-accuracy-target-below-1e-8-raw/README.md)
- ← supersedes: [Third-cumulant truncation deferred through layer 3: +1,049,527,296 FLOPs; replaced by untruncated carry](../third-cumulant-truncation-deferred-through-layer-3-1-049/README.md)

[Research graph](../../../graph/index.html#node=2026-08-24-full-birth-third-cumulant-carrier-recursive-mean-feedback-final) · [Copy provenance](../../../PROVENANCE.json)

