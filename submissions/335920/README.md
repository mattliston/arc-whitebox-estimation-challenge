# Submission #335920: bank-product route fusion, fresh-buffer copying and quarter input loading, bill -0.273%

2026-10-14T02:30:55Z · submission · adopted · measured

Three exact compute-only changes were stacked on the fired 56th-generation file: the UB-cluster route fusion (a planner change that keeps the UB route classes separate and retires a late layer-7 cache), a fresh-buffer copy chain for the plane bank, and loading a quarter of the input per loader (one extra charged copy op each). Predictions were identical to #335894 on all ten native networks; the public gain was entirely bill.

Recorded measurement:

    official 1.514517396424108e-9 (new best), -0.273367% vs #335894; paired public 50 meanlog -0.274069% (SE 0.012211%, 50/50 lower); final MSE unchanged, bill meanlog -0.273%; residual max 0.383 s, setup max 4.383 s; independent fixed-8 seal meanlog net -0.2537031196613863% (SE 0.01614989487165034%, 8/8 lower); native ten: MSE +0.00% on all ten, bill -0.304% mean (-0.375..-0.225%), charged ops +231 mean; first/warm requests 47,143/45,431 (+2,000/+1,598 vs #335894)

Public score: 1.514517396424108e-09. Status: GRADED. Failed networks: 0.

[Run or rebuild the designated estimators](../../estimator/REPRODUCE.md).

## Programs

- [estimator.py](estimator.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L2502](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2502)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2527](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2527)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2541](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2541)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2542](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2542)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2544](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2544)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2566](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2566)
- [SUBMISSION_LEDGER.md#L967](../../records/SUBMISSION_LEDGER.md#L967)
- [SUBMISSION_LEDGER.md#L968](../../records/SUBMISSION_LEDGER.md#L968)

<details>
<summary>All result files and records (11)</summary>

- [submission.tar.gz](submission.tar.gz) · archive
- [receipt.json](receipt.json) · record
- [attempt.json](attempt.json) · record
- [eval_335920.json](eval_335920.json) · result
- [stage_a.json](stage_a.json) · record
- [quick_s56ubq1tw_a.json](quick_s56ubq1tw_a.json) · result
- [ROOT_PUBLIC_GRADE_v1.json](ROOT_PUBLIC_GRADE_v1.json) · result
- [ASTRA_INITIAL8_OWNER_RETIREMENT_v1.json](ASTRA_INITIAL8_OWNER_RETIREMENT_v1.json) · record
- [ASTRA_STAGING_OWNER_RETIREMENT_v1.json](ASTRA_STAGING_OWNER_RETIREMENT_v1.json) · record
- [SUBMISSION_LEDGER.md](../../records/SUBMISSION_LEDGER.md) · record
- [TEAM_LOG_2026-10-01_to_10-22.md](../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

## Related work

- ← enables: [Submission #335894: refit-after-drop coefficients and certified compute fragment, public score 1.51867e-9](../335894/README.md)
- enables: [Submission #335954: same-physics safe fallback for #335920 with fewer requests and bill +0.43 to +0.54%, 1.52151e-9](../335954/README.md)
- enables: [Submission #336075: feasible head mean, dense covariance reuse and Gaussian T22 reuse on #335920, 1.51317e-9](../336075/README.md)
- ← shipped in: [Submission #335954: same-physics safe fallback for #335920 with fewer requests and bill +0.43 to +0.54%, 1.52151e-9](../335954/README.md)

[Research graph](../../graph/index.html#node=submission-335920) · [Copy provenance](../../PROVENANCE.json)

