# Submission #335954: same-physics safe fallback for #335920 with fewer requests and bill +0.43 to +0.54%, 1.52151e-9

2026-10-14T05:51:11Z · submission · adopted · measured

A same-physics safe variant of #335920 (identical estimator, a request-lean runtime with its own 20-network request census and a repaired signature list) was fired as the designation fallback. It trades requests for bill: about 3,900 fewer requests per network at a 0.43 to 0.54% higher bill, scoring 0.46% worse than the primary. An earlier build of it was held because the prefire check found two new request signatures versus the fired twin.

Recorded measurement:

    official 1.5215140004221161e-9, +0.461969% vs #335920; paired 50 meanlog +0.460420% (SE 0.035770%, 0/50 lower); final MSE -0.005%, bill meanlog +0.467%; residual max 0.371 s, setup 4.370 s; native ten vs the twin: MSE pooled -0.07%, requests -3,941/-3,588 (networks 35/100), bill +0.428%/+0.540%, score about +0.41%; release gate: first/warm 43,202/41,843 requests, P(loss) 2.80% per 100 networks, projected worst wall 92.9 s

Public score: 1.5215140004221161e-09. Status: GRADED. Failed networks: 0.

[Run or rebuild the designated estimators](../../estimator/REPRODUCE.md).

## Programs

- [estimator.py](estimator.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L2589](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2589)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2616](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2616)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2666](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2666)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2716](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2716)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2725](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2725)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2728](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2728)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2781](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2781)
- [SUBMISSION_LEDGER.md#L969](../../records/SUBMISSION_LEDGER.md#L969)
- [SUBMISSION_LEDGER.md#L970](../../records/SUBMISSION_LEDGER.md#L970)

<details>
<summary>All result files and records (10)</summary>

- [submission.tar.gz](submission.tar.gz) · archive
- [receipt.json](receipt.json) · record
- [attempt.json](attempt.json) · record
- [eval_335954.json](eval_335954.json) · result
- [stage_a.json](stage_a.json) · record
- [quick_s56k1s_repair1r1_a.json](../../records/research/2026-10-14-release-gate-version-4-checks-requests-binding-phase-setup/quick_s56k1s_repair1r1_a.json) · result
- [ROOT_PUBLIC_SAFE_GRADE_v1.json](ROOT_PUBLIC_SAFE_GRADE_v1.json) · result
- [ROOT_DESIGNATION_PAIR_v1.json](ROOT_DESIGNATION_PAIR_v1.json) · record
- [SUBMISSION_LEDGER.md](../../records/SUBMISSION_LEDGER.md) · record
- [TEAM_LOG_2026-10-01_to_10-22.md](../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

## Related work

- ← enables: [Submission #335920: bank-product route fusion, fresh-buffer copying and quarter input loading, bill -0.273%](../335920/README.md)
- shipped in: [Submission #335920: bank-product route fusion, fresh-buffer copying and quarter input loading, bill -0.273%](../335920/README.md)

[Research graph](../../graph/index.html#node=submission-335954) · [Copy provenance](../../PROVENANCE.json)

