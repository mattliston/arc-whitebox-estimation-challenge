# Submission #336101: rank-two learned output projection with planner request repair, public score 1.50514e-9

2026-10-14T21:23:30Z · submission · adopted · measured

The GOP2 rank-2 output projection of the learned correction was fired on top of #336075 after the LVFS-only request repair. It became the new best by 0.53% and the biggest single step of the day; its physics was absorbed into the refit line from round S62.

Recorded measurement:

    official 1.505139747652245e-9 (new best), -0.530556991% vs #336075; paired 50 log -0.532870182859237% (SE 0.04947962146205677%, 46/50 lower); 100/100, 0 failed; science 24 adjusted -0.571426%; seal 8 -0.644465% (SE log 0.141959%); production risk 13.38%, first/warm 47,403/45,630, setup 4.312 s, worst 93.5 s; gaps after: Luna 0.251301789%, Oliver 1.842008169%

Public score: 1.505139747652245e-09. Status: GRADED. Failed networks: 0.

[Run or rebuild the designated estimators](../../estimator/REPRODUCE.md).

## Programs

- [estimator.py](estimator.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L3104](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3104)
- [TEAM_LOG_2026-10-01_to_10-22.md#L3138](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3138)
- [TEAM_LOG_2026-10-01_to_10-22.md#L3140](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3140)
- [SUBMISSION_LEDGER.md#L973](../../records/SUBMISSION_LEDGER.md#L973)
- [SUBMISSION_LEDGER.md#L975](../../records/SUBMISSION_LEDGER.md#L975)

<details>
<summary>All result files and records (8)</summary>

- [submission.tar.gz](submission.tar.gz) · archive
- [receipt.json](receipt.json) · record
- [attempt.json](attempt.json) · record
- [eval_336101.json](eval_336101.json) · result
- [stage_a.json](stage_a.json) · record
- [ROOT_PUBLIC_GRADE_v1.json](ROOT_PUBLIC_GRADE_v1.json) · result
- [SUBMISSION_LEDGER.md](../../records/SUBMISSION_LEDGER.md) · record
- [TEAM_LOG_2026-10-01_to_10-22.md](../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

## Related work

- ← enables: [Output projection exceeds request risk 23.16%; removing a planner limit reduces it to 13.38%](../../records/research/2026-10-14-output-projection-exceeds-request-risk-23-16-removing-planner/README.md)
- ← shipped in: [Output projection exceeds request risk 23.16%; removing a planner limit reduces it to 13.38%](../../records/research/2026-10-14-output-projection-exceeds-request-risk-23-16-removing-planner/README.md)
- ← shipped in: [Rank-two output projection of the learned correction: fresh-16 adjusted -0.577%, out-of-sample -0.560%](../../experiments/learned-correction/rank-two-output-projection-learned-correction-fresh-16-adjusted/README.md)
- ← shipped in: [Feasible mean and rank-two output projection absorbed into refit rounds 61 and 62](../../records/research/2026-10-15-feasible-mean-rank-two-output-projection-absorbed-into-refit/README.md)
- ← enables: [Submission #336075: feasible head mean, dense covariance reuse and Gaussian T22 reuse on #335920, 1.51317e-9](../336075/README.md)
- enables: [Submission #336119: same-physics safe fallback for #336101 with two-strip covariance request geometry, +0.64% vs primary](../336119/README.md)
- enables: [Submission #336125: young plane read at contraction depth 7 with exact setup prebinding on #336101, 1.50435e-9](../336125/README.md)
- ← shipped in: [Submission #336119: same-physics safe fallback for #336101 with two-strip covariance request geometry, +0.64% vs primary](../336119/README.md)

[Research graph](../../graph/index.html#node=submission-336101) · [Copy provenance](../../PROVENANCE.json)

