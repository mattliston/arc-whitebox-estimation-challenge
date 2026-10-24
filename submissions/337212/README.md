# Submission #337212: combined zero-cost dial optima on #337142, graded 1.4296259e-9

2026-10-19T22:23:50Z · submission · adopted · measured

On the owner's instruction to fire the micro-stack before 00:00Z regardless of the 0.35% bar (unused daily slots lapse), candidate_rel_mstack1.py (the #337142 physics plus the micro-stack of flat-dial optima, zero bill, same request profile) was fired. It graded VALID 100/100 at exactly 1.4296258985e-9, -0.225% vs #337142 against a local read of -0.13%, crossing the 1.43e-9 bank level; the owner later fixed it as the base for all future submissions. The first setup gate failed at 5.026 s under four lanes plus an eval process and passed at 4.82 s with the eval paused.

Recorded measurement:

    exact 1.4296258985e-9 (-0.225% vs #337142 1.432830842e-9); util .109175, MSE 1.3091e-8; gaps after: J2W (1.418) -0.781%, oliver -0.788%, 1.41e-9 -1.373%; setup5 5.026 s FAIL then 4.82 s PASS; stage A 23 signatures none new; predict wall max 113 s; residual median .349 / max .367

Public score: 1.4296258985258333e-09. Status: GRADED. Failed networks: 0.

[Run or rebuild the designated estimators](../../estimator/REPRODUCE.md).

## Programs

- [estimator.py](estimator.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L6051](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L6051)
- [TEAM_LOG_2026-10-01_to_10-22.md#L6053](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L6053)
- [TEAM_LOG_2026-10-01_to_10-22.md#L6065](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L6065)
- [TEAM_LOG_2026-10-01_to_10-22.md#L6067](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L6067)
- [SUBMISSION_LEDGER.md#L1112](../../records/SUBMISSION_LEDGER.md#L1112)
- [SUBMISSION_LEDGER.md#L1114](../../records/SUBMISSION_LEDGER.md#L1114)
- [SUBMISSION_LEDGER.md#L1115](../../records/SUBMISSION_LEDGER.md#L1115)

<details>
<summary>All result files and records (5)</summary>

- [submission.tar.gz](submission.tar.gz) · archive
- [receipt.json](receipt.json) · record
- [eval_337212.json](eval_337212.json) · result
- [SUBMISSION_LEDGER.md](../../records/SUBMISSION_LEDGER.md) · record
- [TEAM_LOG_2026-10-01_to_10-22.md](../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

## Related work

- ← shipped in: [Combined zero-cost dial optima: -0.19% raw error on 28 networks](../../experiments/dials/combined-zero-cost-dial-optima-0-19-raw-error/README.md)
- ← shipped in: [Fourth-cumulant band scales shifted by 5%: all worse, with fitted optima within 1%](../../experiments/fourth-order-terms/fourth-cumulant-band-scales-shifted-5-all-worse-fitted/README.md)
- ← shipped in: [Historical release rule uses #337212 as the reference and favors earlier submissions during quiet platform periods](../../records/research/2026-10-20-historical-release-rule-uses-337212-reference-favors-earlier-submissions/README.md)
- supersedes: [Submission #337142: young dead-row crop z 3.0 extended to layer 13 on #337126, graded 1.4328308e-9 (-0.027%)](../337142/README.md)
- ← rejects: [Submission #337219: combined dials and fourth-order birth coefficient increase, graded 1.4317535e-9](../337219/README.md)

[Research graph](../../graph/index.html#node=submission-337212) · [Copy provenance](../../PROVENANCE.json)

