# Submission #333607: learned norm-coupling diagonal input with two fitted steps and fewer requests

2026-10-02T19:48:59Z · submission · adopted · measured

The submitted file adds one unit-specific norm-coupling diagonal input to the learned mean correction, with 128 first-layer weights after two line-search steps, while retaining the previous submitted dials and table regression. Shared strip views, setup constants and byte-identical request reductions accompany the input; all 100 networks grade successfully and the official public score becomes 1.6665844834926276e-09.

Recorded measurement:

    official public score1.6665844834926276e-09; utilisation0.11700587924982755; 100/100; wall mean77.487s/max83.593s; residual mean0.307s/max0.356s; ledger final-layer MSE1.424e-8,all-layer5.633e-9; native ten ranks-2.15%,10/10,bill-0.03%; stageC43,470/42,573 requests; stageD55,367 handles,3.44GB,zero added swap; slow setup+1.80%

Public score: 1.6665844834926276e-09. Status: GRADED. Failed networks: 0.

[Run or rebuild the designated estimators](../../estimator/REPRODUCE.md).

## Programs

- [estimator.py](estimator.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L444](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L444)
- [TEAM_LOG_2026-10-01_to_10-22.md#L448](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L448)
- [TEAM_LOG_2026-10-01_to_10-22.md#L450](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L450)
- [TEAM_LOG_2026-10-01_to_10-22.md#L451](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L451)
- [TEAM_LOG_2026-10-01_to_10-22.md#L457](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L457)
- [TEAM_LOG_2026-10-01_to_10-22.md#L472](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L472)
- [SUBMISSION_LEDGER.md#L919](../../records/SUBMISSION_LEDGER.md#L919)
- [SUBMISSION_LEDGER.md#L920](../../records/SUBMISSION_LEDGER.md#L920)

<details>
<summary>All result files and records (7)</summary>

- [submission.tar.gz](submission.tar.gz) · archive
- [TEAM_LOG_2026-10-01_to_10-22.md](../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record
- [attempt.json](attempt.json) · record
- [receipt.json](receipt.json) · record
- [eval_333607.json](eval_333607.json) · result
- [SUBMISSION_LEDGER.md](../../records/SUBMISSION_LEDGER.md) · record
- [eval_333607-c89d6bc6.json](eval_333607-c89d6bc6.json) · result

</details>

## Related work

- ← shipped in: [Building constants during setup saves 452 first-network requests with byte-identical predictions](../../experiments/setup-budget/building-constants-during-setup-saves-452-first-network-requests/README.md)
- ← shipped in: [Norm-coupling diagonal as a learned input lowers native MSE 2.15% on ten networks, all ten improved](../../experiments/learned-correction/norm-coupling-diagonal-learned-input-lowers-native-mse-2/README.md)
- ← shipped in: [Sharing strip-kernel leaf views per tree saves 146 / 122 index requests with identical predictions](../../experiments/arithmetic-kernels/sharing-strip-kernel-leaf-views-per-tree-saves-146/README.md)

[Research graph](../../graph/index.html#node=submission-333607) · [Copy provenance](../../PROVENANCE.json)

