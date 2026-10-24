# Submission #335483: byte-identical scatter-based gather variant

2026-10-12T04:30:40Z · submission · adopted · measured

The submitted twin plus the scatter-gather routing, certified byte-identical on 7100136; the dead-cut tiles were dropped from the max-twin design because true-inner-size covariance sandwich (XSDM) already runs the sandwich at its true inner size.

Recorded measurement:

    1.54906e-9 on the public 50 (-0.224% vs #335411); final MSE 1.346e-8; share 0.115055; 100/100; residual max 0.35 s; bill -0.291% vs S48 on the native ten; P(loss) 4.2% per 100; official public score 1.5490655328382898e-09; official utilisation 0.11505468811774336; failed networks 0

Public score: 1.5490655328382898e-09. Status: GRADED. Failed networks: 0.

[Run or rebuild the designated estimators](../../estimator/REPRODUCE.md).

## Programs

- [estimator.py](estimator.py)
- [candidate.py](candidate.py)
- [../../instruments/variant-pipeline-checks-fragments-release-qualification-ten-native-comparisons/code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/engineer/a62_modules/s49/xsgF/candidate.py](../../instruments/variant-pipeline-checks-fragments-release-qualification-ten-native-comparisons/code/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/engineer/a62_modules/s49/xsgF/candidate.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L1277](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1277)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1280](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1280)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1285](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1285)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1286](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1286)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1300](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1300)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1325](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1325)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1326](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1326)
- [TEAM_LOG_2026-10-01_to_10-22.md#L1352](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L1352)
- [SUBMISSION_LEDGER.md#L943](../../records/SUBMISSION_LEDGER.md#L943)
- [SUBMISSION_LEDGER.md#L944](../../records/SUBMISSION_LEDGER.md#L944)
- [SUBMISSION_LEDGER.md#L947](../../records/SUBMISSION_LEDGER.md#L947)

<details>
<summary>All result files and records (6)</summary>

- [submission.tar.gz](submission.tar.gz) · archive
- [stage_a.json](stage_a.json) · record
- [receipt.json](receipt.json) · record
- [eval_335483.json](eval_335483.json) · result
- [SUBMISSION_LEDGER.md](../../records/SUBMISSION_LEDGER.md) · record
- [TEAM_LOG_2026-10-01_to_10-22.md](../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

## Related work

- ← shipped in: [Scatter-based gathers adopted in the higher-request variant: -0.074..-0.078% FLOPs, byte-identical](../../records/research/2026-10-12-scatter-based-gathers-adopted-higher-request-variant-0-074/README.md)
- ← supersedes: [Submission #335451: unfused-tree variant with its own measured prebind list](../335451/README.md)
- supersedes: [Submission #335505: carrier-strength refit with scatter-based gathers and an individual prebind list](../335505/README.md)

[Research graph](../../graph/index.html#node=submission-335483) · [Copy provenance](../../PROVENANCE.json)

