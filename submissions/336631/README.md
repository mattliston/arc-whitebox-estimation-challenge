# Submission #336631: depth peeling, rank-4 exact kernel and read-plan setup: startup timeout

2026-10-17T02:32:15Z · submission · failed · measured

Exact depth peeling, a rank-4 exact W3 kernel, a dial set and grouped strided-read plans were submitted with narrow setup and residual margins. No networks were evaluated; the official startup timeout reproduced locally at an unclipped 5.6208 s, showing that the prior 4.988 s projection omitted the complete startup envelope.

Recorded measurement:

    The submission failed during setup; no per-network score, utilization, wall or residual results were reported.

Public score: None. Status: FAILED. Failed networks: 0.

[Run or rebuild the designated estimators](../../estimator/REPRODUCE.md).

## Programs

- [estimator.py](estimator.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L4756](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4756)
- [TEAM_LOG_2026-10-01_to_10-22.md#L4780](../../records/TEAM_LOG_2026-10-01_to_10-22.md#L4780)
- [SUBMISSION_LEDGER.md#L1034](../../records/SUBMISSION_LEDGER.md#L1034)
- [SUBMISSION_LEDGER.md#L1036](../../records/SUBMISSION_LEDGER.md#L1036)
- [SUBMISSION_LEDGER.md#L1040](../../records/SUBMISSION_LEDGER.md#L1040)

<details>
<summary>All result files and records (8)</summary>

- [submission.tar.gz](submission.tar.gz) · archive
- [TEAM_LOG_2026-10-01_to_10-22.md](../../records/TEAM_LOG_2026-10-01_to_10-22.md) · log
- [SUBMISSION_LEDGER.md](../../records/SUBMISSION_LEDGER.md) · record
- [receipt.json](receipt.json) · result
- [eval_336631.json](eval_336631.json) · result
- [response.json](response.json) · result
- [attempt.json](attempt.json) · result
- [eval_summary.txt](eval_summary.txt) · record

</details>

## Related work

- ← enables: [Eleven runtime revisions reduce projected maximum residual from .430 s to .3989 s but nearly exhaust setup time](../../records/research/2026-10-17-eleven-runtime-revisions-reduce-projected-maximum-residual-430-s/README.md)

[Research graph](../../graph/index.html#node=submission-336631) · [Copy provenance](../../PROVENANCE.json)

