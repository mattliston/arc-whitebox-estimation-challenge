# Submission #337226: Fourth-cumulant dial without the reversing correction; four hosted failures

2026-10-19T23:53:55Z · submission · failed · measured

Fourth-cumulant strength was increased without the separate correction that had reversed in submission #337219. Three networks exhausted wall time and one exceeded residual time.

Recorded measurement:

    Official submission record: {"submission_id": 337226, "date": "2026-10-19T23:53:55Z", "status": "GRADED", "score_public": 0.06308463346097808, "n": 100, "failed_networks": 4, "utilization": 0.10910307826146891, "wall_mean_s": 102.398, "wall_max_s": 118.265, "resid_mean_s": 0.349, "resid_max_s": 0.425, "source_sha256": "9c212fef8d0615ab5802a377c7f6e39361f0d01d5fde6ec79c5630350416f78b", "archive_sha256": "574251f5b45835c89184f79c65a186fe01bc949f8c4e6a2048c534de8fd1720e"}
    Ledger: - [fable-lead] 2026-10-20T00:37:07Z GRADED #337226 FAILED (public 0.06308): nets 1/17/28 time_exhausted, 33 residual .425; server slow on the public half (kernel median 94.0 vs 90.2 on the private half run later). On the 46 completed public nets GK4 x1.04 reads MSE -0.082% vs #337212 (lower 24/46): the piece is real at the +-0.1% noise level. Re-firing the same archive (574251f5b458, gates passed at 23:53Z) now that the server is back at normal speed; base = #337212 per Matt.

Public score: 0.06308463346097808. Status: GRADED. Failed networks: 4.

[Run or rebuild the designated estimators](../../estimator/REPRODUCE.md).

## Programs

- [estimator.py](estimator.py)

## Evidence

- [SUBMISSION_LEDGER.md#L1118](../../records/SUBMISSION_LEDGER.md#L1118)
- [SUBMISSION_LEDGER.md#L1120](../../records/SUBMISSION_LEDGER.md#L1120)

<details>
<summary>All result files and records (9)</summary>

- [submission.tar.gz](submission.tar.gz) · archive
- [SUBMISSION_LEDGER.md](../../records/SUBMISSION_LEDGER.md) · record
- [receipt.json](receipt.json) · record
- [attempt.json](attempt.json) · record
- [package.json](package.json) · result
- [validate.json](validate.json) · result
- [validate_package.json](validate_package.json) · result
- [stage_a.json](stage_a.json) · result
- [eval_337226.json](eval_337226.json) · result

</details>

## Related work

- ← supersedes: [Submission #337247: repeat of #337226 after platform slowdown, graded 1.4288769e-9](../337247/README.md)

[Research graph](../../graph/index.html#node=submission-337226) · [Copy provenance](../../PROVENANCE.json)

