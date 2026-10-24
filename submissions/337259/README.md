# Submission #337259: Client synchronizations removed and fourth-cumulant strength increased

2026-10-20T02:05:05Z · submission · adopted · measured

This source retained the previous physics and retired 32 of 41 client-side synchronizations while preserving the fixed-panel predictions. The official public score differs slightly from the earlier exact-evaluation shorthand.

Recorded measurement:

    Official submission record: {"submission_id": 337259, "date": "2026-10-20T02:05:05Z", "status": "GRADED", "score_public": 1.4284668696282146e-09, "n": 100, "failed_networks": 0, "utilization": 0.10915610638397993, "wall_mean_s": 100.079, "wall_max_s": 113.439, "resid_mean_s": 0.349, "resid_max_s": 0.38, "source_sha256": "b41627a6cc4d3c67280ab85fd7400ff07a0a167621326fcdb07690abe84ccac7", "archive_sha256": "15acfaecdc546782f4441a4c0ce52d5c34969e66b33fb91b48bcf68ef3c1f73b"}
    Ledger: - [fable-lead] 2026-10-20T03:59:07Z GRADED #337259 (fable_v4a_b41627a6_v1 = #337212 physics with 32/41 client sync points retired + GK4 x1.04): VALID 100/100, 1.428440e-9 (public 1.428e-9), MSE 1.3080e-8, util 0.109156, residual median .348 p90 .361 max .380 (#337212 .349/.358/.367), wall max 113.4 s = NEW BEST VALID (-0.03% vs #337247; the sync lever is real on the platform: max residual -0.017 s). BASE for further work = candidate_rel_nosync_v1.py + GK4 (= candidate_rel_v4a.py b41627a6).

Public score: 1.4284668696282146e-09. Status: GRADED. Failed networks: 0.

[Run or rebuild the designated estimators](../../estimator/REPRODUCE.md).

## Programs

- [estimator.py](estimator.py)

## Evidence

- [SUBMISSION_LEDGER.md#L1122](../../records/SUBMISSION_LEDGER.md#L1122)
- [SUBMISSION_LEDGER.md#L1124](../../records/SUBMISSION_LEDGER.md#L1124)

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
- [eval_337259.json](eval_337259.json) · result

</details>

[Research graph](../../graph/index.html#node=submission-337259) · [Copy provenance](../../PROVENANCE.json)

