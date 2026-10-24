# Submission #328628: lifetime-packed depth-5 arena with corrected manifest failed without a score

2026-08-27T08:08:12Z · submission · failed · measured

The corrected official archive passed package validation, official local execution, and later exact official client/server A/B/A, but #328628 generic-failed without a score. This closes that archive attempt only; the failure was not reproduced as an estimator/runtime exception.

Recorded measurement:

    The corrected official archive passed package validation, official local execution, and later exact official client/server A/B/A, but #328628 generic-failed without a score. This closes that archive attempt only; the failure was not reproduced as an estimator/runtime exception. Official public record: {"status": "FAILED", "score_public": null, "n": 17, "failed_networks": 14, "utilization": null, "wall_mean_s": 113.974, "wall_max_s": 119.627, "resid_mean_s": 0.055, "resid_max_s": 0.234}

Public score: None. Status: FAILED. Failed networks: 14.

[Run or rebuild the designated estimators](../../estimator/REPRODUCE.md).

## Programs

- [estimator.py](estimator.py)

## Evidence

- [D18_OFFICIAL_ENV_FULL_GATE_RESULT.json#L1](../../experiments/arithmetic-kernels/unpacked-depth-5-arena-memory-boundary-rejected-this-form/D18_OFFICIAL_ENV_FULL_GATE_RESULT.json#L1)
- [SUBMIT_WATCH_OFFICIAL.log#L1](SUBMIT_WATCH_OFFICIAL.log#L1)
- [D22_REMOTE_RANK35_CLIENT_RESULT.json#L1](../../instruments/packed-depth-5-third-cumulant-arena-official-environment-runtime/D22_REMOTE_RANK35_CLIENT_RESULT.json#L1)
- [D23_D4_REMOTE_NATIVE_DRIFT.json#L1](../../experiments/wall-and-residual/depth-4-arena-runtime-control-network-35-official-client/D23_D4_REMOTE_NATIVE_DRIFT.json#L1)
- [D24_D5_OFFICIAL_REMOTE_ABA_RESULT.json#L1](../../instruments/packed-depth-5-third-cumulant-arena-official-environment-runtime/D24_D5_OFFICIAL_REMOTE_ABA_RESULT.json#L1)
- [ARCHIVE_ORCHESTRATION_BOUNDARY_RECEIPT.json#L1](../328627/ARCHIVE_ORCHESTRATION_BOUNDARY_RECEIPT.json#L1)

<details>
<summary>All result files and records (7)</summary>

- [D18_OFFICIAL_ENV_FULL_GATE_RESULT.json](../../experiments/arithmetic-kernels/unpacked-depth-5-arena-memory-boundary-rejected-this-form/D18_OFFICIAL_ENV_FULL_GATE_RESULT.json) · result
- [SUBMIT_WATCH_OFFICIAL.log](SUBMIT_WATCH_OFFICIAL.log) · log
- [D22_REMOTE_RANK35_CLIENT_RESULT.json](../../instruments/packed-depth-5-third-cumulant-arena-official-environment-runtime/D22_REMOTE_RANK35_CLIENT_RESULT.json) · result
- [D23_D4_REMOTE_NATIVE_DRIFT.json](../../experiments/wall-and-residual/depth-4-arena-runtime-control-network-35-official-client/D23_D4_REMOTE_NATIVE_DRIFT.json) · result
- [D24_D5_OFFICIAL_REMOTE_ABA_RESULT.json](../../instruments/packed-depth-5-third-cumulant-arena-official-environment-runtime/D24_D5_OFFICIAL_REMOTE_ABA_RESULT.json) · result
- [ARCHIVE_ORCHESTRATION_BOUNDARY_RECEIPT.json](../328627/ARCHIVE_ORCHESTRATION_BOUNDARY_RECEIPT.json) · result
- [SUBMISSION_LEDGER.md](../../records/SUBMISSION_LEDGER.md) · record

</details>

[Research graph](../../graph/index.html#node=submission-328628) · [Copy provenance](../../PROVENANCE.json)

