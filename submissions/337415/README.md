# Submission #337415: Edge compression and exact copy removal with residual routing disabled; prediction failure

2026-10-20T22:23:53Z · submission · failed · measured

The compressed-edge and precision source permanently disabled residual-time routing. Although initial setup passed, a hosted prediction exhausted time; this was not an initial setup failure.

Recorded measurement:

    Official submission record: {"submission_id": 337415, "date": "2026-10-20T22:23:53Z", "status": "FAILED", "score_public": null, "n": 10, "failed_networks": 1, "utilization": null, "wall_mean_s": 107.754, "wall_max_s": 118.562, "resid_mean_s": 0.338, "resid_max_s": 0.394, "source_sha256": "4b8baf51fe5ced80f46907aea7ab4461124b993d17a178d73c9c75230a6100b7", "archive_sha256": "faf49095644316e9a5738fecd39cdce33d85b2e03287b9cfb5238e01ef16afa9"}

Public score: None. Status: FAILED. Failed networks: 1.

[Run or rebuild the designated estimators](../../estimator/REPRODUCE.md).

## Programs

- [estimator.py](estimator.py)

## Evidence

- [SUBMISSION_LEDGER.md#L1208-L1210](../../records/SUBMISSION_LEDGER.md#L1208-L1210)
- [eval_337415.json#L1](eval_337415.json#L1)

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
- [eval_337415.json](eval_337415.json) · result

</details>

## Related work

- ← shipped in: [Float32 arithmetic, exact copies and radical edge: -0.276832224% adjusted versus #337393](../../experiments/arithmetic-kernels/float32-arithmetic-exact-copies-radical-edge-0-276832224-adjusted/README.md)
- ← shipped in: [Radical learned-edge approximation: -0.1676% adjusted on four fresh networks](../../experiments/learned-correction/radical-learned-edge-approximation-0-1676-adjusted-four-fresh/README.md)

[Research graph](../../graph/index.html#node=submission-337415) · [Copy provenance](../../PROVENANCE.json)

