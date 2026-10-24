# Submission #337534: Tensor-only on-policy second readout fit; two hosted wall failures

2026-10-21T14:35:23Z · submission · failed · measured

The first-128 source received a tensor-only second readout fit with no added operation profile. Two public cases exceeded the wall limit during a measured hosted slowdown.

Recorded measurement:

    Official submission record: {"submission_id": 337534, "date": "2026-10-21T14:35:23Z", "status": "GRADED", "score_public": 0.035095570249651736, "n": 100, "failed_networks": 2, "utilization": 0.108289224732016, "wall_mean_s": 101.435, "wall_max_s": 115.155, "resid_mean_s": 0.332, "resid_max_s": 0.381, "source_sha256": "8185cf0b6dfa4a9abb95e89d1ba1a0a87aa3160a9ab8cc8dfd3eb868cc1c7240", "archive_sha256": "b506f0ecbdf5ebaa21dfe66d0ad2c602630701895c725588d1c637c24c5aeb7e"}
    Ledger: ### 2026-10-21T15:21:38Z [fable-lead] #337534 FINAL: GRADED score 0.0351 with 2 public TIME_EXHAUSTED failures (idx 3,4) -> excluded from board; BEST stays #337506. Release archive for s1.5 refit prepared NOT fired: tag fable_onpolicy_s1p5_aaec3e8e_v1 archive 7cb0d37a6860 (gates clear). Extended fit relaunched unguarded BLAS4 at 15:20:28Z (guard group_timeout killed the first at 1800 s). Plan per Matt: [instruction removed]) then spend part on a cohort/age cut for >= -8% requests so the net stays <= ~1.404e-9 with margin.

Public score: 0.035095570249651736. Status: GRADED. Failed networks: 2.

[Run or rebuild the designated estimators](../../estimator/REPRODUCE.md).

## Programs

- [estimator.py](estimator.py)

## Evidence

- [SUBMISSION_LEDGER.md#L1777](../../records/SUBMISSION_LEDGER.md#L1777)
- [SUBMISSION_LEDGER.md#L1781](../../records/SUBMISSION_LEDGER.md#L1781)

<details>
<summary>All result files and records (8)</summary>

- [submission.tar.gz](submission.tar.gz) · archive
- [SUBMISSION_LEDGER.md](../../records/SUBMISSION_LEDGER.md) · record
- [receipt.json](receipt.json) · record
- [attempt.json](attempt.json) · record
- [package.json](package.json) · result
- [validate.json](validate.json) · result
- [validate_package.json](validate_package.json) · result
- [stage_a.json](stage_a.json) · result

</details>

## Related work

- ← shipped in: [First-128 on-policy capture uses six source-specific parents and 540 captures plus 18 controls](../../instruments/first-128-policy-capture-uses-six-source-specific-parents/README.md)
- ← shipped in: [First-128 on-policy refit strength: zero added FLOPs and about -0.3% adjusted across native cohorts](../../experiments/on-policy-readout/first-128-policy-refit-strength-zero-added-flops-about/README.md)

[Research graph](../../graph/index.html#node=submission-337534) · [Copy provenance](../../PROVENANCE.json)

