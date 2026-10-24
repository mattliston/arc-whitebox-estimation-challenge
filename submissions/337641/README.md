# Submission #337641: Final 961-column readout at strength 1.35 with residual-meter references removed

2026-10-21T22:29:33Z · submission · adopted · measured

The final source changes readout strength to 1.35 and removes three inactive residual-meter functions while retaining direct gather and the cap-four size-limited kernel. Its public score marginally trails #337619 and the repository owner designated both for private re-evaluation.

Recorded measurement:

    Official submission record: {"submission_id": 337641, "date": "2026-10-21T22:29:33Z", "status": "GRADED", "score_public": 1.4006490071107744e-09, "n": 100, "failed_networks": 0, "utilization": 0.10865887533043861, "wall_mean_s": 98.678, "wall_max_s": 109.745, "resid_mean_s": 0.336, "resid_max_s": 0.363, "source_sha256": "5a4c15405c4462703aeead1bb6ccbc9b98ccf6ba2b5974ce23713303c87d594d", "archive_sha256": "e0a79ccb4c52197659ab321974ae4a4d0fd6e5d161fffc81a6ab4d13a6f93e75"}
    Ledger: ### 2026-10-21T23:35:16Z [fable-lead] #337641 GRADED 1.4006490071107744e-9, 100/100 clean, zero failures (hg+fh readout ridge .03 at strength x1.35 + Smirnov cap4 + size-8e9 drop, XSG OFF, retired governor removed; util .108659, final-layer MSE 1.289e-8, max wall 109.7 s, residual max .363). vs #337619 1.4005566e-9: +0.0066% = dead tie; BEST stays #337619. The x1.35 strength change was a wash on the hosted public set. Designations for the private re-eval = #337619 + #337641 (the two best, both clean). 2 shots held unfired at Matt's decision (private re-eval makes public re-rolls pointless); gated-and-ready archives if needed: x1.28 70e53586da53, x1.35 late-rows f4d2be7f3bd3.

Public score: 1.4006490071107744e-09. Status: GRADED. Failed networks: 0.

[Run or rebuild the designated estimators](../../estimator/REPRODUCE.md).

## Programs

- [estimator.py](estimator.py)
- [../../estimator/submission_337641.py](../../estimator/submission_337641.py)

## Evidence

- [SUBMISSION_LEDGER.md#L1797](../../records/SUBMISSION_LEDGER.md#L1797)
- [SUBMISSION_LEDGER.md#L1799](../../records/SUBMISSION_LEDGER.md#L1799)
- [SUBMISSION_LEDGER.md#L1801](../../records/SUBMISSION_LEDGER.md#L1801)
- [SUBMISSION_LEDGER.md#L1805](../../records/SUBMISSION_LEDGER.md#L1805)

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

- ← shipped in: [961-column readout with cap 4 and size threshold: -0.620% Linux and -0.423% Mac before accounting repair](../../experiments/on-policy-readout/961-column-readout-cap-4-size-threshold-0-620/README.md)
- ← shipped in: [Final designated source removes all residual-meter references from three inactive function bodies](../../records/research/2026-10-21-final-designated-source-removes-all-residual-meter-references-three/README.md)
- ← shipped in: [Final readout strength 1.35 improves Mac and Linux panels; strengths 1.05 through 1.65 give a broad optimum](../../experiments/on-policy-readout/final-readout-strength-1-35-improves-mac-linux-panels/README.md)
- ← shipped in: [First-128 on-policy capture uses six source-specific parents and 540 captures plus 18 controls](../../instruments/first-128-policy-capture-uses-six-source-specific-parents/README.md)
- ← shipped in: [Hidden times gate plus first-128 times hidden: -0.995% held-fold proxy at 961 columns](../../experiments/on-policy-readout/hidden-times-gate-plus-first-128-times-hidden-0/README.md)
- ← shipped in: [Read cap 4 plus an 8e9 matrix-size threshold matches cap-3 slow-network margin with fewer FLOPs](../../experiments/wall-and-residual/read-cap-4-plus-8e9-matrix-size-threshold-matches/README.md)
- ← shipped in: [Scatter-for-gather disabled in every subsequent submission after fair-accounting review](../../records/research/2026-10-21-scatter-gather-disabled-every-subsequent-submission-after-fair-accounting/README.md)
- ← shipped in: [Smirnov read caps: cap 3 adds +0.257% adjusted cost and reduces wall time -11.0%](../../experiments/wall-and-residual/smirnov-read-caps-cap-3-adds-0-257-adjusted/README.md)
- ← shipped in: [Submissions #337619 and #337641 selected for private re-evaluation with direct gather enabled](../../records/research/2026-10-21-submissions-337619-337641-selected-private-re-evaluation-direct-gather/README.md)

[Research graph](../../graph/index.html#node=submission-337641) · [Copy provenance](../../PROVENANCE.json)

