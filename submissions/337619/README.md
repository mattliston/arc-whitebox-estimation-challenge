# Submission #337619: 961-column on-policy readout, grouped-read cap 4 and 8e9 size threshold

2026-10-21T20:39:06Z · submission · adopted · measured

The correction combines hidden-times-gate and first-128-times-hidden features at ridge 0.03 and strength 1.5. Four grouped reads and an 8e9 matrix-size threshold retain margin with direct gather; this is the final best public source and one of the two designated estimators.

Recorded measurement:

    Official submission record: {"submission_id": 337619, "date": "2026-10-21T20:39:06Z", "status": "GRADED", "score_public": 1.4005566073702658e-09, "n": 100, "failed_networks": 0, "utilization": 0.10865223953959685, "wall_mean_s": 98.587, "wall_max_s": 110.304, "resid_mean_s": 0.334, "resid_max_s": 0.362, "source_sha256": "f66cb28baf0c018f25bb749d14ce9e9aedbb47b8f2acd408680cad169485d557", "archive_sha256": "55a93ef002c20419b17c53d1f4bfc1fd7269d9da97d0b6f6fdbbc7086d4a7017"}
    Ledger: ### 2026-10-21T21:14:52Z [fable-lead] #337617 GRADED 1.402028834390547e-9, 100/100 clean (hidden x gate readout ridge .03 x1.5 + cap3, XSG OFF; raw -0.195% vs #337565 on the 50 public nets, util .10879). NEW BEST (-0.11% vs #337565), third place, XSG-free. #337619 (hg+fh + cap4 + 8e9 clean) at 77/100, public half ~1.4005e-9 expected.
    ### 2026-10-21T21:21:40Z [fable-lead] #337619 GRADED 1.4005566073702658e-9, 100/100 clean, zero failures (hg+fh readout ridge .03 x1.5 + Smirnov cap4 + size-8e9 drop, XSG OFF; raw -0.166% vs #337565 on the 50 public nets, util .10865; max wall 110.3 s, residual max .362). NEW BEST (-0.22% vs #337565), third place, XSG-free. Top two public = #337619 + #337617 (both clean). 3 shots left (Matt's count).
    ### 2026-10-21T23:35:16Z [fable-lead] #337641 GRADED 1.4006490071107744e-9, 100/100 clean, zero failures (hg+fh readout ridge .03 at strength x1.35 + Smirnov cap4 + size-8e9 drop, XSG OFF, retired governor removed; util .108659, final-layer MSE 1.289e-8, max wall 109.7 s, residual max .363). vs #337619 1.4005566e-9: +0.0066% = dead tie; BEST stays #337619. The x1.35 strength change was a wash on the hosted public set. Designations for the private re-eval = #337619 + #337641 (the two best, both clean). 2 shots held unfired at Matt's decision (private re-eval makes public re-rolls pointless); gated-and-ready archives if needed: x1.28 70e53586da53, x1.35 late-rows f4d2be7f3bd3.

Public score: 1.4005566073702658e-09. Status: GRADED. Failed networks: 0.

[Run or rebuild the designated estimators](../../estimator/REPRODUCE.md).

## Programs

- [estimator.py](estimator.py)
- [../../estimator/submission_337619.py](../../estimator/submission_337619.py)

## Evidence

- [SUBMISSION_LEDGER.md#L1791](../../records/SUBMISSION_LEDGER.md#L1791)
- [SUBMISSION_LEDGER.md#L1793](../../records/SUBMISSION_LEDGER.md#L1793)
- [SUBMISSION_LEDGER.md#L1795](../../records/SUBMISSION_LEDGER.md#L1795)
- [SUBMISSION_LEDGER.md#L1797](../../records/SUBMISSION_LEDGER.md#L1797)
- [SUBMISSION_LEDGER.md#L1799](../../records/SUBMISSION_LEDGER.md#L1799)
- [SUBMISSION_LEDGER.md#L1801](../../records/SUBMISSION_LEDGER.md#L1801)
- [SUBMISSION_LEDGER.md#L1805](../../records/SUBMISSION_LEDGER.md#L1805)

<details>
<summary>All result files and records (10)</summary>

- [submission.tar.gz](submission.tar.gz) · archive
- [SUBMISSION_LEDGER.md](../../records/SUBMISSION_LEDGER.md) · record
- [receipt.json](receipt.json) · record
- [attempt.json](attempt.json) · record
- [package.json](package.json) · result
- [validate.json](validate.json) · result
- [validate_package.json](validate_package.json) · result
- [stage_a.json](stage_a.json) · result
- [OFFICIAL_RULES_VERBATIM_PASTE_20261021.md](../../records/OFFICIAL_RULES_VERBATIM_PASTE_20261021.md) · record
- [VM_PULL_MANIFEST.md](../../records/VM_PULL_MANIFEST.md) · record

</details>

## Related work

- ← shipped in: [961-column readout with cap 4 and size threshold: -0.620% Linux and -0.423% Mac before accounting repair](../../experiments/on-policy-readout/961-column-readout-cap-4-size-threshold-0-620/README.md)
- ← shipped in: [First-128 on-policy capture uses six source-specific parents and 540 captures plus 18 controls](../../instruments/first-128-policy-capture-uses-six-source-specific-parents/README.md)
- ← shipped in: [Hidden times gate plus first-128 times hidden: -0.995% held-fold proxy at 961 columns](../../experiments/on-policy-readout/hidden-times-gate-plus-first-128-times-hidden-0/README.md)
- ← shipped in: [Read cap 4 plus an 8e9 matrix-size threshold matches cap-3 slow-network margin with fewer FLOPs](../../experiments/wall-and-residual/read-cap-4-plus-8e9-matrix-size-threshold-matches/README.md)
- ← shipped in: [Scatter-for-gather disabled in every subsequent submission after fair-accounting review](../../records/research/2026-10-21-scatter-gather-disabled-every-subsequent-submission-after-fair-accounting/README.md)
- ← shipped in: [Smirnov read caps: cap 3 adds +0.257% adjusted cost and reduces wall time -11.0%](../../experiments/wall-and-residual/smirnov-read-caps-cap-3-adds-0-257-adjusted/README.md)
- ← shipped in: [Submissions #337619 and #337641 selected for private re-evaluation with direct gather enabled](../../records/research/2026-10-21-submissions-337619-337641-selected-private-re-evaluation-direct-gather/README.md)

[Research graph](../../graph/index.html#node=submission-337619) · [Copy provenance](../../PROVENANCE.json)

