# Submission #337617: Hidden-times-gate readout and cap 3 with direct gather

2026-10-21T20:12:15Z · submission · adopted · measured

The previously setup-failed source was changed to use direct gather by disabling scatter-for-gather routing. The source accepted the additional metered cost and passed every hosted case.

Recorded measurement:

    Official submission record: {"submission_id": 337617, "date": "2026-10-21T20:12:15Z", "status": "GRADED", "score_public": 1.402028834390547e-09, "n": 100, "failed_networks": 0, "utilization": 0.10879292387507121, "wall_mean_s": 96.539, "wall_max_s": 107.318, "resid_mean_s": 0.334, "resid_max_s": 0.363, "source_sha256": "98563da1206bad75f02a370bb0b43ce7420f04989ded26ce686430ecd723ef14", "archive_sha256": "4d8a697f7e0eafd45d00deda9e19394128e78d466b48ede6c2343ceab20f89cc"}
    Ledger: ### 2026-10-21T21:14:52Z [fable-lead] #337617 GRADED 1.402028834390547e-9, 100/100 clean (hidden x gate readout ridge .03 x1.5 + cap3, XSG OFF; raw -0.195% vs #337565 on the 50 public nets, util .10879). NEW BEST (-0.11% vs #337565), third place, XSG-free. #337619 (hg+fh + cap4 + 8e9 clean) at 77/100, public half ~1.4005e-9 expected.
    ### 2026-10-21T21:21:40Z [fable-lead] #337619 GRADED 1.4005566073702658e-9, 100/100 clean, zero failures (hg+fh readout ridge .03 x1.5 + Smirnov cap4 + size-8e9 drop, XSG OFF; raw -0.166% vs #337565 on the 50 public nets, util .10865; max wall 110.3 s, residual max .362). NEW BEST (-0.22% vs #337565), third place, XSG-free. Top two public = #337619 + #337617 (both clean). 3 shots left (Matt's count).

Public score: 1.402028834390547e-09. Status: GRADED. Failed networks: 0.

[Run or rebuild the designated estimators](../../estimator/REPRODUCE.md).

## Programs

- [estimator.py](estimator.py)

## Evidence

- [SUBMISSION_LEDGER.md#L1789](../../records/SUBMISSION_LEDGER.md#L1789)
- [SUBMISSION_LEDGER.md#L1793](../../records/SUBMISSION_LEDGER.md#L1793)
- [SUBMISSION_LEDGER.md#L1795](../../records/SUBMISSION_LEDGER.md#L1795)

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

- ← shipped in: [Hidden times gate features improve native score; ridge 0.01 is worse than 0.03](../../experiments/on-policy-readout/hidden-times-gate-features-improve-native-score-ridge-0/README.md)
- ← shipped in: [Scatter-for-gather disabled in every subsequent submission after fair-accounting review](../../records/research/2026-10-21-scatter-gather-disabled-every-subsequent-submission-after-fair-accounting/README.md)
- ← shipped in: [Smirnov read caps: cap 3 adds +0.257% adjusted cost and reduces wall time -11.0%](../../experiments/wall-and-residual/smirnov-read-caps-cap-3-adds-0-257-adjusted/README.md)

[Research graph](../../graph/index.html#node=submission-337617) · [Copy provenance](../../PROVENANCE.json)

