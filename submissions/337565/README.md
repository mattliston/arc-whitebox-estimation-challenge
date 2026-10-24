# Submission #337565: 705-column gate readout at strength 1.5 and grouped-read cap 3

2026-10-21T16:45:31Z · submission · adopted · measured

The extended on-policy readout used gate features at strength 1.5 and a three-read grouped-kernel cap. The cap supplied measured runtime margin and all hosted cases passed.

Recorded measurement:

    Official submission record: {"submission_id": 337565, "date": "2026-10-21T16:45:31Z", "status": "GRADED", "score_public": 1.4036387231743654e-09, "n": 100, "failed_networks": 0, "utilization": 0.1087067432132153, "wall_mean_s": 97.584, "wall_max_s": 111.101, "resid_mean_s": 0.341, "resid_max_s": 0.37, "source_sha256": "8b662ff435cb6fad5539d02e124063f31e32fc773dae6486494a4294beb2bf17", "archive_sha256": "2b5f0e143fc9f3429a3f67e618da111661cc02d3b5658341268ee581f1897e62"}
    Ledger: ### 2026-10-21T17:42:50Z [fable-lead] #337565 GRADED 1.4036387231743654e-9, 100/100 clean, zero failures (hosted -0.40% vs #337506; wall max 111.1 s, mean 97.6 s, residual max 0.370). NEW BEST; J2W gap +0.88%. BEST pointer promoted (prior preserved as BEST_GRADED_REFERENCE_before337565_*). 5 shots remain; next = sub-1.40e-9 with margin; cap4-gate held (FIRE_cap4_gate.sh ready).
    ### 2026-10-21T21:14:52Z [fable-lead] #337617 GRADED 1.402028834390547e-9, 100/100 clean (hidden x gate readout ridge .03 x1.5 + cap3, XSG OFF; raw -0.195% vs #337565 on the 50 public nets, util .10879). NEW BEST (-0.11% vs #337565), third place, XSG-free. #337619 (hg+fh + cap4 + 8e9 clean) at 77/100, public half ~1.4005e-9 expected.
    ### 2026-10-21T21:21:40Z [fable-lead] #337619 GRADED 1.4005566073702658e-9, 100/100 clean, zero failures (hg+fh readout ridge .03 x1.5 + Smirnov cap4 + size-8e9 drop, XSG OFF; raw -0.166% vs #337565 on the 50 public nets, util .10865; max wall 110.3 s, residual max .362). NEW BEST (-0.22% vs #337565), third place, XSG-free. Top two public = #337619 + #337617 (both clean). 3 shots left (Matt's count).

Public score: 1.4036387231743654e-09. Status: GRADED. Failed networks: 0.

[Run or rebuild the designated estimators](../../estimator/REPRODUCE.md).

## Programs

- [estimator.py](estimator.py)

## Evidence

- [SUBMISSION_LEDGER.md#L1783](../../records/SUBMISSION_LEDGER.md#L1783)
- [SUBMISSION_LEDGER.md#L1785](../../records/SUBMISSION_LEDGER.md#L1785)
- [SUBMISSION_LEDGER.md#L1787](../../records/SUBMISSION_LEDGER.md#L1787)
- [SUBMISSION_LEDGER.md#L1791](../../records/SUBMISSION_LEDGER.md#L1791)
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

- ← shipped in: [705-column gate readout with cap 3: strength 1.5 gives -0.176% pooled adjusted across 76 observations](../../experiments/on-policy-readout/705-column-gate-readout-cap-3-strength-1-5/README.md)
- ← shipped in: [First-128 on-policy capture uses six source-specific parents and 540 captures plus 18 controls](../../instruments/first-128-policy-capture-uses-six-source-specific-parents/README.md)
- ← shipped in: [First-128 on-policy refit strength: zero added FLOPs and about -0.3% adjusted across native cohorts](../../experiments/on-policy-readout/first-128-policy-refit-strength-zero-added-flops-about/README.md)
- ← shipped in: [Smirnov read caps: cap 3 adds +0.257% adjusted cost and reduces wall time -11.0%](../../experiments/wall-and-residual/smirnov-read-caps-cap-3-adds-0-257-adjusted/README.md)
- ← shipped in: [Squared and gate readout features: -0.733% and -0.836% cross-validation at 577 and 705 columns](../../experiments/on-policy-readout/squared-gate-readout-features-0-733-0-836-cross/README.md)

[Research graph](../../graph/index.html#node=submission-337565) · [Copy provenance](../../PROVENANCE.json)

