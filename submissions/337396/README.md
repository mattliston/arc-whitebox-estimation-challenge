# Submission #337396: Mean and carrier precision with copy removal; two residual failures

2026-10-20T20:26:20Z · submission · failed · measured

Mean/carrier precision and Gram changes were combined with late join-sketch columns and copy removal. Two hosted residual failures invalidated the improved local score.

Recorded measurement:

    Official submission record: {"submission_id": 337396, "date": "2026-10-20T20:26:20Z", "status": "GRADED", "score_public": 0.03462488788017601, "n": 100, "failed_networks": 2, "utilization": 0.10871736722646347, "wall_mean_s": 102.16, "wall_max_s": 119.086, "resid_mean_s": 0.357, "resid_max_s": 0.416, "source_sha256": "4a40feb6169ebc523f38e217e8d2727b9a4fb54ea5d7531ad2fc945df227544b", "archive_sha256": "73e3d0af6c898dd00bcfc33813d9578da6a7ee7abb30fe7659536c6d44bd23a9"}
    Ledger: [astra-lead] 2026-10-20T20:32:34Z #337393 GRADEDVALID100/100/0fails; exactpublicscore1.4272337950135872e-9, NEW BEST and reportingreference. Need.786137%furtherforJ2W. #337396 submitted20:26:22Z(1:26p.m.Pacific),tagastra_micro_kclm_copyall_4a40feb6_v1,archive73e3d0af6c898dd00bcfc33813d9578da6a7ee7abb30fe7659536c6d44bd23a9; UTC20attempts4/10. Full32localgainvs337393-.1474441%,held-.1703717%; originalgatesPASS,gradingpending. Lastfiredarchive337396. Inheritedrescuewasfoundliveunderdiagnosticload; disclosedtoMatt, futurebuildspermanentlydisableperNOgovernors; firedsourcesimmutable.

Public score: 0.03462488788017601. Status: GRADED. Failed networks: 2.

[Run or rebuild the designated estimators](../../estimator/REPRODUCE.md).

## Programs

- [estimator.py](estimator.py)

## Evidence

- [SUBMISSION_LEDGER.md#L1153](../../records/SUBMISSION_LEDGER.md#L1153)

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
- [eval_337396.json](eval_337396.json) · result

</details>

## Related work

- ← shipped in: [Exact copy removal cuts warm FLOPs 0.3332024% with byte-identical predictions on network 35](../../experiments/arithmetic-kernels/exact-copy-removal-cuts-warm-flops-0-3332024-byte/README.md)
- ← shipped in: [Gram arithmetic plus float32 matrix-vector changes: -0.206095% adjusted on 32 networks](../../experiments/arithmetic-kernels/gram-arithmetic-plus-float32-matrix-vector-changes-0-206095/README.md)

[Research graph](../../graph/index.html#node=submission-337396) · [Copy provenance](../../PROVENANCE.json)

