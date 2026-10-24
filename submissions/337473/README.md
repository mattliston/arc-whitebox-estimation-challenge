# Submission #337473: Hidden-complement readout; one hosted residual failure

2026-10-21T05:18:18Z · submission · failed · measured

Adding the per-layer hidden-complement head improved local accuracy but one hosted residual-time failure invalidated the submission.

Recorded measurement:

    Official submission record: {"submission_id": 337473, "date": "2026-10-21T05:18:18Z", "status": "GRADED", "score_public": 0.014044704624470581, "n": 100, "failed_networks": 1, "utilization": 0.10817423277769195, "wall_mean_s": 101.874, "wall_max_s": 116.832, "resid_mean_s": 0.364, "resid_max_s": 0.402, "source_sha256": "4685e1454ebd2d9607fcd88b64cc1d8435050761212efd1902b04d3f83151a46", "archive_sha256": "3848cfaed85175d0f0f817e99234201cc5736bad51e071eabc553b719145947a"}
    Ledger: #337473 is final GRADED with penalized score0.014044704624470581,99completed/1failed(index27 brandy-johnston RESIDUAL_WALL_TIME_EXHAUSTED0.401729281s); final observed05:56:32Z/rawSHAd9153627. All7 finalsetuprecordsSUCCESS; earlier5setupcount was intermediate. BEST remains3374661.4167660830412778e-9. Anonymous board refreshed06:08Z unchanged: J2W1.416013781559781e-9,oliver1.3386340705735589e-9,marius1.2888878435691834e-9. Exactly2UTC21 attempt+accepted-IDreceipts(337466/337473);8remain,failed473consumedone. Latest user policy(relayedroot): afterfive continue banking qualified meaningful improvements within final10; third repairedruntime/J2Wmargin,fourth<1.41e-9,fifth<1.40e-9;rootonlyfires.
    #337473 FINAL GRADED penalized0.014044704624470581,99completed/1failed(index27 residual0.401729281),all7 finalsetup executionsSUCCESS. It is NOT a cleanbank. BEST337466 remains1.4167660830412778e-9. Board11:08p.m. unchanged J2W1.416013781559781e-9,oliver1.3386340705735589e-9. Do not promote LASTFIRED or retry its archive.

Public score: 0.014044704624470581. Status: GRADED. Failed networks: 1.

[Run or rebuild the designated estimators](../../estimator/REPRODUCE.md).

## Programs

- [estimator.py](estimator.py)

## Evidence

- [SUBMISSION_LEDGER.md#L1525](../../records/SUBMISSION_LEDGER.md#L1525)
- [SUBMISSION_LEDGER.md#L1535](../../records/SUBMISSION_LEDGER.md#L1535)

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

- ← shipped in: [Hidden complement readout adds .001660192% FLOPs and improves both host judges](../../experiments/learned-correction/hidden-complement-readout-adds-001660192-flops-improves-both-host/README.md)
- ← shipped in: [Width-eight edge plus hidden complement: -.191409749% Mac32 incremental over #337466](../../experiments/learned-correction/width-eight-edge-plus-hidden-complement-191409749-mac32-incremental/README.md)

[Research graph](../../graph/index.html#node=submission-337473) · [Copy provenance](../../PROVENANCE.json)

