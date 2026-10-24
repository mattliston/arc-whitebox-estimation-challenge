# Submission #337466: Edge width 8 with per-layer correction fitted on 90 networks

2026-10-21T04:12:34Z · submission · adopted · measured

The edge-width-eight source used a correction fitted on 90 training networks with per-layer feature geometry. It improved the public score while leaving only a small maximum residual margin.

Recorded measurement:

    Official submission record: {"submission_id": 337466, "date": "2026-10-21T04:12:34Z", "status": "GRADED", "score_public": 1.4167660830412778e-09, "n": 100, "failed_networks": 0, "utilization": 0.10817131545498342, "wall_mean_s": 101.7, "wall_max_s": 113.448, "resid_mean_s": 0.364, "resid_max_s": 0.397, "source_sha256": "660bc2d6daa5950c8614ca505775d305d3e19031a7a57eb0ac815b90ab5b8200", "archive_sha256": "24216de79706eabd70deb923083a3b9a51b0aede9856d1d424d8c8de6e534c2c"}
    Ledger: #337466 final GRADED at04:55:54Z (9:55 p.m. Pacific): score1.4167660830412778e-9, actual-.413553051% versus422,100/100 complete,zero failures/exhausted,five setupsSUCCESS. Root independently verified raw5d5cb926 and promoted BEST337466 at05:02:22Z; pointer9597222f, promotionreceiptbaae1310. All NEW gains now versus actual434b/660bc on same host. J2W lastverified1.416013781559781e-9 needs only-.0530999076% more; ideal1.41 needs-.4775723475%. LASTFIRED remains337466/archive24216de7. Nine submissions remain; onlyone UTC21 attempt verified by root.
    #337473 is final GRADED with penalized score0.014044704624470581,99completed/1failed(index27 brandy-johnston RESIDUAL_WALL_TIME_EXHAUSTED0.401729281s); final observed05:56:32Z/rawSHAd9153627. All7 finalsetuprecordsSUCCESS; earlier5setupcount was intermediate. BEST remains3374661.4167660830412778e-9. Anonymous board refreshed06:08Z unchanged: J2W1.416013781559781e-9,oliver1.3386340705735589e-9,marius1.2888878435691834e-9. Exactly2UTC21 attempt+accepted-IDreceipts(337466/337473);8remain,failed473consumedone. Latest user policy(relayedroot): afterfive continue banking qualified meaningful improvements within final10; third repairedruntime/J2Wmargin,fourth<1.41e-9,fifth<1.40e-9;rootonlyfires.
    #337473 FINAL GRADED penalized0.014044704624470581,99completed/1failed(index27 residual0.401729281),all7 finalsetup executionsSUCCESS. It is NOT a cleanbank. BEST337466 remains1.4167660830412778e-9. Board11:08p.m. unchanged J2W1.416013781559781e-9,oliver1.3386340705735589e-9. Do not promote LASTFIRED or retry its archive.

Public score: 1.4167660830412778e-09. Status: GRADED. Failed networks: 0.

[Run or rebuild the designated estimators](../../estimator/REPRODUCE.md).

## Programs

- [estimator.py](estimator.py)

## Evidence

- [SUBMISSION_LEDGER.md#L1459](../../records/SUBMISSION_LEDGER.md#L1459)
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

- ← shipped in: [Learned edge hidden width eight: -.150665691% Mac and -.2338808% fresh adjusted versus #337422](../../experiments/learned-correction/learned-edge-hidden-width-eight-150665691-mac-2338808-fresh/README.md)
- ← shipped in: [Per-layer readout fitted on 90 networks: -.371029209% Mac32 and -.235745959% Linux32](../../experiments/learned-correction/per-layer-readout-fitted-90-networks-371029209-mac32-235745959/README.md)
- ← shipped in: [Width-eight edge plus per-layer 90-network readout: -.533265988% Mac32 and -.441676584% fresh](../../experiments/learned-correction/width-eight-edge-plus-per-layer-90-network-readout/README.md)

[Research graph](../../graph/index.html#node=submission-337466) · [Copy provenance](../../PROVENANCE.json)

