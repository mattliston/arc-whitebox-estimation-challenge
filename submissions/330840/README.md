# Submission #330840: shared rank-384 old basis and corrected scalar gains failed every attempted network

2026-09-13T00:26:29Z · submission · failed · measured

The shared old-tier rank-384 candidate with corrected scalar gain semantics was submitted with SHA256 c2c2ffada6feb16c08eb820c9401be68d5e8e7fe12d4034a20eed15af849ccea. The immediate status API reported failure without per-network causes; the subsequent wall-repair record established that every attempted network timed out. Five local production walls of 54.603..57.851s had therefore not established hosted reliability.

Recorded measurement:

    Accepted 2026-09-13T00:26:29.018Z; official status failed with score=null; local five-call wall54.603..57.851s versus #330390 local34.594..43.693s.

Public score: None. Status: FAILED. Failed networks: 17.

[Run or rebuild the designated estimators](../../estimator/REPRODUCE.md).

## Programs

- [estimator.py](estimator.py)

## Evidence

- [FAILED_330840_20260912_PDT.md#L3](FAILED_330840_20260912_PDT.md#L3)
- [RECEIPTS.md#L4](../../records/research/2026-09-13-batched-six-level-factor-propagation-pair-series-fusion-split/RECEIPTS.md#L4)
- [SUBMISSION_LEDGER.md#L705](../../records/SUBMISSION_LEDGER.md#L705)

<details>
<summary>All result files and records (4)</summary>

- [FAILED_330840_20260912_PDT.md](FAILED_330840_20260912_PDT.md) · record
- [submit_result.json](submit_result.json) · record
- [RECEIPTS.md](../../records/research/2026-09-13-batched-six-level-factor-propagation-pair-series-fusion-split/RECEIPTS.md) · record
- [SUBMISSION_LEDGER.md](../../records/SUBMISSION_LEDGER.md) · record

</details>

## Related work

- ← enables: [Shared rank-384 basis for cubic cohorts older than four layers: conditional bill ratio 0.8968](../../records/research/2026-09-12-shared-rank-384-basis-cubic-cohorts-older-than-four/README.md)
- ← shipped in: [Shared rank-384 basis for cubic cohorts older than four layers: conditional bill ratio 0.8968](../../records/research/2026-09-12-shared-rank-384-basis-cubic-cohorts-older-than-four/README.md)
- ← shipped in: [Shared rank-384 old basis on 40 networks: adjusted ratio 0.9191502236702563, 40/40 improve](../../experiments/join-sketch/shared-rank-384-old-basis-40-networks-adjusted-ratio/README.md)
- ← enables: [Shared rank-384 old basis: five-call walls 54.3 to 59.5 s, residual 0.26 to 0.29 s, exact repeat](../../experiments/wall-and-residual/shared-rank-384-old-basis-five-call-walls-54/README.md)
- ← shipped in: [Shared rank-384 old basis on four networks: adjusted ratio 0.9249351388935633, all four improve](../../experiments/join-sketch/shared-rank-384-old-basis-four-networks-adjusted-ratio/README.md)
- ← shipped in: [Shared rank384 old-factor basis on network 37 improves adjusted error: ratio 0.905199269759271](../../experiments/join-sketch/shared-rank384-old-factor-basis-network-37-improves-adjusted/README.md)

[Research graph](../../graph/index.html#node=submission-330840) · [Copy provenance](../../PROVENANCE.json)

