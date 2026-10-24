# Old-plane matrix is low rank, but per-cohort symmetric tensor fits miss 22–51% of the plane

2026-09-22T20:48Z · finding · informational · measured

A dump of the bank at layer 14 (22,784 columns, 17,728 old) shows the old plane growing from 25% of the consumed (2,1) plane at layer 6 to 60% at layer 14 with cosine 0.3-0.4 to the young plane; as a matrix it is 73% rank-1, 98% rank-64 and 99.99% rank-256. A symmetric CP-ALS of each old cohort in its own frame leaves 0.43-0.70 relative plane error at 64 terms and 0.22-0.51 at 256, so column-count compression of cohort tensors is dead (matching the public chain's F76 result).

Recorded measurement:

    old plane norm / total plane 25% (L6), 45% (L9), 57% (L12), 60% (L14); matrix energy 73% rank-1, 89% rank-16, 98% rank-64, 99.99% rank-256; cohort column rank(99.9%) 150-190 at L14; CP-ALS rel error at K'=1/4/16/64/256 = 0.93-0.97 / 0.81-0.93 / 0.62-0.85 / 0.43-0.70 / 0.22-0.51

## Programs

- [dump_bank.py](dump_bank.py)
- [analyze_bank.py](analyze_bank.py)

## Evidence

- [NOTES.md#L24](../../../experiments/third-order-planes/removing-cohorts-aged-least-eleven-raises-raw-error-6685/NOTES.md#L24)

<details>
<summary>All result files and records (2)</summary>

- [analyze_bank_r35.log](analyze_bank_r35.log) · log
- [NOTES.md](../../../experiments/third-order-planes/removing-cohorts-aged-least-eleven-raises-raw-error-6685/NOTES.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-22-old-plane-matrix-low-rank-but-per-cohort-symmetric) · [Copy provenance](../../../PROVENANCE.json)

