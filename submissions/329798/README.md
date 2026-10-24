# Submission #329798: radial closure with split reads lost its expected compute saving to a remote dtype bug

2026-09-04T23:11:57Z · submission · rejected · measured

RemoteArray.dtype returned a string while the helper compared it with a dtype-label object, so every remote product selected ordinary matrix multiplication. Comparable raw accuracy therefore produced a worse adjusted score than #329790.

Recorded measurement:

    Official adjusted 1.747652393096604e-8; raw 3.17094401225404e-8; implied utilization about .551 versus local .341.

Public score: 1.747652393096604e-08. Status: GRADED. Failed networks: 0.

[Run or rebuild the designated estimators](../../estimator/REPRODUCE.md).

## Programs

- [estimator.py](estimator.py)
- [../../experiments/flop-accounting/128-state-recurrent-hermite-observer-rank-1536-carry-protected/code/estimator.py](../../experiments/flop-accounting/128-state-recurrent-hermite-observer-rank-1536-carry-protected/code/estimator.py)
- [candidate_radial_gnn_remote_dtype_v2.py](candidate_radial_gnn_remote_dtype_v2.py)

## Evidence

- [SUBMISSION_329798.md#L19](SUBMISSION_329798.md#L19)
- [REMOTE_DTYPE_BUG_2026_09_04.md#L11](REMOTE_DTYPE_BUG_2026_09_04.md#L11)

<details>
<summary>All result files and records (3)</summary>

- [SUBMISSION_329798.md](SUBMISSION_329798.md) · record
- [REMOTE_DTYPE_BUG_2026_09_04.md](REMOTE_DTYPE_BUG_2026_09_04.md) · record
- [SUBMISSION_LEDGER.md](../../records/SUBMISSION_LEDGER.md) · record

</details>

[Research graph](../../graph/index.html#node=submission-329798) · [Copy provenance](../../PROVENANCE.json)

