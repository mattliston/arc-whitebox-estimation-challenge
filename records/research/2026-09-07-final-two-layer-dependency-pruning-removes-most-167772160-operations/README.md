# Final-two-layer dependency pruning removes at most 167772160 operations, less than 0.035% of the total bill

2026-09-07 · finding · rejected · derived

A circuit audit of the current bank candidate (SHA 3b7f55f9) at post-layer-13 and post-layer-14 found every dominant product at pre-14 and pre-15 has a live consumer (head-14 plane statistics, final diagonal covariance query, final cubic sums), so no GEMM becomes dead with two nonlinear layers left. Four exact eliminations (unused newest-factor p1/p2 replicas, a diagonal-only residual temporary, an overwritten identity-metric radial calculation, the unread rho-power update after r=6) are bounded by 144 n^2 + 16384 n charged operations, far below the roughly 20% whole-card change needed for an unchanged-raw sub-4e-9 bank.

Recorded measurement:

    Envelope 167,772,160 charged operations at n=1024, below 0.035% of the archived full bills 490,466,018,197 (rank 35) and 489,374,517,341 (rank 100).

## Evidence

- [LAST_TWO_LAYER_DEPENDENCY_PRUNING_SOURCE_AUDIT_V1.md#L10](LAST_TWO_LAYER_DEPENDENCY_PRUNING_SOURCE_AUDIT_V1.md#L10)
- [LAST_TWO_LAYER_DEPENDENCY_PRUNING_SOURCE_AUDIT_V1.md#L133](LAST_TWO_LAYER_DEPENDENCY_PRUNING_SOURCE_AUDIT_V1.md#L133)

<details>
<summary>All result files and records (1)</summary>

- [LAST_TWO_LAYER_DEPENDENCY_PRUNING_SOURCE_AUDIT_V1.md](LAST_TWO_LAYER_DEPENDENCY_PRUNING_SOURCE_AUDIT_V1.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-09-07-final-two-layer-dependency-pruning-removes-most-167772160-operations) · [Copy provenance](../../../PROVENANCE.json)

