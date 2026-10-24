# Same-rank join audit: the within-signed-pair tensor mode-1 Gram is the closed pair-mode kernel (+0.229% on 8)

2026-10-16T23:35:17Z · finding · informational · derived

A derivation audit showed the proposed same-rank join variants duplicate measured history: the energy ranking lam^2 ||J_col||^4 is the old native K3W4 arm (+0.385% MSE, SE .19, 3/8), the pair-aware diagonal was null, and the exact within-signed-pair tensor mode-1 Gram including the negative dot^2 cross term is the earlier kernel_pair_mode1 candidate 88cd8151 (native pilot 8: MSE +0.160648%, bill +0.068045%, adjusted +0.228802%). No cheap exact route to the full tensor Gram with cross-pair terms exists.

Recorded measurement:

    K3W4 +.385% MSE (SE .19, 3/8); T120 pair-aware diagonal -.147% pooled (SE .352, 8/15); pair mode1 pilot 8: MSE +.160648%, bill +.068045%, adjusted +.228802%, 4/8, +411 ops.

## Programs

- [candidate.py](candidate.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L4530](../../TEAM_LOG_2026-10-01_to_10-22.md#L4530)

<details>
<summary>All result files and records (2)</summary>

- [DUPLICATE_AND_DERIVATION_v1.json](DUPLICATE_AND_DERIVATION_v1.json) · record
- [TEAM_LOG_2026-10-01_to_10-22.md](../../TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

[Research graph](../../../graph/index.html#node=2026-10-16-same-rank-join-audit-within-signed-pair-tensor-mode) · [Copy provenance](../../../PROVENANCE.json)

