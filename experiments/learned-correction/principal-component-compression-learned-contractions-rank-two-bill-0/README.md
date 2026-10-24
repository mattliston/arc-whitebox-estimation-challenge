# Principal-component compression of learned contractions to rank two: bill -0.31%, fresh-16 adjusted -0.245%

2026-10-14T18:25:41Z · experiment · inconclusive · measured

Compressing all nonnegative contractions of the learned correction network to a rank-2 energy-PCA basis (AEP2) saved about 0.3% of bill with a small two-network accuracy loss and passed the 16-network fresh screen. Its out-of-sample run was held as a fallback while its successors (GNR2, then the GOP2 output projection) were tested and won.

Recorded measurement:

    native price: full bill -0.313292370% / -0.299451020%; two-network diagnostic geometric adjusted -0.065471606%; fresh 16 PASS -0.244601% log; OOS held (one 8-call shard run, never paired)

Program source is in [`code/`](code/), preserving the relative layout below `pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/astra_energy_pca_rescue_v1`. Original machine paths and data requirements remain in the archived programs. [Run the designated estimators](../../../estimator/REPRODUCE.md); [research data availability](../../../data/DATA_INDEX.md).

## Programs

- [code/fragment_energy_pca_UNVALIDATED_v1.py](code/fragment_energy_pca_UNVALIDATED_v1.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L2941](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2941)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2945](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2945)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2947](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2947)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2949](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2949)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2968](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L2968)
- [TEAM_LOG_2026-10-01_to_10-22.md#L3043](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3043)
- [TEAM_LOG_2026-10-01_to_10-22.md#L3045](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3045)
- [TEAM_LOG_2026-10-01_to_10-22.md#L3055](../../../records/TEAM_LOG_2026-10-01_to_10-22.md#L3055)

<details>
<summary>All result files and records (3)</summary>

- [AEP2_NATIVE_FRESH_DECISION_v1.json](AEP2_NATIVE_FRESH_DECISION_v1.json) · result
- [ASTRA_FRESH_RETIREMENT_v1.json](ASTRA_FRESH_RETIREMENT_v1.json) · record
- [TEAM_LOG_2026-10-01_to_10-22.md](../../../records/TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

## Related work

- enables: [Rank-two learned-message compression with a shared second message: two-network adjusted -0.29%](../rank-two-learned-message-compression-shared-second-message-two/README.md)

[Research graph](../../../graph/index.html#node=2026-10-14-principal-component-compression-learned-contractions-rank-two-bill-0) · [Copy provenance](../../../PROVENANCE.json)

