# Exact compute-copy and retained-tree changes measured: -0.0533768% to -0.1328245% FLOPs

2026-10-13T23:55Z · decision · adopted · derived

Four exact-code cuts certified on four paired networks each (all 16 prediction bytes and MSEs identical, counted credit): the compute-copy went into the candidate generation 56 (S56) stack's extras; UBcluster, bank-copy and the quarter-load suffix were held for the successor stack.

Recorded measurement:

    compute-copy 4 networks mean bill -0.0533768%, +330 requests; UBcluster mean -0.0916766%, +100 requests; bank9 -0.105245%; bank + quarter -0.1328245% combined

## Programs

- [fragment_astra_ubcluster_v1.py](fragment_astra_ubcluster_v1.py)
- [candidate.py](candidate.py)
- [../../../submissions/335782/candidate-db2d5075.py](../../../submissions/335782/candidate-db2d5075.py)
- [check_astra_compute_join_v1.py](check_astra_compute_join_v1.py)
- [check_npx_parent_v1.py](check_npx_parent_v1.py)
- [../../../experiments/join-sketch/rank-redistribution-rejected-restoring-layer-10-rank-384-gives/code/opus_reexam_20261001_v1/scout_native/fragment_astra_compute_v1.py](../../../experiments/join-sketch/rank-redistribution-rejected-restoring-layer-10-rank-384-gives/code/opus_reexam_20261001_v1/scout_native/fragment_astra_compute_v1.py)
- [../../../experiments/join-sketch/rank-redistribution-rejected-restoring-layer-10-rank-384-gives/code/opus_reexam_20261001_v1/scout_native/fragment_npx_v1.py](../../../experiments/join-sketch/rank-redistribution-rejected-restoring-layer-10-rank-384-gives/code/opus_reexam_20261001_v1/scout_native/fragment_npx_v1.py)

## Evidence

- [TEAM_LOG_2026-10-01_to_10-22.md#L2252](../../TEAM_LOG_2026-10-01_to_10-22.md#L2252)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2254](../../TEAM_LOG_2026-10-01_to_10-22.md#L2254)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2255](../../TEAM_LOG_2026-10-01_to_10-22.md#L2255)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2262](../../TEAM_LOG_2026-10-01_to_10-22.md#L2262)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2263](../../TEAM_LOG_2026-10-01_to_10-22.md#L2263)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2277](../../TEAM_LOG_2026-10-01_to_10-22.md#L2277)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2301](../../TEAM_LOG_2026-10-01_to_10-22.md#L2301)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2321](../../TEAM_LOG_2026-10-01_to_10-22.md#L2321)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2323](../../TEAM_LOG_2026-10-01_to_10-22.md#L2323)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2324](../../TEAM_LOG_2026-10-01_to_10-22.md#L2324)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2407](../../TEAM_LOG_2026-10-01_to_10-22.md#L2407)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2412](../../TEAM_LOG_2026-10-01_to_10-22.md#L2412)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2421](../../TEAM_LOG_2026-10-01_to_10-22.md#L2421)
- [TEAM_LOG_2026-10-01_to_10-22.md#L2425](../../TEAM_LOG_2026-10-01_to_10-22.md#L2425)

<details>
<summary>All result files and records (7)</summary>

- [astra_compute_v1_arm136_comparison.json](astra_compute_v1_arm136_comparison.json) · result
- [summary.json](summary.json) · result
- [ASTRA_COMPLETION_RECEIPT_v1.json](ASTRA_COMPLETION_RECEIPT_v1.json) · record
- [summary-29c1e0eb.json](summary-29c1e0eb.json) · result
- [summary-9814c326.json](summary-9814c326.json) · result
- [ASTRA_S56_AMEND_RECEIPT_20261013_v1.json](ASTRA_S56_AMEND_RECEIPT_20261013_v1.json) · record
- [TEAM_LOG_2026-10-01_to_10-22.md](../../TEAM_LOG_2026-10-01_to_10-22.md) · record

</details>

## Related work

- enables: [Refit with absorbed late-layer quartic terms accepted; combined candidate seals at -0.2155%](../2026-10-13-refit-absorbed-late-layer-quartic-terms-accepted-combined-candidate/README.md)
- ← enables: [Reassociated push/read algebra adopted: -0.195% FLOPs at rounding-level prediction change](../2026-10-13-reassociated-push-read-algebra-adopted-0-195-flops-rounding/README.md)

[Research graph](../../../graph/index.html#node=2026-10-13-exact-compute-copy-retained-tree-changes-measured-0-0533768) · [Copy provenance](../../../PROVENANCE.json)

