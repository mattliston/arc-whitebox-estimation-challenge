# Setup-bound super-views fail: strided tree operations add 8-59% CPU for little binding saving

2026-09-25T15:59Z · experiment · rejected · measured

Free slicing from larger prebound trees creates gaps in the level-operation layout. Held-out shape coverage was too low to justify the measured strided-operation cost.

Code availability: Only supporting lazy-binding census code was recovered. The bespoke gapped-tree CPU microbenchmark and super-view selection program behind the reported +8–59% CPU and 0.11% held-out bill saving were not found as retained scripts.

Recorded measurement:

    - 2026-09-25 08:59 lead: FREE BINDS (b) CLOSED (engineer): getitem views of setup-bound super views leave gaps between tree blocks (block stride = super rows x c). Level ops +8..+59% CPU (the strided-view law); first-time lazy binds (~0.6%/net) are spread over ~170 net-specific (kind, depth, width) keys; the best 20 super views remove only 0.11% of a held-out net's bill. The lazy-bind lever is the cache-cap raise (a) only (bit-identical, CPU-free, removes the rebind half).

The measured result is retained here. A separate executable program was not identified; this entry is indexed as a result record rather than a code directory.

## Programs

- [e75_lazybind_census_v2.py](e75_lazybind_census_v2.py)

## Evidence

- [TEAM_LOG_2026-09-22_to_09-26.md#L1037](../../TEAM_LOG_2026-09-22_to_09-26.md#L1037)

<details>
<summary>All result files and records (1)</summary>

- [TEAM_LOG_2026-09-22_to_09-26.md](../../TEAM_LOG_2026-09-22_to_09-26.md) · log

</details>

[Research graph](../../../graph/index.html#node=2026-09-25-setup-bound-super-views-fail-strided-tree-operations-add) · [Copy provenance](../../../PROVENANCE.json)

